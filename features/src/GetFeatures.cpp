#include "GetFeatures.hpp"

// the structure element radius for MP features
std::array<int, 3> morph_size = { 1,3,5 };
/*===================================================================
 * Function: GetHistOfMaskArea
 *
 * Summary:
 *   Caculate the historgram vector of a Mat with mask
 *
 * Arguments:
 *   Mat & src - IGBP Mat or LCCS Mat
 *   const Mat& mask -  single channel mask Matrix
 *   int minVal - the min of bin boundaries
 *   int maxVal - the max of bin boundaries
 *   int histSize
 *   bool normed - normalized to make the sum become 1
 * Returns:
 *  Mat of Size(1,histSize)
=====================================================================
*/
Mat cvFeatures::GetHistOfMaskArea(const Mat& src, const Mat& mask, int minVal, int maxVal, int histSize, bool normed)
{
    Mat result;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
    const float* histRange = { range };
    // calc histogram with mask
    calcHist(&src, 1, 0, mask, result, 1, &histSize, &histRange, true, false);
    // normalize
    if (normed) {
        result /= (int)src.total();
    }
    return result.reshape(1, 1);
}

void cvFeatures::GetLBP(vector<Mat> &features, vector<unsigned char> &classValue, int radius, int neighbors, int histsize) {
     
        Mat lbp = elbp::CaculateElbp(image, radius, neighbors, true);
            // Apply mask
        Mat mask = Mat();
        Mat lbp_hist = GetHistOfMaskArea(lbp, mask, 0, 255, histsize, true);
        features.push_back(lbp_hist);
         classValue.push_back(class_type);
}


/*===================================================================
* Function: GetFeatureGLCM
*
* Summary:
*   Calculate energy, contrast, homogenity and entropy of the mask area
*    output the result and class labels.
*
*
* Arguments:
*   vector<Mat>& features
*   vector<unsigned char>& classValue
*   int size - size of Mat Window (only support 5*5, 7*7)
*   GrayLevel level - Destination image's Gray Level (choose in 4/8/16/32)
*   int histsize - length of the feature vector
*
* Returns:
*   void
=====================================================================
*/
void cvFeatures::GetGLCM(vector<Mat> &features, vector<unsigned char> &classValue,int winsize, GrayLevel level, int histsize) {
     
    Mat src = image.clone();
        // src should be nomalized to color images
        normalize(src, src, 0, 255, NORM_MINMAX);
        src.convertTo(src, CV_8UC3);
        
        vector<Mat> channels;
        split(src, channels);

        for (auto& dstChannel : channels) {
            // Magnitude Gray Image
            Mat result;
            GLCM::GrayMagnitude(dstChannel, dstChannel, level);
            // Calculate Energy, Contrast, Homogenity, Entropy of the whole Image
            Mat Energy_tmp, Contrast_tmp, Homogenity_tmp, Entropy_tmp;
            GLCM::CalcuTextureImages(dstChannel, Energy_tmp, Contrast_tmp, Homogenity_tmp, Entropy_tmp, winsize, level, true);

            Mat tmp;
            Mat Energy_hist = GetHistOfMaskArea(Energy_tmp, Mat(), 0, 255, histsize, true);
            Mat Contrast_hist = GetHistOfMaskArea(Contrast_tmp, Mat(), 0, 255, histsize, true);
            Mat Homogenity_hist = GetHistOfMaskArea(Homogenity_tmp, Mat(), 0, 255, histsize, true);
            Mat Entropy_hist = GetHistOfMaskArea(Entropy_tmp, Mat(), 0, 255, histsize, true);
            tmp.push_back(Energy_hist);
            tmp.push_back(Contrast_hist);
            tmp.push_back(Homogenity_hist);
            tmp.push_back(Entropy_hist);
            vconcat(tmp, result);
            features.push_back(result);
            classValue.push_back(class_type);
        }
 }






// numOfColor = 3
void cvFeatures::GetMPEG7DCD(vector<Mat>& features, vector<unsigned char>& classValue, int numOfColor) {
    Mat src = image.clone();
    Mat dst;
    normalize(src, dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8UC3);
    Frame* frame = new Frame(dst.cols, dst.rows, true, true, true);
    frame->setImage(dst);
    // color: DCD, return the weights and color value of each dominant color
    XM::DominantColorDescriptor* dcd = Feature::getDominantColorD(frame, false, false, false);
    // number of dominant colors
    int ndc = dcd->GetDominantColorsNumber();
    if (ndc >= numOfColor) { 
        XM::DOMCOL* domcol = dcd->GetDominantColors();
        float weight = 0.0;
        // normalize the weight
        for (int w = 0; w < ndc; w++) {
            weight = weight + domcol[w].m_Percentage;
        }
        Mat feature_temp;
        for (int w = 0; w < numOfColor; w++) {
            feature_temp.push_back(int(domcol[w].m_Percentage / weight));
            feature_temp.push_back(domcol[w].m_ColorValue[0]);
            feature_temp.push_back(domcol[w].m_ColorValue[1]);
            feature_temp.push_back(domcol[w].m_ColorValue[2]);
        }
        features.push_back(feature_temp);
        classValue.push_back(class_type);
        // release descriptor
        delete dcd;
    }
 }


void cvFeatures::GetMPEG7CSD(vector<Mat>& features, vector<unsigned char>& classValue, int Size) {
    Mat src = image.clone();
    Mat dst;
    normalize(src, dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8UC3);
    Frame* frame = new Frame(dst.cols, dst.rows, true, true, true);
    frame->setImage(dst);
    // compute the descriptor
    XM::ColorStructureDescriptor* csd = Feature::getColorStructureD(frame, Size);

    Mat feature_temp;
    for (unsigned int i = 0; i < csd->GetSize(); i++) {
        feature_temp.push_back((int)csd->GetElement(i));
    }
    features.push_back(feature_temp);
    classValue.push_back(class_type);
    // release descriptor
    delete csd;
 }

auto* cvFeatures::matToArray(Mat& image) {
    if (image.channels() != 1) {
        cvtColor(image, image, COLOR_BGR2GRAY);
    }
    // flatten the Mat.
    uint totalElements = static_cast<int>(image.total()) * image.channels();
    Mat flat = image.reshape(1, totalElements);
    if (!image.isContinuous()) {
        flat = flat.clone();
    }
    auto* ptr = flat.data;
    return ptr;
}

// compute opening-closing by reconstruction from image
// example:https://de.Mathworks.com/help/images/marker-controlled-watershed-segmentation.html
Mat cvFeatures::CaculateMP(const Mat& src, int morph_size) {
    //convert img to grayscale
    Mat dst;
    if (src.channels() != 1) {
        cvtColor(src, dst, COLOR_BGR2GRAY);
    }
    else {
        src.copyTo(dst);
    }
    equalizeHist(dst, dst);
    //imshow("image", dst);
    // waitKey(0);

    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

    
    //erode and reconstruct ( opening-by-reconstruction )
    Mat Iobr = Mat(Size(dst.size()), dst.type());
    erode(dst, Iobr, element);
    auto* ptr = matToArray(Iobr);
    mp::Reconstruct(ptr, matToArray(dst), dst.cols, dst.rows);
    //restore cv Mat
    Mat dst2 = Mat(dst.rows, dst.cols, dst.type(), ptr);
    //imshow("openning by reconstruction: ", dst2);
    //waitKey(0);

     //dilate and reconstruct (closing-by-Reconstruction)
    Mat Icbr = Mat(Size(dst2.size()), dst2.type());
    dilate(dst2, Icbr, element);
    // imcomplement
    dst2 = 255 - dst2;
    Icbr = 255 - Icbr;
    auto* ptr2 = matToArray(Icbr);
    mp::Reconstruct(ptr2, matToArray(dst2), dst2.cols, dst2.rows);
    //restore cv Mat
    Mat dst3 = Mat(dst.rows, dst.cols, dst.type(), ptr2);
    // imcomplement
    dst3 = 255 - dst3;
    //imshow("opening-closing by reconstruction: ", dst3);
    //waitKey(0);
    return dst3;
}

void cvFeatures::GetMP(vector<Mat>& features, vector<unsigned char>& classValue) {
    Mat src = image.clone();
    normalize(src, src, 0, 255, NORM_MINMAX);
    src.convertTo(src, CV_8UC3);
    vector<Mat> channels;
    split(src, channels);

    for (auto& dstChannel : channels) {
        for (int i = 0; i < morph_size.size(); i++) {
            Mat result = CaculateMP(dstChannel, morph_size[i]);
            features.push_back(result);
            classValue.push_back(class_type);
        }
    }
}
        




