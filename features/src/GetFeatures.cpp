#include "GetFeatures.hpp"



/*===================================================================
* Function: GetMP
*
* Summary:
*   Get Morphological profiles (MP) composed of opening-closing by reconstruction
*
* Arguments:
*   vector<Mat>& features
*   vector<unsigned char>& classValue
*   const array<int,3> & morph_size - the diameter of circular structureing element,default {1,2,3} 
*
* Returns:
*   void
=====================================================================
*/
void cvFeatures::GetMP(vector<Mat>& features, vector<unsigned char>& classValue, const array<int,3> & morph_size) {

    Mat src = image.clone();
    normalize(src, src, 0, 255, NORM_MINMAX);
    src.convertTo(src, CV_8UC3);
    vector<Mat> channels;
    split(src, channels);

    for (auto& dstChannel : channels) {
        for (int i = 0; i < morph_size.size(); i++) {
            Mat result = mp::CaculateMP(dstChannel, morph_size[i]);
            features.push_back(result);
            classValue.push_back(class_type);
        }
    }
}


/*===================================================================
* Function: GetLBP
*
* Summary:
*   Get local binary pattern 
*
* Arguments:
*   vector<Mat>& features
*   vector<unsigned char>& classValue
*   int radius  - default 1
*   int neighbors - default 8
*   int histsize - length of the feature vector,default 32
*
* Returns:
*   void
=====================================================================
*/
void cvFeatures::GetLBP(vector<Mat> &features, vector<unsigned char> &classValue, int radius, int neighbors, int histsize) {
     
        Mat lbp = elbp::CaculateElbp(this->image, radius, neighbors, true);
            // Apply mask
        Mat mask = Mat();
        Mat lbp_hist = GetHistOfMaskArea(lbp, mask, 0, 255, histsize, true);
        features.push_back(lbp_hist);
         classValue.push_back(class_type);
}


/*===================================================================
* Function: GetGLCM
*
* Summary:
*   Calculate energy, contrast, homogenity and entropy of all channels
*
*
* Arguments:
*   vector<Mat>& features  - Destination  
*   vector<unsigned char>& classValue - Destination 
*   int size - size of Mat Window (only support 5*5, 7*7)
*   GrayLevel level - Destination image's Gray Level (choose in 4/8/16/32)
*   int histsize - length of the feature vector,default 32
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



/*===================================================================
* Function: GetMPEG7DCD
*
* Summary:
*   Get MPEG-7 dominant color descriptor (DCD)
*   include three color components and the weight of the most dominant color
*
* Arguments:
*   vector<Mat>& features  - Destination
*   vector<unsigned char>& classValue - Destination
*   int numOfColor - default 3
*
* Returns:
*   void
=====================================================================
*/
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

/*===================================================================
* Function: GetMPEG7CSD
*
* Summary:
*   Get MPEG-7 color structure descriptor (CSD)
*   include three color components and the weight of the most dominant color
*
* Arguments:
*   vector<Mat>& features  - Destination
*   vector<unsigned char>& classValue - Destination
*   int Size - length of the feature vector,default 32
*
* Returns:
*   void
=====================================================================
*/
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

/*===================================================================
 * Function: GetStatistic
 *
 * Summary:
 *   Compute min, max, mean, std, median of mask area
 *
 * Arguments:
 *   Mat src -  PolSAR data single channel
 *
 * Returns:
 *   void
=====================================================================
*/
void cvFeatures::GetStatistic(vector<Mat>& features, vector<unsigned char>& classValue) {

    for (int i = 0; i < image.channels(); i++) {

        Mat image_temp;
        extractChannel(image, image_temp, i);
        Mat result = Mat(1, 5, CV_32FC1);

        vector<float> vec(image_temp.begin<float>(), image_temp.end<float>());
        int size = static_cast<int>(image_temp.total());
        // sort the vector
        std::sort(vec.begin(), vec.end());
        if (size % 2 == 0)
        {
            //median
            result.at<float>(0, 4) = (vec[size / 2.0 - 1] + vec[size / 2]) / 2;
        }
        else
        {
            //median
            result.at<float>(0, 4) = vec[size / 2];
        }

        //min
        result.at<float>(0, 0) = vec[0];
        //max
        result.at<float>(0, 1) = vec[size - 1];

        Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
        meanStdDev(vec, mean, stddev);
        //mean
        result.at<float>(0, 2) = mean[0];
        //stddev
        result.at<float>(0, 3) = stddev[0];

        features.push_back(result);
        classValue.push_back(class_type);
    }
}
 
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




