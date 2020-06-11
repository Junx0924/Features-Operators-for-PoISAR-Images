#include "cvFeatures.hpp"



/*===================================================================
* Function: GetMP
*
* Summary:
*   Get Morphological profiles (MP) composed of opening-closing by reconstruction
*
* Arguments:
*   const Mat &src - grayscale image
*   const array<int,3> & morph_size - the diameter of circular structureing element,default {1,2,3} 
*
* Returns:
*   Matrix of Size(src.rows*3, src.cols) - 3 stacked mp profiles
=====================================================================
*/
Mat cvFeatures::GetMP(const Mat &src, const array<int,3> & morph_size) {
    Mat output;
    Mat dst = src.clone();
    normalize(dst, dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8UC1);

    for (int i = 0; i < morph_size.size(); i++) {
        Mat result = mp::CaculateMP(dst, morph_size[i]);
        output.push_back(result);
    }
    return output;
}


/*===================================================================
* Function: GetLBP
*
* Summary:
*   Get local binary pattern 
*
* Arguments:
*   const Mat& src -  grayscale
*   int radius  - default 1
*   int neighbors - default 8
*   int histsize - length of the feature vector,default 32
*
* Returns:
*   Matrix of Size( 1, histsize)
=====================================================================
*/
Mat cvFeatures::GetLBP(const Mat& src, int radius, int neighbors, int histsize) {
     
        Mat lbp = elbp::CaculateElbp(src, radius, neighbors, true);
            // Apply mask
        Mat mask = Mat();
        Mat lbp_hist = GetHistOfMaskArea(lbp, mask, 0, 255, histsize, true);
        return lbp_hist;
}


/*===================================================================
* Function: GetGLCM
*
* Summary:
*   Calculate energy, contrast, homogenity and entropy of all channels
*
*
* Arguments:
*   const Mat& src -  grayscale
*   int winSize - size of Mat Window (only support 5*5, 7*7),default 7
*   GrayLevel level - Destination image's Gray Level (choose in 4/8/16/32),default 16
*   int histsize - length of the feature vector,default 32
*
* Returns:
*   Matrix of Size( 1, histsize)
=====================================================================
*/
Mat cvFeatures::GetGLCM(const Mat& src,int winsize, GrayLevel level, int histsize) {
     
    Mat dst = src.clone();
    // src should be nomalized to color images
    normalize(dst, dst, 0, 255, NORM_MINMAX);
    src.convertTo(dst, CV_8UC1);

    // Magnitude Gray Image
    Mat result;
    GLCM::GrayMagnitude(dst, dst, level);
    // Calculate Energy, Contrast, Homogenity, Entropy of the whole Image
    Mat Energy_tmp, Contrast_tmp, Homogenity_tmp, Entropy_tmp;
    GLCM::CalcuTextureImages(dst, Energy_tmp, Contrast_tmp, Homogenity_tmp, Entropy_tmp, winsize, level, true);

    Mat output;
    Mat Energy_hist = GetHistOfMaskArea(Energy_tmp, Mat(), 0, 255, histsize/4, true);
    Mat Contrast_hist = GetHistOfMaskArea(Contrast_tmp, Mat(), 0, 255, histsize/4, true);
    Mat Homogenity_hist = GetHistOfMaskArea(Homogenity_tmp, Mat(), 0, 255, histsize/4, true);
    Mat Entropy_hist = GetHistOfMaskArea(Entropy_tmp, Mat(), 0, 255, histsize/4, true);
    output.push_back(Energy_hist);
    output.push_back(Contrast_hist);
    output.push_back(Homogenity_hist);
    output.push_back(Entropy_hist);
    return output.reshape(1,1);
 }



/*===================================================================
* Function: GetMPEG7DCD
*
* Summary:
*   Get MPEG-7 dominant color descriptor (DCD)
*   include three color components and the weight of the most dominant color
*
* Arguments:
*   const Mat& src -  BGR img
*   int numOfColor - default 3
*
* Returns:
*   Mat of Size(1,12) for default numOfColor = 3
=====================================================================
*/
Mat cvFeatures::GetMPEG7DCD(const Mat& src, int numOfColor) {

    Mat feature_temp;
    Mat dst = src.clone();
    normalize(dst, dst, 0, 255, NORM_MINMAX);
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
        
        for (int w = 0; w < numOfColor; w++) {
            feature_temp.push_back(int(domcol[w].m_Percentage / weight));
            feature_temp.push_back(domcol[w].m_ColorValue[0]);
            feature_temp.push_back(domcol[w].m_ColorValue[1]);
            feature_temp.push_back(domcol[w].m_ColorValue[2]);
        }
    }
    // release descriptor
    delete dcd;
    return feature_temp.reshape(1,1);
 }

/*===================================================================
* Function: GetMPEG7CSD
*
* Summary:
*   Get MPEG-7 color structure descriptor (CSD)
*   include three color components and the weight of the most dominant color
*
* Arguments:
*   const Mat& src -  BGR img
*   int Size - length of the feature vector,default 32
*
* Returns:
*   Mat of Size(1,32)
=====================================================================
*/
Mat cvFeatures::GetMPEG7CSD(const Mat& src, int Size) {
    Mat dst = src.clone();
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
    delete csd;
    return feature_temp.reshape(1,1);
 }

/*===================================================================
 * Function: GetStatistic
 *
 * Summary:
 *   Compute median, min, max, mean, std of mask area
 *
 * Arguments:
 *   Mat src -  CV_32FC1
 *
 * Returns:
 *   Mat of Size(5,1)
=====================================================================
*/
Mat cvFeatures::GetStatistic(const Mat& src) {

        Mat result;

        vector<float> vec(src.begin<float>(), src.end<float>());
        int size = static_cast<int>(src.total());
        // sort the vector
        std::sort(vec.begin(), vec.end());
        if (size % 2 == 0)
        {
            //median
            result.push_back ( (vec[size / 2.0 - 1] + vec[size / 2]) / 2);
        }
        else
        {
            //median
            result.push_back( vec[size / 2]);
        }

        //min
        result.push_back(vec[0]);
        //max
        result.push_back(vec[size - 1]);

        Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
        meanStdDev(vec, mean, stddev);
        float m = mean[0];
        float s = stddev[0];
        //mean
        result.push_back(m);
        //stddev
        result.push_back(s);
        return result;
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




