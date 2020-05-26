#include <string>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "sen12ms.hpp"

// land class type
//  IGBP[17] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 };
//  LCCS_LC[16] = { 11,12,13,14,15,16,21,22,31,32,41,43,42,2,1,3 };
//  LCCS_LU[11] = { 10,20,30,40,36,9,25,35,2,1,3 };
//  LCCS_SH[10] = { 10,20,40,30,27,50,51,2,1,3 };
std::array<unsigned char, 17>  IGBP_label = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 };
// LCCS_LC, LCCS_LU,LCCS_SH merged into LCCS
std::array<unsigned char, 26>  LCCS_label = { 11,12,13,14,15,16,10,21,22,30,31,32,40,41,43,42,27,50,36,9,25,35,51,2,1,3 };

/*===================================================================
 * Function: GetFalseColorImage
 *
 * Summary:
 *   Generate false color image from SAR data;
 *
 * Arguments:
 *   Mat src - 2 channel matrix(values in dB) from tiff file
 *   bool normed - normalized to 0-255 
 *
 * Returns:
 *   3 channel matrix: R: VV, G:VH, B: VV/VH
=====================================================================
*/
Mat sen12ms::GetFalseColorImage(const Mat& src, bool normed) {
    vector<Mat>  Channels;
    split(src, Channels);

    Mat R = cv::abs(Channels[0]); //VV
    Mat G = cv::abs(Channels[1]);  //VH
    
    

    Mat B = Mat::zeros(src.rows, src.cols, CV_32FC1); //VV/VH
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (Channels[1].at<float>(i, j) != 0) {
                B.at<float>(i, j) =  Channels[0].at<float>(i, j)  /  Channels[1].at<float>(i, j) ;
            }
        }
    }
    Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC3);
    vector<Mat> temp;
    temp.push_back(B);
    temp.push_back(G);
    temp.push_back(R);
    merge(temp, dst);
    if (normed) {
        cv::normalize(dst, dst, 0, 255, NORM_MINMAX);
        dst.convertTo(dst, CV_8UC3);
    }
    return dst;
}


/*===================================================================
 * Function: GetLabelClass
 *
 * Summary:
 *   Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
 *   Generate IGBP, LCCS from the ground truth 
 *
 * Arguments:
 *   Mat src - 4 channel matrix from groundtruth file
 *   Mat& IGBP - Destination Mat for IGBP class
 *   Mat& LCCS - Destination Mat for LCCS class
 *
 * Returns:
 *   3 channel matrix: R: VV, G:VH, B: VV/VH
=====================================================================
*/
void sen12ms::GetClassCategory(const Mat& lc, Mat& IGBP, Mat& LCCS) {
   
    vector<Mat> temp(lc.channels());
    split(lc, temp);

    IGBP = temp[0];
    Mat LCCS_LC = temp[1];
    Mat LCCS_LU = temp[2];
    Mat LCCS_SH = temp[3];

   
    for (int i = 0; i < lc.rows; i++) {
        for (int j = 0; j < lc.cols; j++) {
            if (LCCS_LC.at<unsigned char>(i, j) != 0) {
                LCCS.at<unsigned char>(i, j) = LCCS_LC.at<unsigned char>(i, j);
            }
            else if (LCCS_LU.at<unsigned char>(i, j) != 0) {
                LCCS.at<unsigned char>(i, j) = LCCS_LU.at<unsigned char>(i, j);
            }
            else {
                LCCS.at<unsigned char>(i, j) = LCCS_SH.at<unsigned char>(i, j);
            }
        }
    }
}
 
/*===================================================================
 * Function: GetPolStatistic
 *
 * Summary:
 *   Compute min, max, mean, std, median of mask area
 *
 * Arguments:
 *   Mat src - matrix of PolSAR data ( VV, VH for each channel)
 *   const Mat& mask - single channel mask matrix 
 *
 * Returns:
 *   Single channel Mat of Size(src.channels(), 5)
=====================================================================
*/
Mat sen12ms::GetPolStatistic(const Mat& src, const Mat& mask) {

    Mat stat;
    
    for (int i =0; i<src.channels(); i++){
        Mat result= Mat(1, 5, CV_32FC1);

        Mat src_temp = Mat(src.rows, src.cols, CV_32FC1);
        extractChannel(src, src_temp, i);
       
        // put the mask area into a vector
        vector<float> srcWithMask;
        for (int x = 0; x < src.rows; x++) {
            for (int y = 0; y < src.cols; y++) {
                if (mask.at<unsigned char>(x, y) == 0) {
                    continue;
                }
                else {
                    srcWithMask.push_back(src_temp.at<float>(x, y));

                }
            }
        }
        //median
        int size = srcWithMask.size();
        std::sort(srcWithMask.begin(), srcWithMask.end());
        if (size % 2 == 0)
        {
            result.at<float>(0,4) = (srcWithMask[size / 2 - 1] + srcWithMask[size / 2]) / 2;
        }
        else
        {
            result.at<float>(0,4) = srcWithMask[size / 2];
        }

        double min,  max;
        cv::minMaxIdx(srcWithMask, &min, &max);

        //min
        result.at<float>(0,0) = srcWithMask[0];
        //max
        result.at<float>(0, 1) = srcWithMask[size-1];

        cv::Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
        cv::meanStdDev(srcWithMask, mean, stddev);
        //mean
        result.at<float>(0, 2) = mean[0];
        //stddev
        result.at<float>(0, 3) = stddev[0];

        stat.push_back(result);
    }
    return stat;
}

/*===================================================================
 * Function: GetMask
 *
 * Summary:
 *   Create Masks for each class category
 *
 * Arguments:
 *   Mat & lc - 4 channel matrix from groundtruth file
 *   vector<Mat> &IGBP_mask - Destination Mask Mat for IGBP class
 *   vector<Mat> &LCCS_mask - Destination Mask Mat for  LCCS class
 *   vector<unsigned char>IGBP_list  - record each IGBP class value in the masks  
 *   vector<unsigned char> LCCS_list - record each LCCS class value in the masks  

 * Returns:
 *  void
=====================================================================
*/
void sen12ms::GetMask(const Mat& lc, vector<Mat>& IGBP_mask, vector<Mat>& LCCS_mask, vector<unsigned char> &IGBP_list, vector<unsigned char>& LCCS_list) {
    Mat igbp = Mat(lc.rows, lc.cols, CV_8UC1);
    Mat lccs = Mat(lc.rows, lc.cols, CV_8UC1);
    // merge different LCCS class channels to one channel
    sen12ms::GetClassCategory(lc, igbp, lccs);
    
    //get IGBP mask
    for (int i = 0; i < IGBP_label.size(); i++) {
        vector<std::pair<int, int> > ind;
        if (FindLandClass(igbp,ind, IGBP_label[i])) {
            IGBP_list.push_back(IGBP_label[i]);
            Mat tmp = Mat::zeros(lc.rows, lc.cols, CV_8UC1);
            for (auto const& p : ind) {
               tmp.at<unsigned char>(p.first, p.second) = IGBP_label[i];
            }
            IGBP_mask.push_back(tmp);
        }
    }
    // get LCCS_mask
    for (int i = 0; i < LCCS_label.size(); i++) {
        vector<std::pair<int, int> > ind;
        if (FindLandClass(lccs, ind, LCCS_label[i])) {
            LCCS_list.push_back(LCCS_label[i]);
            Mat tmp = Mat::zeros(lc.rows, lc.cols, CV_8UC1);
            for (auto const& p : ind) {
               tmp.at<unsigned char>(p.first, p.second) = LCCS_label[i];
            }
            LCCS_mask.push_back(tmp);
        }
    }
}

/*===================================================================
 * Function: FindLandClass
 *
 * Summary:
 *   check if cetain class type existed in a class category
 *
 * Arguments:
 *   Mat & src - IGBP mat or LCCS mat
 *   vector<std::pair<int, int> > &ind - record the index of the class type
 *   const int &landclass: value of the class type

 * Returns:
 *  bool
=====================================================================
*/
 bool sen12ms::FindLandClass(const Mat& src, vector<std::pair<int, int> > &ind, const int &landclass) {
     bool flag = false;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<unsigned char>(i,j) == landclass) {
                ind.push_back(std::make_pair(i, j));
                flag = true;
            }
        }
    }
    return flag;
}

 /*===================================================================
 * Function: GetHistWithMask
 *
 * Summary:
 *   Caculate the historgram vector of a mat with mask
 *
 * Arguments:
 *   Mat & src - IGBP mat or LCCS mat
 *   const Mat& mask -  single channel mask matrix 
 *   int minVal - the min of bin boundaries
 *   int maxVal - the max of bin boundaries
 *   int histSize 
 *   bool normed - normalized to make the sum become 1
 * Returns:
 *  Mat of Size(1,histSize)
=====================================================================
*/
 Mat sen12ms::GetHistWithMask(const Mat& src, const Mat& mask, int minVal, int maxVal, int histSize, bool normed)
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