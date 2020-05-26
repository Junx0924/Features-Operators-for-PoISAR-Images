#include <string>
#include <fstream>
#include <iostream>
#include "stdlib.h"
#include <opencv2/opencv.hpp>
#include "Geotiff.cpp"
#include "sen12ms.hpp"

using namespace cv;
using namespace std;


std::string type2str(int type) {
    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch (depth) {
    case CV_8U:  r = "CV_8U"; break;
    case CV_8S:  r = "CV_8S"; break;
    case CV_16U: r = "CV_16U"; break;
    case CV_16S: r = "CV_16S"; break;
    case CV_32S: r = "CV_32S"; break;
    case CV_32F: r = "CV_32F"; break;
    case CV_64F: r = "CV_64F"; break;
    default:     r = "User"; break;
    }
    r += "C";
    r += (chans + '0');
    return r;
}



int main() {
	// 32 bits per channel, VV,VH
	const char* s1 = "E:\\SEN12MS\\ROIs1158_spring\\s1\\s1_1\\ROIs1158_spring_s1_1_p30.tif";
     
	 //8 bits per channel, (IGBP, LCCS_LC, LCCS_LU, LCCS_SH) 
    const char* lc = "E:\\SEN12MS\\ROIs1158_spring\\lc\\lc_1\\ROIs1158_spring_lc_1_p30.tif";
   
    // create object of Geotiff class
    Geotiff s(s1);
   //stores VH ,VV values in dB
    Mat patch = s.GetMat();  

    // normalized for display
    Mat colorImg = sen12ms::GetFalseColorImage(patch, true);
    namedWindow("False color image", 1); 
    imshow("False color image",  colorImg);
    waitKey(0);
    s.~Geotiff();

    // get masks for this patch
    Geotiff l(lc);
    Mat lc_mat = l.GetMat();
    vector<Mat> IGBP_mask,  LCCS_mask;
    sen12ms::GetMask(lc_mat, IGBP_mask, LCCS_mask);

    //get local statistic for this patch with mask
    Mat test = Mat::ones(patch.rows, patch.cols, CV_8UC1);
    Mat statPol = sen12ms::GetPolStatistic(patch, IGBP_mask[0]);
	return 0;
}