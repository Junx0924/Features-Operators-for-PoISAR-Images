#include <opencv2/opencv.hpp>
#include  <opencv2/highgui.hpp>
#include "MPEG-7/FexWrite.h"
#include "GLCM/glcm.hpp"
#include "ELBP/elbp.cpp"
#include "PolSAR/Geotiff.cpp"
#include "PolSAR/sen12ms.hpp"

using namespace std;
using namespace cv;


int main() {
   
    //-----------read SEN12MS dataset (tiff files)
    // 32 bits per channel, VV,VH
    const char* s1 = "../features/data/ROIs1158_spring_s1_1_p30.tif";

    //8 bits per channel, (IGBP, LCCS_LC, LCCS_LU, LCCS_SH) 
    const char* lc = "../features/data/ROIs1158_spring_lc_1_p30.tif";

    // get PolSAR data for this patch
    GeoTiff *sar = new GeoTiff(s1);
    //stores VH ,VV values in dB
    Mat patch = sar->GetMat().clone();
    delete sar;

    // get ground truth for this patch
    GeoTiff* ground = new GeoTiff(lc);
    Mat lc_mat = ground->GetMat().clone();
    delete ground;

    // get flase colar image from PolSAR data
    Mat colorImg = sen12ms::GetFalseColorImage(patch, true);
    namedWindow("False color image", 1);
    imshow("False color image", colorImg);
    waitKey(0);
   
   // get the Masks of current patch
    vector<Mat> IGBP_mask, LCCS_mask;
    vector<unsigned char> IGBP_list;
    vector<unsigned char> LCCS_list;
    sen12ms::GetMask(lc_mat, IGBP_mask, LCCS_mask, IGBP_list,LCCS_list);

    Mat mask = LCCS_mask[1];
    int class_type = LCCS_list[1];
    Mat tmp;
    normalize(mask, tmp, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("mask: ", tmp);
    waitKey(0);
//----------------- local statistic-------------------------
    Mat statPol = sen12ms::GetPolStatistic(patch, mask);
    cout << "statistic: "<< statPol << endl;

    Mat src = colorImg;
    int histsize  = 32; // feature length

//-------------------------lbp-------------------------------
    // to get lbp of the whole img
    int radius = 2;
    int neighbors = 8;
    Mat lbp = elbp::CaculateElbp(src, radius, neighbors, true);
    imshow("lbp: ", lbp);
    waitKey(0);
    // Apply mask
    Mat lbp_hist = sen12ms::hist_with_mask(lbp, mask, 0, 255, histsize, true);
    cout << "lbp hist: " << lbp_hist << endl;
    
//--------------------color features: MPEG7 ---------------------------
    // create a Frame object (see include/Frame.h)
    Frame* frame = new Frame(src.cols, src.rows, true, true, true);
    frame->setImage(src);
    // Apply mask
   frame->setMaskAll(mask, class_type, 255, 0);
    // color: DCD, return the weights and color value of each dominant color
    cout << "DCD with mask:" << endl;
    FexWrite::computeWriteDCD(frame, false, false, false);

    // color: CSD
    cout << "CSD with mask:" << endl;
    FexWrite::computeWriteCSD(frame, histsize);;


    //--------------------------GLCM---------------------------
  
    int size = 7; // only support size 5,7
    GrayLevel level = GrayLevel::GRAY_8;
     

    Mat dstChannel;
    GLCM::getOneChannel(src, dstChannel, RGBChannel::CHANNEL_R);
    // Magnitude Gray Image
    GLCM::GrayMagnitude(dstChannel, dstChannel, level);

    // Calculate Energy, Contrast, Homogenity, Entropy of the whole Image
    Mat Energy ,Contrast,Homogenity,Entropy ;
    GLCM::CalcuTextureImages(dstChannel, Energy, Contrast, Homogenity, Entropy, size, level, true);
    // Apply the mask
    Mat Energy_hist = sen12ms::hist_with_mask(Energy, mask, 0, 255, histsize, true);
    cout << "GLCM Energy with mask:" << endl;
    cout << Energy_hist << endl;
     
     
    return 0; // success
}
