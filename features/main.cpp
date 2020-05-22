#include <opencv2/opencv.hpp>
#include  <opencv2/highgui.hpp>
#include "MPEG-7/FexWrite.h"
#include "GLCM/glcm.hpp"
#include "ELBP/elbp.cpp"


using namespace std;

int main() {

    
   Mat src = cv::imread("D\:\\4th semester\\HTCV\\project_code\\data\\PauliRGB.bmp", 1);
   Mat labeled = cv::imread("D\:\\4th semester\\HTCV\\project_code\\data\\SF-GF3-label3d.png", 1); // ground truth
  
   Mat mask = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
   // get the mountain area 
   // readme.txt: 1.Mountain, label color(RGB): [132,112,255]
    int cl[3] = { 132,112,255 };
    int regVal = 255;
    for (int i = 0; i < labeled.rows; i++) {
        for (int j = 0; j < labeled.cols; j++) {
            if (labeled.at<Vec3b>(i, j)[0] == cl[2] && labeled.at<Vec3b>(i, j)[1] == cl[1] && labeled.at<Vec3b>(i, j)[2] == cl[0]) {
                mask.at<unsigned char>(i, j) = regVal;
            }
        }
    }
   
   
   //--------------------color features: MPEG7 ---------------------------
   // create a Frame object (see include/Frame.h)
    // allocate memory for 3 - channel color image and 1 - channel gray mask
      Frame* frame = new Frame(src.cols, src.rows, true, true, true);
      frame->setImage(src);
      frame->setMaskAll(mask, regVal, 255, 0);
    // compute and display the descriptors for the 'region'
    // color: DCD
    // DCD without normalization (to MPEG-7 ranges), without variance, without spatial coherency
    // attention: should recompute the weight(sum to 100 percentage)
       cout << "DCD with mask:" << endl;
       FexWrite::computeWriteDCD(frame, false, false, false);
    // color: CSD
       cout << "CSD with mask:" << endl;
       FexWrite::computeWriteCSD(frame, 32);;

   //--------------------texture features: MPEG7 ---------------------------
    // Textur: EHD
       cout << "EHD with mask:" << endl;
       FexWrite::computeWriteEHD(frame);

    // release frame
       delete frame;
    //-------------------------lbp-------------------------------
   
    // to get lbp image
       int radius = 1;
       int neighbors = 8;
    // to get spatial histogram vector of lbp
       int grid_x = 4;
       int grid_y = 4;
       int numPatterns = 32; // // number of possible patterns,32 bins

       //Mat tmp;
       //threshold(mask, tmp, 0, 255, THRESH_BINARY);

    // p is the elbp vector of channel 1, row 1, col grid_x*grid_y*numPatterns
       Mat p = elbp::computLBPvector(
        src, //image 
        radius,
        neighbors,
        numPatterns,
        grid_x, 
        grid_y, 
        true);
   


    //--------------------------GLCM---------------------------
 /*
    int size = 7; // only support size 5,7
    GrayLevel level = GrayLevel::GRAY_8;
    TextureEValues EValues;

    Mat dstChannel;
    GLCM::getOneChannel(src, dstChannel, RGBChannel::CHANNEL_B);
    // Magnitude Gray Image
    GLCM::GrayMagnitude(dstChannel, dstChannel, level);

    // Calculate Energy, Contrast, Homogenity, Entropy of the whole Image
    GLCM::CalcuTextureEValue(dstChannel, EValues, size, level);
    cout << "EValues: " << EValues.contrast << " , " << EValues.energy  << endl;
    */
    return 0; // success
}
