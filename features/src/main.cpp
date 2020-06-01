//#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "glcm.hpp"
#include "elbp.cpp"
#include "sen12ms.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
 
int main() {


     // torch::Tensor tensor = torch::rand({ 2, 3 });
     // std::cout << tensor << std::endl;
    
    // generate the list by use "dir /a-D /S /B >s1FileList.txt" in window10 command
    string output = "E:\\sen12ms_png";
    string s1FileListPath = "../data/s1FileList.txt";
    string lcFileListPath = "../data/lcFileList.txt";
   
    sen12ms  test = sen12ms(s1FileListPath, lcFileListPath);
    test.LoadDataToMemeory(2, MaskType::IGBP);
     
    // for dataloader
    vector<Mat>   imagesOfMaskArea;
    vector<unsigned char> labels;
    test.ProcessData(imagesOfMaskArea, labels);
     
     
     
   /*
    
  Mat src = list_images[0];
  Mat mask = list_masks[0][0];
  int class_type = list_classValue[0][0];

//----------------- local statistic-------------------------
    Mat statPol = sen12ms::GetPolStatistic(src, mask);
    cout << "statistic: "<< statPol << endl;

    
    int histsize  = 32; // feature length

//-------------------------lbp-------------------------------
    // to get lbp of the whole img
    int radius = 2;
    int neighbors = 8;
    Mat lbp = elbp::CaculateElbp(src, radius, neighbors, true);
    imshow("lbp: ", lbp);
    waitKey(0);
    // Apply mask
    Mat lbp_hist = sen12ms::GetHistWithMask(lbp, mask, 0, 255, histsize, true);
    cout << "lbp hist: " << lbp_hist << endl;
    
//--------------------color features: MPEG7 ---------------------------
    // create a Frame object (see include/Frame.h)
    // normalize to 0-255
    cv::normalize(src, src, 0, 255, NORM_MINMAX);
    src.convertTo(src, CV_8UC3);
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
    Mat Energy_hist = sen12ms::GetHistWithMask(Energy, mask, 0, 255, histsize, true);
    cout << "GLCM Energy with mask:" << endl;
    cout << Energy_hist << endl;
     */
     
    return 0; // success
}
