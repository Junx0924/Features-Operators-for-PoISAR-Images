#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "sen12ms.hpp"
#include "torchDataSet.cpp"
#include <iostream>
#include <fstream>
#include <string>
#include <valarray>


using namespace std;
using namespace cv;
 
int main() {

    // generate the list by using "dir /a-D /S /B >s1FileList.txt" in window10 command
    string s1FileListPath = "../data/s1FileList.txt";
    string lcFileListPath = "../data/lcFileList.txt";
    sen12ms* sar = new sen12ms(s1FileListPath, lcFileListPath);
  
    MaskType mask_type = MaskType::IGBP;
    int batch_size = 3;
    sar->SetMaskType(mask_type);
    sar->SetBatchSize(batch_size);
    sar->LoadBatchToMemeory(0); // load the first batch

    vector<Mat> imageOfMaskArea;
    vector<unsigned char> classValue;
    sar->ProcessData(imageOfMaskArea, classValue);

    // load to torch
    //auto custom_dataset = torchDataset(imageOfMaskArea, classValue).map(torch::data::transforms::Stack<>());
    //auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(custom_dataset),batch_size);
     
    vector<Mat> LBPfeatures;
    vector<unsigned char> LBPLabels;
    sar->GetFeatureLBP(LBPfeatures, LBPLabels, 1, 8, 32);

    vector<Mat> GLCMfeatures;
    vector<unsigned char> GLCMLabels;
    sar->GetFeatureGLCM(GLCMfeatures, GLCMLabels, 5, GrayLevel::GRAY_8, 32);

    vector<Mat> Statisticfeatures;
    vector<unsigned char> StatLabels;
    sar->GetFeatureStatistic(Statisticfeatures, StatLabels, 32);

    return 0; // success
}
