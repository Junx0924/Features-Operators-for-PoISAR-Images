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
    int batch_size = 10;
    
    auto custom_dataset = torchDataset(sar->s1FileList, sar->lcFileList, mask_type).map(torch::data::transforms::Stack<>());
    //auto custom_dataset = torchDataset(s1FileListPath, lcFileListPath, mask_type).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(custom_dataset),batch_size);
     
 
    return 0; // success
}
