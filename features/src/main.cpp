//#include "torchDataSet.cpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
//#include "sen12ms.hpp"
#include "ober.hpp"
#include "sarFeatures.hpp"



using namespace std;
using namespace cv;
 
int main() {

    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";

    ober* ob = new ober(ratfolder, labelfolder);

     
    // set patch size 20, maximum sample points per class is 10
    ob->LoadSamplePoints(20, 10);
    ob->SetFilterSize(5);

    vector<Mat> texture;
    vector<unsigned char> textureLabels;
    ob->GetTextureFeature(texture, textureLabels);

    vector<Mat> colorfeatures;
    vector<unsigned char> colorLabels;
    ob->GetColorFeature(colorfeatures, colorLabels);

    vector<Mat> MPfeatures;
    vector<unsigned char>MPLabels;
    ob->GetMPFeature(MPfeatures, MPLabels);

    vector<Mat> sarfeatures;
    vector<unsigned char> sarLabels;
    ob->GetAllPolsarFeatures(sarfeatures, sarLabels);

    //int batch_size = 64;
    //auto custom_dataset =  torchDataset(texture, textureLabels).map(torch::data::transforms::Stack<>());
    //auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(custom_dataset), batch_size);
     

    return 0; // success
}
