#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "sen12ms.hpp"
#include "GetFeatures.hpp"
#include "polsar.hpp"



using namespace std;
using namespace cv;
 
int main() {
    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";

    // set patch size 64, maximum sample points per class is 2000d
    polsar* ober = new polsar(ratfolder, labelfolder,64,2000); 
     

    vector<Mat> patches;
    vector<unsigned char> labels;
    ober->GetData(patches, labels);

    cvFeatures f = cvFeatures(patches[0], labels[0]);
    vector<Mat> MPfeatures;
    vector<unsigned char> MPLabels;
    f.GetMP(MPfeatures, MPLabels);

    vector<Mat> MPfeatures;
    vector<unsigned char> MPLabels;
    f.GetMP(MPfeatures, MPLabels);
    
    vector<Mat> texture;
    vector<unsigned char> textureLabels;
    f.GetLBP(texture, textureLabels, 1, 8, 32);
    f.GetGLCM(texture, textureLabels, 5, GrayLevel::GRAY_8, 32);

    vector<Mat> colorfeatures;
    vector<unsigned char> colorLabels;
    f.GetMPEG7DCD(colorfeatures, colorLabels, 3);
    f.GetMPEG7CSD(colorfeatures, colorLabels, 32);

    return 0; // success
}
