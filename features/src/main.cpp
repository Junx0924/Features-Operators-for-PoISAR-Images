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

    // set patch size 64, maximum sample points per class is 2000
    polsar* ober = new polsar(ratfolder, labelfolder,64,2000); 
     

    vector<Mat> patches;
    vector<unsigned char> labels;
    ober->GetData(patches, labels);

    cvFeatures f = cvFeatures(patches[0], labels[0]);
    vector<Mat> MPfeatures;
    vector<unsigned char> MPLabels;
    f.GetMP(MPfeatures, MPLabels);

    
    return 0; // success
}
