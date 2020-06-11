#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "sen12ms.hpp"
#include "ober.hpp"



using namespace std;
using namespace cv;
 
int main() {

    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";

    // set patch size 10, maximum sample points per class is 100
    ober* ob = new ober(ratfolder, labelfolder, 10, 10);


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


    return 0; // success
}
