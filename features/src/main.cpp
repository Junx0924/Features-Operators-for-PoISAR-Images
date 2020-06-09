#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "sen12ms.hpp"
#include "polsar.hpp"



using namespace std;
using namespace cv;
 
int main() {

    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";

    // set patch size 64, maximum sample points per class is 100
    polsar* ober = new polsar(ratfolder, labelfolder, 10, 1000);


   vector<Mat> texture;
   vector<unsigned char> textureLabels;
   ober->GetTextureFeature(texture, textureLabels);

   vector<Mat> colorfeatures;
   vector<unsigned char> colorLabels;
   ober->GetColorFeature(colorfeatures, colorLabels);

   vector<Mat> Statfeatures;
   vector<unsigned char> StatLabels;
   ober->GetStatisticFeature(Statfeatures, StatLabels);

   vector<Mat> MPfeatures;
   vector<unsigned char>MPLabels;
   ober->GetMPFeature(MPfeatures, MPLabels);

    return 0; // success
}
