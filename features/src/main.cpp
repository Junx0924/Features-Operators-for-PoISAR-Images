//#include "torchDataSet.cpp"
//#include "sen12ms.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "ober.hpp"
#include "KNN.hpp"


using namespace std;
using namespace cv;
 
int main() {

    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";

    ober* ob = new ober(ratfolder, labelfolder);
     
    // set patch size 20, maximum sample points per class is 100
    ob->LoadSamplePoints(20, 100);
    ob->SetFilterSize(5);

    KNN* knn = new KNN();
    vector<Mat> feature;
    vector<unsigned char> featureLabels;

    //ob->GetTextureFeature(feature, featureLabels);
    
    ob->GetColorFeature(feature, featureLabels);

   // ob->GetMPFeature(feature, featureLabels);

    // polsar features
   // ob->GetDecompFeatures(feature, featureLabels);

   // ob->GetCTFeatures(feature, featureLabels);

   // ob->GetPolsarStatistic(feature, featureLabels);


    knn->applyKNN(feature, featureLabels, 20, 80);

    return 0; // success
}
