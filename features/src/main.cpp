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
     
    // set patch size 20, maximum sample points per class is 10
    ob->LoadSamplePoints(20, 100);
    ob->SetFilterSize(5);

    KNN* knn = new KNN();
    vector<Mat> feature;
    vector<unsigned char> featureLabels;
    ob->GetTextureFeature(feature, featureLabels);

    knn->applyKNN(feature, featureLabels, 20, 80);

    //ob->GetColorFeature(feature, featureLabels);

    //ob->GetMPFeature(feature, featureLabels);

    //ob->GetAllPolsarFeatures(feature, featureLabels);


    return 0; // success
}
