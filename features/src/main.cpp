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
    
    ob->SetFilterSize(5);

    // set patch size 20, maximum sample points per class is 1000
    ob->LoadSamplePoints(20, 1000);

    KNN* knn = new KNN();
    vector<Mat> feature;
    vector<unsigned char> featureLabels;

    ob->GetTextureFeature(feature, featureLabels);
    knn->applyKNN(feature, featureLabels, 20, 80);

    feature.clear();
    featureLabels.clear();
    ob->GetColorFeature(feature, featureLabels);
    knn->applyKNN(feature, featureLabels, 20, 80);

    feature.clear();
    featureLabels.clear();
   ob->GetMPFeature(feature, featureLabels);
   knn->applyKNN(feature, featureLabels, 20, 80);

   feature.clear();
   featureLabels.clear();
    // polsar features
   ob->GetDecompFeatures(feature, featureLabels);
   knn->applyKNN(feature, featureLabels, 20, 80);

   feature.clear();
   featureLabels.clear();
   ob->GetCTFeatures(feature, featureLabels);
   knn->applyKNN(feature, featureLabels, 20, 80);

   feature.clear();
   featureLabels.clear();
   ob->GetPolsarStatistic(feature, featureLabels);
   knn->applyKNN(feature, featureLabels, 20, 80);

   delete knn;
   delete ob;

    return 0; // success
}
