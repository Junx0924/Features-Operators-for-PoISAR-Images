//#include "torchDataSet.cpp"
//#include "sen12ms.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "ober.hpp"
#include "KNN.hpp"

using namespace std;
using namespace cv;

string ctElements = "/CTelememts";
string MP = "/MP";
string decomp = "/decomp";
string color = "/color";
string texture = "/texture";
string polStatistic = "/polStatistic";
string knn_result = { "/knn" };
vector<string> dataset_name = { "/feature" ,"/patchLabel" };

int main() {

    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";
    string oberfile = "E:\\ober.h5";
   
    int filterSize = 0;
    int patchSize = 20;
    int numOfPoints = 1000;
    string feature_name = polStatistic;


    ober* ob = new ober(ratfolder, labelfolder);
    ob->caculFeatures(oberfile, feature_name,filterSize, patchSize, numOfPoints );

    Utils::classifyFeaturesKNN(oberfile, feature_name, 20, 80, filterSize, patchSize);
    Utils::generateColorMap(oberfile, feature_name, knn_result, filterSize, patchSize);

   delete ob;

    return 0; // success
}
