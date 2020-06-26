#include <opencv2/opencv.hpp>
#include "ober.hpp"
#include "Utils.h"

string MP = "/MP";
string decomp = "/decomp";
string color = "/color";
string texture = "/texture";
string polStatistic = "/polStatistic";
string ctElements = "/CTelememts";
string knn_result = { "/knn" };
string all = "all";
vector<string> dataset_name = { "/feature" ,"/patchLabel" };
vector<string> feature_type = { MP , decomp, color, texture, polStatistic,ctElements};

using namespace std;
using namespace cv;
 
int main() {

    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";
    string oberfile = "E:\\ober2.h5";

    ober* ob = new ober(ratfolder, labelfolder, oberfile);

    // test PolStatistic features
    int patchSize = 20;
    int numOfSamplePoints = 100;
    int filterSize = 0;
    string feature_name = all;
    ob->caculFeatures(filterSize, patchSize, numOfSamplePoints, feature_name);
    delete ob;

    for(const auto & name: feature_type){
    Utils::classifyFeaturesKNN(oberfile, name, 20, 80, filterSize, patchSize);
    Utils::generateColorMap(oberfile, name, knn_result, filterSize, patchSize);
    }

    return 0;
}
   

