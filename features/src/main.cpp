#include <opencv2/opencv.hpp>
#include "ober.hpp"
#include "Utils.h"

using namespace std;
using namespace cv;

string MP = "/MP";
string decomp = "/decomp";
string color = "/color";
string texture = "/texture";
string polStatistic = "/polStatistic";
string ctElements = "/CTelememts";
string all = "all";
vector<string> dataset_name = { "/feature" ,"/patchLabel" };
vector<string> feature_type = { MP , decomp, color, texture, polStatistic,ctElements};


 
int main() {

    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";
    string oberfile = "E:\\ober.h5";
    
   
    // test features
    int patchSize = 10;
    int numOfSamplePoints = 0;
    int filterSize = 0;
    unsigned char classlabel = 255;
    string feature_name = MP;
    
    ober* ob = new ober(ratfolder, labelfolder, oberfile);
    ob->caculFeatures(filterSize, patchSize, numOfSamplePoints, classlabel, feature_name);
    delete ob;
    
    Utils::classifyFeaturesML(oberfile, feature_name, "opencvFLANN", 80, filterSize, patchSize);
    
    Utils::generateColorMap(oberfile, feature_name, "opencvFLANN", filterSize, patchSize);
    Utils::featureDimReduction(oberfile, feature_name, 5000,filterSize, patchSize);
   
}
   

