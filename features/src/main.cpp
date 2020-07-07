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
string ctElements = "/CTelements";
string all = "all";
vector<string> dataset_name = { "/feature" ,"/patchLabel" };
vector<string> feature_type = { MP , decomp, color, texture, polStatistic,ctElements};


 
int main(int argc, char** argv) {

    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <ratFolder> <labelFolder> <oberFile> <featureName>" << endl;
        cout << "e.g. " << argv[0] << " E:\\Oberpfaffenhofen\\sar-data E:\\Oberpfaffenhofen\\label E:\\MP.h5 /MP" << endl;
        return 0;
    }

    string ratfolder = argv[1]; // "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = argv[2]; //"E:\\Oberpfaffenhofen\\label";
    string oberfile = argv[3]; //"E:\\MP.h5";
    string feature_name = argv[4]; //MP;

    cout << "Using following params:" << endl;
    cout << "ratfolder = " << ratfolder << endl;
    cout << "labelfolder = " << labelfolder << endl;
    cout << "oberfile = " << oberfile << endl;
    cout << "feature_name = " << feature_name << "\n\n"<< endl;

      // test features
      unsigned char classlabel = 255; //all the class
      int numOfSamplePoints = 0; //all the points
      int patchSize = 10;
      int filterSize = 0;
      int batchSize = 5000;
    
    ober* ob = new ober(ratfolder, labelfolder, oberfile);
    ob->caculFeatures(feature_name, classlabel, filterSize, patchSize, numOfSamplePoints, batchSize);
    delete ob;
    
    Utils::classifyFeaturesML(oberfile, feature_name, "opencvFLANN", 80, filterSize, patchSize, batchSize);
    
    Utils::generateColorMap(oberfile, feature_name, "opencvFLANN", filterSize, patchSize);
    Utils::featureDimReduction(oberfile, feature_name, 3000,filterSize, patchSize);

    return 0;
   
}
   

//int main() {
//
//    string ratfolder =   "E:\\Oberpfaffenhofen\\sar-data";
//    string labelfolder =   "E:\\Oberpfaffenhofen\\label";
//    string oberfile =  "E:\\CTelements.h5";
//
//        // test features
//    string feature_name = ctElements;
//    unsigned char classlabel = 255;
//    int numOfSamplePoints = 3000;
//    int patchSize = 3;
//    int filterSize = 0;
//    int batchSize = 5000;
//
//    ober* ob = new ober(ratfolder, labelfolder, oberfile);
//    ob->caculFeatures(feature_name, classlabel, filterSize, patchSize, numOfSamplePoints, batchSize);
//    delete ob;
//    
//    Utils::classifyFeaturesML(oberfile, feature_name, "opencvFLANN", 80, filterSize, patchSize,batchSize);
//    
//    Utils::generateColorMap(oberfile, feature_name, "opencvFLANN", filterSize, patchSize);
//    Utils::featureDimReduction(oberfile, feature_name, 3000,filterSize, patchSize);
//
//    return 0;
//}
