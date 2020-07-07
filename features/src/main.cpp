#include <opencv2/opencv.hpp>
#include "ober.hpp"
#include "Utils.h"
#include <string> 

using namespace std;
using namespace cv;

string MP = "/MP";
string decomp = "/decomp";
string color = "/color";
string texture = "/texture";
string polStatistic = "/polStatistic";
string CTelements = "/CTelements";
vector<string> dataset_name = { "/feature" ,"/patchLabel" };
 
int main(int argc, char** argv) {

    if (argc < 5) {
        cout << "Usage: " << argv[0] << " <ratFolder> <labelFolder> <oberFile> <featureName>  <filterSize> \n" << endl;
        cout << "e.g. " << argv[0] << " E:\\Oberpfaffenhofen\\sar-data E:\\Oberpfaffenhofen\\label E:\\MP.h5 /MP 0\n" << endl;
        cout << "featureName choose from: \n" << MP << "," << decomp << "," << color << "," << texture << "," << polStatistic << "," << CTelements << endl;
        cout << "filterSize choose from: \n" <<  "0,5,7,9,11 \n"  << endl;
        return 0;
    }

    string ratfolder = argv[1]; // "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = argv[2]; //"E:\\Oberpfaffenhofen\\label";
    string oberfile = argv[3]; //"E:\\MP.h5";
    string feature_name = argv[4]; //MP;
    int filterSize = stoi(argv[5]);  
    if ((filterSize != 5) && (filterSize != 7) && (filterSize != 9) && (filterSize != 11)) { filterSize = 0;}
    unsigned char classlabel = 255; //all the class
    int numOfSamplePoints = 0; //all the points
    int batchSize = 5000;
    int patchSize = 10;
    if (feature_name == CTelements) { patchSize = 3; }
    if (feature_name == MP) { batchSize = 3000; }

    cout << "Using following params:" << endl;
    cout << "ratfolder = " << ratfolder << endl;
    cout << "labelfolder = " << labelfolder << endl;
    cout << "oberfile = " << oberfile << endl;
    cout << "feature_name = " << feature_name << endl;
    cout << "filterSize = " << filterSize << endl;
    cout << "patchSize = " << patchSize << "\n"<< endl;

    ober* ob = new ober(ratfolder, labelfolder, oberfile);
    ob->caculFeatures(feature_name, classlabel, filterSize, patchSize, numOfSamplePoints, batchSize);
    delete ob;
    
    Utils::classifyFeaturesML(oberfile, feature_name, "opencvFLANN", 80, filterSize, patchSize, batchSize);
    
    Utils::generateColorMap(oberfile, feature_name, "opencvFLANN", filterSize, patchSize, batchSize);
    Utils::featureDimReduction(oberfile, feature_name, 3000,filterSize, patchSize);

    return 0;
   
}
   

//int main() {
//
//    string ratfolder =   "E:\\Oberpfaffenhofen\\sar-data";
//    string labelfolder =   "E:\\Oberpfaffenhofen\\label";
//    string oberfile =  "E:\\CTelements.h5";
//
//    // test features
//    string feature_name = CTelements;
//    unsigned char classlabel = 255;
//    int numOfSamplePoints = 3000;
//    int patchSize = 3;
//    int filterSize = 0;
//    int batchSize = 5000;
//
//    //ober* ob = new ober(ratfolder, labelfolder, oberfile);
//    //ob->caculFeatures(feature_name, classlabel, filterSize, patchSize, numOfSamplePoints, batchSize);
//    //delete ob;
//    
//    //Utils::classifyFeaturesML(oberfile, feature_name, "opencvFLANN", 80, filterSize, patchSize,batchSize);
//    
//    Utils::generateColorMap(oberfile, feature_name, "opencvFLANN", filterSize, patchSize, batchSize);
//    Utils::featureDimReduction(oberfile, feature_name, 3000,filterSize, patchSize);
//
//    return 0;
//}
