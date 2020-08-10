#include <opencv2/opencv.hpp>
#include "ober.hpp"
#include "Utils.h"
#include <string> 
 

using namespace std;
using namespace cv;
 

string ctelements = "ctelements";
string decomp = "decomp";
string mp = "mp";
string color = "color";
string texture = "texture";
string polstatistic = "polstatistic";
std::vector<std::string> features = { color,texture,polstatistic,mp,decomp,ctelements };

 
int main(int argc, char** argv) {

    if (argc < 7) {
        cout << "Usage: " << argv[0] << " <ratFolder> <labelFolder> <Hdf5File> <featureName> <filterSize> <patchSize> \n" << endl;
        cout << "e.g. " << argv[0] << " E:\\Oberpfaffenhofen\\sar-data E:\\Oberpfaffenhofen\\label E:\\mp.h5  mp 0 10\n" << endl;
        cout << "featureName choose from: " << mp << "," << decomp << "," << color << "," << texture << "," << polstatistic << "," << ctelements <<"\n"<< endl;
        cout << "filterSize choose from: " <<  "0,5,7,9,11 \n"  << endl;
        cout << "mp stands for: " << "morphological profile features\n" << endl;
        cout << "decomp stands for: " << "target decomposition features\n" << endl;
        cout << "color stands for: " << "MPEG-7 CSD,DCD and HSV features\n" << endl;
        cout << "texture stands for: " << "GLCM and LBP features\n" << endl;
        cout << "polstatistic stands for: " <<  "the statistic of polsar parameters\n"  << endl;
        cout << "ctelements stands for: " <<  "the 6 upcorner elements of covariance and coherence matrix\n"  << endl;
        return 0;
    }

    string ratfolder = argv[1];  
    string labelfolder = argv[2]; 
    string hdf5file = argv[3];  
    string feature_name = argv[4];  
    int filterSize = stoi(argv[5]);  
    if ((filterSize != 5) && (filterSize != 7) && (filterSize != 9) && (filterSize != 11)) { filterSize = 0;}
    int patchSize = stoi(argv[6]);
    if (feature_name == ctelements) { patchSize = 3; }
    if (feature_name == decomp) { patchSize = 3; }
    int batchSize = 3000;
     

    cout << "Using following params:" << endl;
    cout << "ratfolder = " << ratfolder << endl;
    cout << "labelfolder = " << labelfolder << endl;
    cout << "hdf5file = " << hdf5file << endl;
    cout << "feature_name = " << feature_name << endl;
    cout << "filterSize = " << filterSize << endl;
    cout << "patchSize = " << patchSize << "\n"<< endl;

    ober* ob = new ober(ratfolder, labelfolder);
     
    ob->caculFeatures(hdf5file,feature_name,filterSize, patchSize,batchSize);
    delete ob;
    
    Utils::classifyFeaturesML(hdf5file, feature_name, "opencvFLANN", 80, filterSize, patchSize, batchSize);
    
    Utils::generateColorMap(hdf5file, feature_name, "opencvFLANN", filterSize, patchSize, batchSize);

    Utils::featureDimReduction(hdf5file, feature_name, filterSize, patchSize, batchSize);

    Utils::generateFeatureMap(hdf5file, feature_name, filterSize, patchSize, batchSize);

    return 0;
}

