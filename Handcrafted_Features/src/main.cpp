#include <opencv2/opencv.hpp>
#include "Data.hpp"
#include "FeatureProcess.h"
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
        cout << "filterSize choose from: " <<  "0,3,5,7,9,11 \n"  << endl;
        cout << "mp stands for: " << "morphological profile features\n" << endl;
        cout << "decomp stands for: " << "target decomposition features\n" << endl;
        cout << "color stands for: " << "color features (MPEG-7 CSD,DCD)\n" << endl;
        cout << "texture stands for: " << "texture features (GLCM,LBP)\n" << endl;
        cout << "polstatistic stands for: " <<  "statistic of polsar parameters (median, min, max, mean, std)\n"  << endl;
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

    Data* ob = new Data(ratfolder, labelfolder);

    FeatureProcess* f = new FeatureProcess(hdf5file);
    f->setParam(feature_name, filterSize, patchSize, batchSize);
    f->caculFeatures(ob->data,ob->LabelMap, ob->classNames);
    delete ob;
    
    int trainPercent = 80, K = 20;
    f->classifyFeaturesML( "opencvKNN", trainPercent, K);
    
    f->generateColorMap("opencvKNN");

    int batchID = 0;
    f->featureDimReduction(batchID);

    f->generateFeatureMap();
    delete f;

    return 0;
}


