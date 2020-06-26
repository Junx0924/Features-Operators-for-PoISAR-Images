#pragma once
#ifndef  OBER_HPP_
#define  OBER_HPP_
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "specklefilter.hpp"
#include "sarFeatures.hpp"
#include "cv_hdf5.hpp"
#include "cvFeatures.hpp"
#include "sarFeatures.hpp"
#include "Utils.h"


/*defining compiler versions for some
compiler specific includes*/
#define VC				//GCC/VC

using namespace std;
using namespace cv;

//load data from Oberpfaffenhofen
class ober{

private:
	unsigned border = 3;

	vector<Point> samplePoints;
	vector<unsigned char> sampleLabel;
	 
	string hdf5_file;

public:
	// data = complex mat with values [HH, VV, HV]
	  vector<Mat> data;

	  Mat LabelMap;

	// record the class name of each label
	std::map<unsigned char, string>classNames; 

	// constructor
	// input: rat file folder, label file folder 
	ober(const string& RATfileFolder, const string& labelFolder, const string & hdf5_fileName) {

		samplePoints = vector<Point>();
		sampleLabel = vector<unsigned char>();

		hdf5_file = hdf5_fileName;

		// read rat data, can't save them directly to hdf5, it will lost precision
		loadData(RATfileFolder);

		// read labels
		vector<Mat> masks;
		vector<string>  labelNames;
		ReadClassLabels(labelFolder, labelNames, masks);

		// read the labelNames to dict
		classNames[unsigned char(0)] = "Unclassified";
		for (int i = 0; i < labelNames.size(); i++) {
			classNames.insert(pair<unsigned char, string>(i + 1, labelNames[i]));
		}

		 writeLabelMapToHDF(hdf5_file, masks, this->LabelMap);
		 

	}


	~ober() {
		
	}

	// calulate features and save them to hdf5 file
	// filterSize: apply refined Lee despeckling filter, choose from (0, 5, 7, 9, 11)
	// patchSize: to draw samples
    // classlabel: choose which class to load, 255 means to load all the classes
    // numOfSamplePoint, the number of samples for one type of class, 0 means load all the possible sample points
	// feature_name: choose from { texture, color, ctElements,polStatistic,decomp, MP}
	 void caculFeatures(int filterSize, int patchSize, int numOfSamplePoint, unsigned char classlabel, string feature_name);

private:
	// input sample size and the maximum number of samples per class 
	//numOfSamplePoint =0 means load all the possible sample points
	void LoadSamplePoints(const int& sampleSize, const int& numOfSamplePoint, const unsigned char& classlabel, int stride = 1);
	void getSample(const Point& p, int patchSize, int filtersize, Mat& hh, Mat& vv, Mat& hv);
	void generateSamplePoints(const string& hdf5_fileName, const Mat& labelmap, int patchSize);
	void getSampleInfo(const string& hdf5_fileName, const Mat& pts, int patchSize);

	// get texture features(LBP and GLCM) on HH,VV,VH
	Mat caculTexture(const Mat& hh, const Mat& vv, const Mat& hv);
	// get color features(MPEG-7 DCD,CSD) on Pauli Color image
	Mat caculColor(const Mat& hh, const Mat& vv, const Mat& hv);
	// get MP features on HH,VV,VH 
	Mat caculMP(const Mat& hh, const Mat& vv, const Mat& hv);
	// calculate covariance and coherency matrix and store to hdf5 file
	// get polsar features on elements of covariance matrix C and coherency matrix T
	Mat caculCTelements(const Mat& hh, const Mat& vv, const Mat& hv);
	// get polsar features on target decompostion 
	Mat caculDecomp(const Mat& hh, const Mat& vv, const Mat& hv);
	// get polsar features on statistic of polsar parameters
	Mat caculPolStatistic(const Mat& hh, const Mat& vv, const Mat& hv);

	// write the labelmap to hdf5 and return the labelmap
	void writeLabelMapToHDF(const string& hdf5_fileName, const vector<Mat>& masks,Mat& labelMap);

	/***Author: Anupama Rajkumar***/
	void loadData(string RATfolderPath);
	void ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages);
	Size loadRAT(string fname, vector<Mat>& data, bool metaOnly = false);
	Size loadRAT2(string fname, vector<Mat>& data, bool metaOnly = false);
	void getTileInfo(cv::Size size, unsigned border, unsigned& tile, vector<unsigned>& tileSize, vector<unsigned>& tileStart);
};

#endif