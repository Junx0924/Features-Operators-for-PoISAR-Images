#pragma once
#ifndef  OBER_HPP_
#define  OBER_HPP_
#include <opencv2/opencv.hpp>
#include "specklefilter.hpp"
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
	int filterSize;
public:
	// data = complex mat with values [HH, VV, HV]
	vector<Mat> data;
	vector<unsigned char> labels;  
	vector<Mat> masks;

	// record the class name of each label
	std::map<unsigned char, string>classNames; 

	vector<Point> *samplePoints;
	vector<unsigned char> *samplePointClassLabel;
	int sampleSize;
	int samplePointNum;

	// constructor
	// input: rat file folder, label file folder 
	ober(const string& RATfileFolder, const string& labelFolder) {

		samplePoints = new vector<Point>();
		samplePointClassLabel = new vector<unsigned char>();
		sampleSize = 0;
		samplePointNum = 0;
		filterSize = 0;

		// read rat data
		loadData(RATfileFolder);
		// read labels
		vector<string>  labelNames;
		ReadClassLabels(labelFolder, labelNames, masks);

		// read the labelNames to dict
		classNames[signed char(0)] = "Unclassified";
		for (int i = 0; i < labelNames.size(); i++) {
			classNames.insert(pair<unsigned char, string>(i + 1 , labelNames[i]));
			labels.push_back(i + 1);
		}
	}

	~ober() {
		if (samplePoints) { delete samplePoints; }
		if (samplePointClassLabel) { delete samplePointClassLabel; }
	}

	// set despeckling filter size, choose from ( 5, 7, 9, 11)
	 void SetFilterSize(int filter_size);

	// input sample size and the maximum number of sample points 
	 void LoadSamplePoints(const int& sampleSize, const int& samplePointNum);

	 // get patches of 3 channel (HH+VV,HV,HH-VV) intensity(dB)
	 void GetPauliColorPatches(vector<Mat>& patches, vector<unsigned char>& classValue);

	 // get patches of 3 channel (HH,HV,VV) intensity(dB)
	 void GetPatches(vector<Mat>& patches, vector<unsigned char>& classValue);

	 // get texture features(LBP and GLCM) on HH,VV,VH
	 void GetTextureFeature(vector<Mat>& features, vector<unsigned char>& classValue);

	 // get color features(MPEG-7 DCD,CSD) on Pauli Color image
	 void GetColorFeature(vector<Mat>& features, vector<unsigned char>& classValue);

	 // get MP features on HH,VV,VH, default feature mat size (sampleSize*3,sampleSize)
	 void GetMPFeature(vector<Mat>& features, vector<unsigned char>& classValue);
	 
	 // get polsar features on elements of covariance matrix C and coherency matrix T
	 void GetCTFeatures(vector<Mat>& features, vector<unsigned char>& classValue);

	 // get polsar features on target decompostion 
	 void GetDecompFeatures(vector<Mat>& features, vector<unsigned char>& classValue);

	 // get polsar features on statistic of polsar parameters
	 void GetPolsarStatistic(vector<Mat>& features, vector<unsigned char>& classValue);

private:
	// calculate target decompostion features
	// vector<mat> result, vector length: , mat size: (hh.rows,hh.cols)
	void getTargetDecomposition(const Mat & hh, const Mat &vv, const Mat hv, vector<Mat>& result);

	// get upper triangle matrix elements of C, T
	// vector<mat> result, vector length: 12, mat size: (hh.rows,hh.cols)
	void getCTelements(const Mat& hh, const Mat& vv, const Mat hv, vector<Mat>& result);

	// get statistical (min,max,mean,median,std) on polsar parameters
	// vector<mat> result, vector length : 7, mat size: 1*5
	void getStatisticFeature(const Mat& hh, const Mat& vv, const Mat hv, vector<Mat>& result);

	// apply refined Lee filter to samples, filterSize choose from (5,7,9,11)
	void getSample(const Point& sample_point, Mat& hh, Mat& vv, Mat& hv);

	/***Author: Anupama Rajkumar***/
	void loadData(string RATfolderPath);
	void ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages);
	Size loadRAT(string fname, vector<Mat>& data, bool metaOnly = false);
	Size loadRAT2(string fname, vector<Mat>& data, bool metaOnly = false);
	void getTileInfo(cv::Size size, unsigned border, unsigned& tile, vector<unsigned>& tileSize, vector<unsigned>& tileStart);
};

#endif