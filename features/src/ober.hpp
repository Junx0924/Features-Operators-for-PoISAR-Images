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



//load data from Oberpfaffenhofen
class ober{

private:
	unsigned border = 3;

	std::vector<cv::Point> samplePoints;
	std::vector<unsigned char> sampleLabel;
	 
	std::string hdf5_file;

public:
	// data = complex mat with values [HH, VV, HV]
	  std::vector<cv::Mat> data;

	  cv::Mat LabelMap;

	// record the class name of each label
	std::map<unsigned char, std::string>classNames; 

	// constructor
	// input: rat file folder, label file folder 
	ober(const std::string& RATfileFolder, const std::string& labelFolder, const std::string & hdf5_fileName) {

		samplePoints = std::vector<cv::Point>();
		sampleLabel = std::vector<unsigned char>();

		hdf5_file = hdf5_fileName;

		// read rat data, can't save them directly to hdf5, it will lost precision
		loadData(RATfileFolder);

		// read labels
		std::vector<cv::Mat> masks;
		std::vector<std::string>  labelNames;
		ReadClassLabels(labelFolder, labelNames, masks);

		// read the labelNames to dict
		classNames[unsigned char(0)] = "Unclassified";
		for (int i = 0; i < labelNames.size(); i++) {
			classNames.insert(std::pair<unsigned char, std::string>(i + 1, labelNames[i]));
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
	 void caculFeatures(std::string feature_name, unsigned char classlabel, int filterSize, int patchSize, int numOfSamplePoint,int batchSize =5000);

private:
	// input sample size and the maximum number of samples per class 
	// numOfSamplePoint =0 means load all the possible sample points
	// shuffle all the points based on its label
	void LoadSamplePoints(const int& sampleSize, const int& numOfSamplePoint, const unsigned char& classlabel, int stride = 1);
	void getSample(const cv::Point& p, int patchSize, int filtersize, cv::Mat& hh, cv::Mat& vv, cv::Mat& hv);
	
	void getSampleInfo(const std::string& hdf5_fileName, const cv::Mat& pts, int patchSize);

	// get texture features(LBP and GLCM) on HH,VV,VH
	cv::Mat caculTexture(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);
	// get color features(MPEG-7 DCD,CSD) on Pauli Color image
	cv::Mat caculColor(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);
	// get MP features on HH,VV,VH 
	cv::Mat caculMP(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);
	// calculate covariance and coherency matrix and store to hdf5 file
	// get polsar features on elements of covariance matrix C and coherency matrix T
	cv::Mat caculCTelements(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);
	// get polsar features on target decompostion 
	cv::Mat caculDecomp(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);
	// get polsar features on statistic of polsar parameters
	cv::Mat caculPolStatistic(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// write the labelmap to hdf5 and return the labelmap
	void writeLabelMapToHDF(const std::string& hdf5_fileName, const std::vector<cv::Mat>& masks,cv::Mat& labelMap);

	/***Author: Anupama Rajkumar***/
	void loadData(std::string RATfolderPath);
	void ReadClassLabels(std::string labelPath, std::vector<std::string>& labelNames, std::vector<cv::Mat>& labelImages);
	cv::Size loadRAT(std::string fname, std::vector<cv::Mat>& data, bool metaOnly = false);
	cv::Size loadRAT2(std::string fname, std::vector<cv::Mat>& data, bool metaOnly = false);
	void getTileInfo(cv::Size size, unsigned border, unsigned& tile, std::vector<unsigned>& tileSize, std::vector<unsigned>& tileStart);
};

#endif