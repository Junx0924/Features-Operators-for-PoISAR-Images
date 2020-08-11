#pragma once
#ifndef  FEATUREPROCESS_HPP_
#define  FEATUREPROCESS_HPP_
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include "specklefilter.hpp"
#include "cvFeatures.hpp"
#include "sarFeatures.hpp"
#include "dataset_hdf5.hpp"
#include "DataProcess.hpp"

class FeatureProcess {
private:
	std::string hdf5_file;
	int filter_Size =0; //apply refined Lee despeckling filter, choose from (0, 3, 5, 7, 9, 11)
	int patch_Size =0;
	int batch_Size = 5000;
	std::string featureName; //choose from {  "Texture" ,"Color", "CT", "PolStat", "TD", "MP" }
	
public:
	// constructor
	FeatureProcess(const std::string& hdf5_fileName) {
		this->hdf5_file = hdf5_fileName;
	}

	//set parameters
	void setParam(std::string& feature_name, int filterSize, int patchSize, int batchSize) {
		this->featureName = feature_name;
		this->filter_Size = filterSize;
		this->patch_Size = patchSize;
		this->batch_Size = batchSize;
	}

	//calulate features and save to hdf5 file
	void caculFeatures(const std::vector<cv::Mat>& data, const cv::Mat& LabelMap, const std::map<unsigned char, std::string>& classNames,int numOfSamplePoint = 0, int stride = 1, unsigned char classlabel = 255);

	//Read the features from hdf5 file, classify them and write the classifiy results into hdf5 file
	void classifyFeaturesML(const std::string classifier_type, int trainPercent, int K);

	//Generate the colormap of classified results, calculate the overall accuracy for each class
	void generateColorMap(const std::string& classifier_type);

	//Get the visulization of feature map for each dimension of feature group
	void generateFeatureMap();
	
	//reduced the feature dimension by T-SNE, dump to txt for plotting
	void featureDimReduction(int batchID);

private:
	// write labelmap and classNames save to hdf5
	void writeLabelMapToHDF(const std::string& hdf5_fileName, const std::vector<cv::Mat>& data, const cv::Mat& labelMap, const std::map<unsigned char, std::string>& classNames);

	// get class name from hdf5 file
	std::map<unsigned char, std::string> getClassName(const std::string& hdf5_fileName);

	void LoadSamplePoints(std::vector<cv::Point>& samplePoints, std::vector<unsigned char>& sampleLabel, const cv::Mat& LabelMap, const std::map<unsigned char, std::string>& classNames, const int& patchSize, const  int& numOfSamplePoint, int stride , const unsigned char& classlabel);

	void getSample(const std::vector<cv::Mat>& data, const cv::Point& p, int patchSize, int filtersize, cv::Mat& hh, cv::Mat& vv, cv::Mat& hv);

	void getSampleInfo(const std::string& hdf5_fileName, const cv::Mat& pts, int patchSize);

	// get texture features(LBP and GLCM) on HH,VV,VH
	cv::Mat caculTexture(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get color features(MPEG-7 DCD,CSD) on Pauli Color image
	cv::Mat caculColor(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get MP features on HH,VV,VH 
	cv::Mat caculMP(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get polsar features on elements of covariance matrix C and coherency matrix T
	cv::Mat caculCTelements(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get polsar features on target decompostion 
	cv::Mat caculDecomp(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get polsar features on statistic of polsar parameters
	cv::Mat caculPolStatistic(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	cv::Vec3b getLabelColor(unsigned char class_label);

	//get features data and its groundtruth from hdf5
	void getFeaturesFromHDF(const std::string& hdf5_fileName, const std::string& feature_name,
		std::vector<cv::Mat>& features, std::vector<unsigned char>& featureLabels, std::vector<cv::Point>& labelPoints, int offset_row = 0, int counts_rows = 0);

	//save class result to hdf5
	void saveClassResultToHDF(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classResult_name,
		const std::vector<unsigned char>& class_result, const std::vector<cv::Point>& points);

};








#endif