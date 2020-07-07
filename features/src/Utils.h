#pragma once
#ifndef UTILS_H
#define UTILS_H


#include "featureProcess.hpp"
#include "sarFeatures.hpp"
#include "cv_hdf5.hpp"


namespace Utils {

		void featureDimReduction(const std::string& hdf5_fileName, const std::string& feature_name, int numSamples, int filterSize, int patchSize);

		void classifyFeaturesML(const std::string& hdf5_fileName, const std::string& feature_name, const std::string classifier_type, int trainPercent, int filterSize, int patchSize, int batchSize);

		void generateColorMap(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classifier_type, int filterSize,int patchSize, int batchSize);
	    
		cv::Vec3b getLabelColor(unsigned char class_label);
	
		void getFeaturesFromHDF(const std::string& hdf5_fileName, const std::string& feature_name, std::vector<std::string>& dataset_name,
			std::vector<cv::Mat>& features, std::vector<unsigned char>& featureLabels, std::vector<cv::Point>& labelPoints, int filterSize , int patchSize, int offset_row = 0, int counts_rows = 0);
		
		void saveClassResultToHDF(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classResult_name,
			const std::vector<unsigned char>& class_result, const std::vector<cv::Point>& points, int filterSize, int patchSize);
		  
		std::vector<cv::Point>  generateSamplePoints(const cv::Mat& labelMap, const int& sampleSize, const int & stride );
	 
		void getRandomSamplePoint(const cv::Mat& labelMap, std::vector<cv::Point>& samplePoints, const unsigned char& sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass);
		
		cv::Mat generateLabelMap(const std::vector<cv::Mat>& masks);
		
		void splitVec(const std::vector<unsigned char>& labels, std::vector<std::vector<int>>& subInd, int batchSize=5000);

		std::map<unsigned char, std::string> getClassName(const std::string& filename);

		//************* HDF5 file read/write/insert/delete *****************//
		// eg: filename = "ober.h5", parent_name = "/MP", dataset_name = "/feature_patchSize_10"
		bool checkExistInHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name,int filterSize=0,int patchSize=0);
		void writeDataToHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, const std::vector<cv::Mat>& data, int filterSize = 0, int patchSize = 0);
		void deleteDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, int filterSize = 0, int patchSize = 0);
		void readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, std::vector<cv::Mat>& data, int filterSize = 0, int patchSize = 0, int offset_row = 0, int counts_rows = 0);
		bool insertDataToHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, const std::vector<cv::Mat>& data, int filterSize = 0, int patchSize = 0);

		bool checkExistInHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);
		bool insertDataToHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data);
		void readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, cv::Mat& data, int offset_row = 0, int counts_rows = 0);
		void deleteDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);
		void writeDataToHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data);
		int getRowSize(const std::string& filename, const std::string& parent_name, const std::string& dataset_name,int filterSize=0, int patchSize=0);
		//write attribute to the root group
		void writeAttrToHDF(const std::string& filename, const std::string& attribute_name, const int &attribute_value);
		void writeAttrToHDF(const std::string& filename, const std::string& attribute_name, const std::string &attribute_value);
		void readAttrFromHDF(const std::string& filename, const std::string& attribute_name, int& attribute_value);
		void readAttrFromHDF(const std::string& filename, const std::string& attribute_name, std::string& attribute_value);


};
#endif
