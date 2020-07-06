#pragma once
#ifndef UTILS_H
#define UTILS_H


#include "featureProcess.hpp"
#include "sarFeatures.hpp"
#include "cv_hdf5.hpp"


namespace Utils {

    	// randomly select N feature samples from hdf5 file, reduce the feature dimension by T-SNE
		// check the KNN accuracy on dim reduced features
		// save the results to txt file for plotting in matlab
		// feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		void featureDimReduction(const std::string& hdf5_fileName, const std::string& feature_name, int numSamples, int filterSize, int patchSize);


		// use opencv ml functions
		// feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		// classifier_type: choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
		void classifyFeaturesML(const std::string& hdf5_fileName, const std::string& feature_name, const std::string classifier_type, int trainPercent, int filterSize, int patchSize);
		void splitVec(const std::vector<int>& labels, std::vector<std::vector<int>>& subInd, int n);
		
		
		// get the colormap of classified results
		// feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		// classifier_type: choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
		void generateColorMap(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classifier_type, int filterSize,int patchSize);
		// input: the class label
		// return: the color
		cv::Vec3b getLabelColor(unsigned char class_label);




		// get feature, featureLabels and labelPoints from hdf5 file
        // feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		void getFeaturesFromHDF(const std::string& hdf5_fileName, const std::string& feature_name, std::vector<std::string>& dataset_name,
			std::vector<cv::Mat>& features, std::vector<unsigned char>& featureLabels, std::vector<cv::Point>& labelPoints, int filterSize , int patchSize, int offset_row = 0, int counts_rows = 0);

		// write back the classified result to hdf ( sample points, class result from classifier)
		// feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		// classResult_name: choose from {"/KNN","/RF","/FLANN" }
		void saveClassResultToHDF(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classResult_name,
			const std::vector<unsigned char>& class_result, const std::vector<cv::Point>& points, int filterSize, int patchSize);


		// generate all the possible sample points
		std::vector<cv::Point>  generateSamplePoints(const cv::Mat& labelMap, const int& sampleSize, const int & stride );
		// get random samples of homogeneous area for one type of class, numOfSamplePointPerClass =0 means to return all the possible sample points
		void getRandomSamplePoint(const cv::Mat& labelMap, std::vector<cv::Point>& samplePoints, const unsigned char& sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass);
		cv::Mat generateLabelMap(const std::vector<cv::Mat>& masks);



		//************* HDF5 file read/write/insert/delete *****************//
		bool checkExistInHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name,int filterSize,int patchSize);
		bool checkExistInHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);
		// delete dataset from hdf5 file
		void deleteDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);
		void deleteDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, int filterSize, int patchSize);
		// eg: filename = "ober.h5", parent_name = "/MP", dataset_name = "/feature_patchSize_10"
		void writeDataToHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data);
		void writeDataToHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, const std::vector<cv::Mat>& data,int filterSize, int patchSize);
		// eg: filename = "ober.h5" ,parent_name = "/MP", dataset_name = "/feature_patchSize_10"
		void readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, cv::Mat& data, int offset_row = 0, int counts_rows = 0);
		void readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name,  std::vector<cv::Mat>& data, int filterSize, int patchSize, int offset_row=0, int counts_rows=0);
		int getRowSize(const std::string& filename, const std::string& parent_name, const std::string& dataset_name,int filterSize, int patchSize);
		
		//write attribute to the root group
		void writeAttrToHDF(const std::string& filename, const std::string& attribute_name, const int &attribute_value);
		void writeAttrToHDF(const std::string& filename, const std::string& attribute_name, const std::string &attribute_value);
		void readAttrFromHDF(const std::string& filename, const std::string& attribute_name, int& attribute_value);
		void readAttrFromHDF(const std::string& filename, const std::string& attribute_name, std::string& attribute_value);
		//insert data
		bool insertDataToHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data);
		bool insertDataToHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, const std::vector<cv::Mat>& data, int filterSize, int patchSize);


};
#endif
