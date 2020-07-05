#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/cvdef.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include "sarFeatures.hpp"
#include "KNN.hpp"
#include "cv_hdf5.hpp"
#include "../tsne/tsne.h"

namespace Utils {
	    // calculate the accuracy for each class, and return the overal accuracy
		float calculatePredictionAccuracy(const std::string& feature_name, const std::vector<unsigned char>& classResult, const std::vector<unsigned char>& testLabels);
		cv::Mat generateLabelMap(const std::vector<cv::Mat>& masks);


		// by Jun Xiang

		// feature dimension reduction by T-SNE
		// save the dimension reduced features to txt file
		// check the KNN accuracy on dim reduced features
		void featureDimReduction(const std::string& hdf5_fileName, const std::string& feature_name, int filterSize, int patchSize);
		cv::Mat featureDimReduction(const cv::Mat & feature, int new_dims =2);

		// use opencv ml functions
		// feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		// classifier_type: choose from {"KNN", "RF", "FLANN"}
		void classifyFeaturesML(const std::string& hdf5_fileName, const std::string& feature_name, const std::string classifier_type, int trainPercent, int filterSize, int patchSize);
		// return the class results
		void applyML(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_labels, int trainPercent, const std::string& classifier_type, std::vector<unsigned char>& class_result);
		void splitVec(const std::vector<cv::Mat>& features, const std::vector<unsigned char>& labels, const std::vector<cv::Point>& labelPoints, std::vector<std::vector<cv::Mat>>& subFeatures,
			std::vector<std::vector<unsigned char>>& subLables, std::vector<std::vector<cv::Point>>& subLabelPoints, int n = 4);
		
		// self written KNN
		// feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		void classifyFeaturesKNN(const std::string& hdf5_fileName, const std::string& feature_name, int k, int trainPercent, int filterSize, int patchSize);

		// input: the class label
		// return: the color
		cv::Vec3b getLabelColor( unsigned char class_label);
		
		// get the colormap of classified results
		// feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		// classifier_type: choose from {"KNN", "RF", "FLANN"}
		void generateColorMap(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classifier_type, int filterSize,int patchSize);
		
		// get feature, featureLabels and labelPoints from hdf5 file
        // feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		void getFeaturesFromHDF(const std::string& hdf5_fileName, const std::string& feature_name, std::vector<std::string>& dataset_name,
			std::vector<cv::Mat>& features, std::vector<unsigned char>& featureLabels, std::vector<cv::Point>& labelPoints, int filterSize = 5, int patchSize = 20); 

		// write back the classified result to hdf ( sample points, class result from classifier)
		// feature_name: choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
		// classResult_name: choose from {"/KNN","/RF","/FLANN" }
		void saveClassResultToHDF(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classResult_name,
			const std::vector<unsigned char>& class_result, const std::vector<cv::Point>& points, int filterSize, int patchSize);


		// generate all the possible sample points
		std::vector<cv::Point>  generateSamplePoints(const cv::Mat& labelMap, const int& sampleSize, const int & stride );
		// get random samples of homogeneous area for one type of class, numOfSamplePointPerClass =0 means to return all the possible sample points
		void getRandomSamplePoint(const cv::Mat& labelMap, std::vector<cv::Point>& samplePoints, const unsigned char& sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass);


		//************* HDF5 file read/write/insert/delete *****************//
		bool checkExistInHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name,int filterSize,int patchSize);
		bool checkExistInHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);
		// delete dataset from hdf5 file
		void deleteDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);
		void deleteDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, int filterSize, int patchSize);
		// eg: filename = "ober.h5", parent_name = "/filtered_data", dataset_name = "/hh_filterSize_5"
		void writeDataToHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data);
		void writeDataToHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, const std::vector<cv::Mat>& data,int filterSize =0, int patchSize =0);
		// eg: filename = "ober.h5" ,parent_name = "/filtered_data", dataset_name = "/hh_filterSize_5"
		void readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, cv::Mat& data);
		void readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name,  std::vector<cv::Mat>& data, int filterSize =0, int patchSize =0);
		//write attribute to the root group
		void writeAttrToHDF(const std::string& filename, const std::string& attribute_name, const int &attribute_value);
		void writeAttrToHDF(const std::string& filename, const std::string& attribute_name, const std::string &attribute_value);
		void readAttrFromHDF(const std::string& filename, const std::string& attribute_name, int& attribute_value);
		void readAttrFromHDF(const std::string& filename, const std::string& attribute_name, std::string& attribute_value);
		//insert data
		bool insertDataToHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data);
		bool insertDataToHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, const std::vector<cv::Mat>& data, int filterSize, int patchSize);


		
		
		//************** Prepare dataset for KNN / Random Forest ***************************//
		// split the data into train/test set balancely in different classes
		// return the index of the test data in original data
		// fold : crossvalidation number,an integer between {1, 100 / (100 - percentOfTrain)}
		std::vector<int> DivideTrainTestData(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_label, int percentOfTrain,
			std::vector<cv::Mat>& train_img, std::vector<unsigned char>& train_label, std::vector<cv::Mat>& test_img, std::vector<unsigned char>& test_label, int fold);
		std::vector<int> shuffleDataSet(std::vector<cv::Mat>& data, std::vector<unsigned char>& data_label);
		cv::Mat getConfusionMatrix(const std::map<unsigned char, std::string>& className, std::vector<unsigned char>& classResult, std::vector<unsigned char>& testLabels);

};
#endif
