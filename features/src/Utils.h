#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include "sarFeatures.hpp"
#include "cv_hdf5.hpp"
#include "Geotiff.hpp"
#include "KNN.hpp"

using namespace std;
using namespace cv;

namespace Utils {
	    // by Anupama Rajkumar
		void GetPatchIndex(int sizeOfPatch, Point2i& samplePoint, const Mat& LabelMap, int& min_col, int& min_row, int& max_col, int& max_row);
		float calculatePredictionAccuracy(const vector<unsigned char>& classResult, const vector<unsigned char>& testLabels);
		Mat generateLabelMap(const vector<Mat>& masks);


		// by Jun Xiang
		// input: the ground_truth label and the test label
		// return: the color
		Vec3b getLabelColor(unsigned char ground_truth, unsigned char class_result);
	    // feature_name: choose from { texture, color, ctElements,polStatistic,decomp, MP}
		void generateColorMap(const String& hdf5_fileName, const string& feature_name, const string& class_result, int filterSize,int patchSize);
		
		// feature_name: choose from { texture, color, ctElements,polStatistic,decomp, MP}
		void classifyFeaturesKNN(const String& hdf5_fileName, const string& feature_name, int k, int trainPercent, int filterSize, int patchSize);
		// write back the classified result to hdf ( sample points, class result from classifier)
		void saveClassResultToHDF(const String& hdf5_fileName, const String& parent_name, const string& dataset_name,
			const vector<unsigned char>& class_result, const vector<Point>& points, int filterSize, int patchSize);
		void getFeaturesFromHDF(const String& hdf5_fileName, const String& parent_name, const vector<String>& dataset_name,
			vector<Mat>& features, vector<unsigned char>& featureLabels, vector<Point>& labelPoints, int filterSize = 5, int patchSize = 20);


		// generate all the possible sample points
		vector<Point>  generateSamplePoints(const Mat& labelMap, const int& sampleSize, const int & stride );
		// get random samples of homogeneous area for each class, numOfSamplePointPerClass =0 means to return all the possible sample points
		void getRandomSamplePoint(const Mat& labelMap, vector<Point>& samplePoints, vector<unsigned char>& sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass);


		//************* HDF5 file read/write/insert/delete *****************//
		bool checkExistInHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name,int filterSize,int patchSize);
		bool checkExistInHDF(const String& filename, const String& parent_name, const string& dataset_name);
		// delete dataset from hdf5 file
		void deleteDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name);
		void deleteDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, int filterSize, int patchSize);
		// eg: filename = "ober.h5", parent_name = "/filtered_data", dataset_name = "/hh_filterSize_5"
		void writeDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& data);
		void writeDataToHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, const vector<Mat>& data,int filterSize =0, int patchSize =0);
		// eg: filename = "ober.h5" ,parent_name = "/filtered_data", dataset_name = "/hh_filterSize_5"
		void readDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name, Mat& data);
		void readDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name,  vector<Mat>& data, int filterSize =0, int patchSize =0);
		//write attribute to the root group
		void writeAttrToHDF(const String& filename, const String& attribute_name, const int &attribute_value);
		void writeAttrToHDF(const String& filename, const String& attribute_name, const string &attribute_value);
		void readAttrFromHDF(const String& filename, const String& attribute_name, int& attribute_value);
		void readAttrFromHDF(const String& filename, const String& attribute_name, string& attribute_value);
		//insert data
		bool insertDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& data);
		bool insertDataToHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name, const vector<Mat>& data, int filterSize, int patchSize);


		//************** Tiff file  read ***************************//
		Mat readTiff(string filepath);

		
		//************** Prepare dataset for KNN / Random Forest ***************************//
		// split the data into train/test set balancely in different classes
		// return the index of the test data in original data
		// fold : crossvalidation number,an integer between {1, 100 / (100 - percentOfTrain)}
		vector<int> DivideTrainTestData(const vector<Mat>& data, const vector<unsigned char>& data_label, int percentOfTrain,
			vector<Mat>& train_img, vector<unsigned char>& train_label, vector<Mat>& test_img, vector<unsigned char>& test_label, int fold);
		vector<int> shuffleDataSet(vector<Mat>& data, vector<unsigned char>& data_label);
		Mat getConfusionMatrix(const map<unsigned char, string>& className, vector<unsigned char>& classResult, vector<unsigned char>& testLabels);

};
#endif
