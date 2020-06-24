#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include "cv_hdf5.hpp"

using namespace std;
using namespace cv;

namespace Utils {
	    // by Anupama Rajkumar
		void WriteToFile(Mat& labelMap, string& fileName);
		map<string, Vec3f> loadLabelsMetadata();
		Mat_<Vec3f> visualiseLabels(Mat &image, string& imageName);
		void GetLabelPatchIndex(int sizeOfPatch, Point2i samplePoints, Mat& LabelMap, int& pStart_r, int& pStart_c, int& pEnd_r, int& pEnd_c);
		void Visualization(string& fileName, string& imageName, Size size);

		double calculatePredictionAccuracy(const vector<unsigned char>& classResult, const vector<unsigned char>& testLabels);

		//by Jun
		void shuffleDataSet(vector<Mat>& data, vector<unsigned char>& data_label);
		Mat getConfusionMatrix(const map<unsigned char, string>& className, vector<unsigned char>& classResult, vector<unsigned char>& testLabels);
		void getFeaturesFromHDF(const String& hdf5_fileName, const String& parent_name, const vector<String>& dataset_name,
			vector<Mat>& features, vector<unsigned char>& featureLabels, vector<Point>& labelPoints, int filterSize, int patchSize);
	    void getSafeSamplePoints(const Mat& img, const int& samplePointNum, const int& sampleSize, vector<Point>& pts);


		void DivideTrainTestData(const vector<Mat>& data, const vector<unsigned char>& data_label, int percentOfTrain,
			vector<Mat>& train_img, vector<unsigned char>& train_label, vector<Mat>& test_img, vector<unsigned char>& test_label);
		bool checkExistInHDF(const String& filename, const String& parent_name, const string& dataset_name);
		bool checkExistInHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name, int filterSize, int patchSize);
		void readDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, vector<Mat>& data, int filterSize, int patchSize);
		void readDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name, Mat& data);
		bool insertDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& data);
		bool insertDataToHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name, const vector<Mat>& data, int filterSize, int patchSize);
};
#endif
