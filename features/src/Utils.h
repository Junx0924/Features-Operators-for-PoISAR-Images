#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<opencv2/opencv.hpp>

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
		void getSafeSamplePoints(const Mat& img, const int & samplePointNum, const int& sampleSize, vector<Point>& pts);

		void shuffleDataSet(vector<Mat>& data, vector<unsigned char>& data_label);

		void DivideTrainTestData(const vector<Mat>& data, const vector<unsigned char>& data_label, int percentOfTrain,
			vector<Mat>& train_img, vector<unsigned char>& train_label, vector<Mat>& test_img, vector<unsigned char>& test_label);

		Mat convertDataToPNG(const Mat& src);

		Mat getConfusionMatrix(const map<unsigned char, string>& className, vector<unsigned char>& classResult, vector<unsigned char>& testLabels);
};
#endif
