#pragma once
#ifndef KNN_HPP_
#define KNN_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class KNN {
public:
	KNN() {
	}
	~KNN() {
	}
	void applyKNN(const vector<Mat>& data, const vector<unsigned char> & data_labels, int k, int trainPercent, vector<unsigned char>& class_result);

private:
	// return the classify result for the test data
	float KNNTest(const vector<Mat>& trainVal, const vector<unsigned char>& trainLabels, const vector<Mat>& testVal, const vector<unsigned char>& testLabels, int k, vector<unsigned char>& test_result);

	float Euclidean(Mat& testVal, Mat& trainVal);

	unsigned char Classify(vector<pair<float, unsigned char>>& distVec, int k);
};


#endif
