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

	void applyKNN(vector<Mat> data, vector<unsigned char> data_labels, int k, int trainPercent);

private:
	void KNNTest(const vector<Mat>& trainVal, const vector<unsigned char>& trainLabels, const vector<Mat>& testVal, const vector<unsigned char>& testLabels, int k);

	double Euclidean(Mat& testVal, Mat& trainVal);

	unsigned char Classify(vector<pair<double, unsigned char>>& distVec, int k);
};


#endif
