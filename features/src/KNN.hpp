#pragma once
#ifndef KNN_HPP_
#define KNN_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>


class KNN {
public:
	KNN() {
	}
	~KNN() {
	}
	void applyKNN(const std::vector<cv::Mat>& data, const std::vector<unsigned char> & data_labels, int k, int trainPercent, std::vector<unsigned char>& class_result);

//private:
	// return the classify result for the test data
	float KNNTest(const std::vector<cv::Mat>& trainVal, const std::vector<unsigned char>& trainLabels, const std::vector<cv::Mat>& testVal, const std::vector<unsigned char>& testLabels, int k, std::vector<unsigned char>& test_result);

	float Euclidean(cv::Mat& testVal, cv::Mat& trainVal);

	unsigned char Classify(std::vector<std::pair<float, unsigned char>>& distVec, int k);


};


#endif
