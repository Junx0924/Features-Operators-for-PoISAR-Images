#pragma once
#ifndef FEATUREPROCESS_HPP_
#define FEATUREPROCESS_HPP_
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include "KNN.hpp"
#include "../tsne/tsne.h"

namespace featureProcess{

	std::vector<int> DivideTrainTestData(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_label, int percentOfTrain,
		std::vector<cv::Mat>& train_img, std::vector<unsigned char>& train_label, std::vector<cv::Mat>& test_img, std::vector<unsigned char>& test_label, int fold);
	
	// shuffle the data and return the index of original data
	std::vector<int> shuffleDataSet(std::vector<cv::Mat>& data, std::vector<unsigned char>& data_label);
	
	cv::Mat getConfusionMatrix(const std::map<unsigned char, std::string>& className, std::vector<unsigned char>& classResult, std::vector<unsigned char>& testLabels);

	void applyML(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_labels, int trainPercent, 
		const std::string& classifier_type, std::vector<unsigned char>& class_result,int k=20);
	
	float calculatePredictionAccuracy(const std::string& feature_name, const std::vector<unsigned char>& classResult, 
		const std::vector<unsigned char>& groundtruth, const std::map<unsigned char, std::string>& className);

	cv::Mat featureDimReduction(const cv::Mat& feature, int new_dims = 2);

};

#endif
