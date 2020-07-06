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

	//************** Prepare dataset for ML ***************************//
	// split the data into train/test set balancely in different classes
	// return the index of the test data in original data
	// fold : crossvalidation number,an integer between {1, 100 / (100 - percentOfTrain)}
	std::vector<int> DivideTrainTestData(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_label, int percentOfTrain,
		std::vector<cv::Mat>& train_img, std::vector<unsigned char>& train_label, std::vector<cv::Mat>& test_img, std::vector<unsigned char>& test_label, int fold);
	
	std::vector<int> shuffleDataSet(std::vector<cv::Mat>& data, std::vector<unsigned char>& data_label);
	
	cv::Mat getConfusionMatrix(const std::map<unsigned char, std::string>& className, std::vector<unsigned char>& classResult, std::vector<unsigned char>& testLabels);



	// classifier_type: choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
    // do cross validation on each test part, return the class results
	void applyML(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_labels, int trainPercent, const std::string& classifier_type, std::vector<unsigned char>& class_result);
	
	// calculate the accuracy for each class, and return the overal accuracy
	float calculatePredictionAccuracy(const std::string& feature_name, const std::vector<unsigned char>& classResult, const std::vector<unsigned char>& testLabels);

	// input: feature, each row is a vector with size (feature.cols)
	// return: Mat, each row is a dim reduced vector with size(new_dims)
	cv::Mat featureDimReduction(const cv::Mat& feature, int new_dims = 2);

};

#endif