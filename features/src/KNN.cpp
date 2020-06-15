#include "KNN.hpp"
#include "Utils.h"

#include <iostream>
#include <math.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <cmath>

using namespace std;
using namespace cv;

void KNN::applyKNN(vector<Mat> data, vector<unsigned char> data_labels, int k, int trainPercent) {
	Utils::shuffleDataSet(data, data_labels);
	vector<Mat> train;
	vector<unsigned char> train_labels;
	vector<Mat> test;
	vector<unsigned char> test_labels;
	Utils::DivideTrainTestData(data, data_labels, trainPercent, train, train_labels, test, test_labels);
	KNNTest(train, train_labels, test, test_labels, k);
}


/***********************************************************************
Calculating euclidean distance between image and label map points
Author : Anupama Rajkumar
Date : 12.06.2020
Description: This function is used to calculate the Euclidean distance 
between the training and test samples
*************************************************************************/

double KNN::Euclidean(Mat& testVal, Mat& trainVal) {
	double distance = 0.0;
	double sum = 0.0;
	//ensure that the dimensions of testVal and trainVal are same
	if ((testVal.rows != trainVal.rows) || (testVal.cols != trainVal.cols))
	{
		cerr << "Matrix dimensions should be same for distance calculation" << endl;
		exit(-1);
	}
	else {
		int row = testVal.rows;
		int col = testVal.cols;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				double diff = testVal.at<float>(i, j) - trainVal.at<float>(i, j);
				sum = sum + pow(diff, 2);
				distance = sqrt(sum);
			}
		}
	}

	//cout << "distance = " << distance;
	return distance;
}




/***********************************************************************
Verifying KNN Classifier with random samples - Overloaded Function
Author : Anupama Rajkumar
Date : 28.05.2020
Modified by: Jun Xiang 15.06.2020
Description : This function counts the number of classes in k neighborhood
Based on which class has the highest count, appropriate class label is returned
*************************************************************************/
unsigned char KNN::Classify(vector<pair<double, unsigned char>>& distVec, int k) {
	
	for (int i = k; i >0; --i) {
		float max = -1;
		unsigned char max_label = static_cast<unsigned char>(0);
		map<unsigned char, int> count;

		//In case max index cannot be found in K elements, then check K-1 elements
		for(int j=0; j< i; ++j){
			unsigned char c = distVec[i].second;
			count[c]++;
		}

		// get the max label
		for (auto it = count.begin(); it != count.end(); it++)
		{
			unsigned char temp = it->first;
			int elem = it->second;
			if (elem >= max) {
				max = elem;
				max_label = temp;
			}
		}

		bool only_one_max = false;
		// Find if there is only one max value
		for (auto it= count.begin(); it != count.end(); it++)
		{
			unsigned char temp = it->first;
			int elem = it->second;
			if ((elem == max) && (temp != max_label)) {
				only_one_max = true;
				break;
			}
		}

		if (only_one_max == true)
			continue;
		else {
			return max_label;
		}

	}
	return  static_cast<unsigned char>(0); // 0 is Unclassified
}


/***********************************************************************
Verifying KNN Classifier with random samples
Author : Anupama Rajkumar
Date : 12.06.2020
Description: Classify test points using KNN Classifier
*************************************************************************/
void KNN::KNNTest(const vector<Mat>& trainVal, const vector<unsigned char>& trainLabels, const vector<Mat>& testVal, const vector<unsigned char>& testLabels, int k) {
	/*for each sample in the testing data, caculate distance from each training sample */
	vector<unsigned char> classResult;
	for (int i = 0; i < testVal.size(); i++) {								//for each test sample
		vector<pair<double, unsigned char>> distVec;
		for (int j = 0; j < trainVal.size(); j++) {							//for every training sample
			pair<double, unsigned char> dist;
			Mat test = testVal[i];
			Mat train = trainVal[j];
			dist.first = this->Euclidean(test,train);			//calculate euclidean distance
			dist.second = trainLabels[j];
			distVec.push_back(dist);
		}
		//sort the distance in the ascending order
		sort(distVec.begin(), distVec.end());
		//classify for each row the label patch
		unsigned char classVal = this->Classify(distVec, k);
		classResult.push_back(classVal);
	}	
	double accuracy = Utils::calculatePredictionAccuracy(classResult, testLabels);
	cout << "Accuracy: " << accuracy << endl;
}


