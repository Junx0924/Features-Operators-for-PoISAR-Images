#pragma once
#ifndef  SPECKLEFILTER_HPP_
#define  SPECKLEFILTER_HPP_
#include <opencv2/opencv.hpp>


class RefinedLee {
private:
	float NonValidPixelValue = -1.0;

	int filterSize = 9;
	int stride;
	int subWindowSize;
	float sigmaVSqr;

public:
	// Constructor
	// filter size choose from (5, 7, 9, 11)
	RefinedLee(int filter_size, int numLooks = 1) {
		filterSize = filter_size;
		switch (filterSize) {
		case 5:
			subWindowSize = 3;
			stride = 1;
			break;
		case 7:
			subWindowSize = 3;
			stride = 2;
			break;
		case 9:
			subWindowSize = 5;
			stride = 2;
			break;
		case 11:
			subWindowSize = 5;
			stride = 3;
			break;
		default:
			std::cout << "Unknown filter size: " << filterSize << std::endl;
			exit(-1);
		}

		float sigmaV = 1.0f / std::sqrt(numLooks);
		sigmaVSqr = sigmaV * sigmaV;
	}

	~RefinedLee() {}

	void filterFullPol(cv::Mat& hh, cv::Mat& vv, cv::Mat& hv);
	void filterFullPol(cv::Mat& hh, cv::Mat& vv, cv::Mat& hv, cv::Mat& span);

private:
	float computePixelValueUsingLocalStatistics(const cv::Mat& neighborPixelValues, int numSamples);

	float computePixelValueUsingEdgeDetection(const cv::Mat& neighborPixelValues, const cv::Mat& neighborSpanValues);

	int getLocalData(int x, int y, const cv::Mat& src, const cv::Mat& span, cv::Mat& neighborPixelValues, cv::Mat& neighborSpanValues);

	float getLocalMeanValue(const cv::Mat& neighborValues, int numSamples);

	float getLocalVarianceValue(const cv::Mat& neighborValues, int numSamples, float mean);

	// Compute mean values for the 3x3 sub-areas in the sliding window 
	void computeSubAreaMeans(const cv::Mat& neighborPixelValues, cv::Mat& subAreaMeans);

	// Compute the gradient in 3x3 subAreaMeans 
	int getDirection(const cv::Mat& subAreaMeans);

	// Get pixel values from the non-edge area indicated by the given direction
	int getNonEdgeAreaPixelValues(const cv::Mat& neighborPixelValues, int d, cv::Mat& pixels);
};


#endif