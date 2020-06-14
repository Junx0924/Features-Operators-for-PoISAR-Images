#pragma once
#ifndef  SPECKLEFILTER_HPP_
#define  SPECKLEFILTER_HPP_
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class RefinedLee {
private:
	float NonValidPixelValue = -1.0;

	int filterSize;
	int stride;
	int subWindowSize;
	float sigmaVSqr;

public:
	// Constructor
	// filter size choose from (5, 7, 9, 11)
	RefinedLee(int filter_size, int numLooks) {
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
			cout << "Unknown filter size: " << filterSize << endl;
			exit(-1);
		}

		float sigmaV = 1.0 / std::sqrt(numLooks);
		sigmaVSqr = sigmaV * sigmaV;
	}

	~RefinedLee() {}

	void filterFullPol(  Mat& hh, Mat& vv, Mat & hv);
	//void filterDualPol();

private:
	float computePixelValueUsingLocalStatistics(const Mat& neighborPixelValues, int numSamples);

	float computePixelValueUsingEdgeDetection(const Mat& neighborPixelValues, const Mat& neighborSpanValues);

	int getLocalData(int x, int y, const Mat& src, const Mat& span, Mat& neighborPixelValues, Mat& neighborSpanValues);

	float getLocalMeanValue(const Mat& neighborValues, int numSamples);

	float getLocalVarianceValue(const Mat& neighborValues, int numSamples, float mean);

	// Compute the span image from the trace of the covariance or coherence matrix for the pixel
	void createSpanImage(const vector<Mat>& covariance, Mat& span);

	// Compute mean values for the 3x3 sub-areas in the sliding window 
	void computeSubAreaMeans(const Mat& neighborPixelValues, Mat& subAreaMeans);

	// Compute the gradient in 3x3 subAreaMeans 
	int getDirection(const Mat& subAreaMeans);

	// Get pixel values from the non-edge area indicated by the given direction
	int getNonEdgeAreaPixelValues(const Mat& neighborPixelValues, int d, Mat& pixels);
};


#endif