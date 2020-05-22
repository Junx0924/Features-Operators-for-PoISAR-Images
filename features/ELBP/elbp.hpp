#ifndef ELBP_HPP_
#define ELBP_HPP_ 

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using namespace cv;
namespace elbp {

	// templated functions
	template <typename _Tp>  void elbp_(const Mat &src, Mat &dst, int radius, int neighbors);
	

	// wrapper functions
	void elbp(const Mat& src, Mat & dst, int radius, int neighbors);
	

	// Mat return type functions
	Mat elbp(const Mat& src, int radius, int neighbors, bool normed );
	Mat histc(const Mat& src, int minVal, int maxVal, bool normed );
	Mat spatial_histogram(const Mat& src, int numPatterns, int grid_x, int grid_y, bool normed);

	/*
	 use the extended lbp operator
	input: src, radius = 1 and neighbors = 8 to generate LBP image
	input: grid_x = 5, grid_y = 5, numPatterns = 32 to compute the spatial histogram of the LBP image
    return: elbp vector of row 1, col grid_x*grid_y*numPatterns, channel 1
	*/
	Mat computLBPvector(const Mat& src, int radius, int neighbors, int numPatterns, int grid_x, int grid_y, bool normed);


	//others
	std::string type2str(int type);
}


#endif
