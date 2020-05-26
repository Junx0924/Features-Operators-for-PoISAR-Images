#ifndef ELBP_HPP_
#define ELBP_HPP_ 

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using namespace cv;
namespace elbp {

	// templated functions
	template <typename _Tp>  void ElbpWrapper_(const Mat &src, Mat &dst, int radius, int neighbors);
	

	// wrapper functions
	void ElbpWrapper(const Mat& src, Mat & dst, int radius, int neighbors);
	

	// compute the elbp of the whole image
	Mat CaculateElbp(const Mat& src, int radius, int neighbors, bool normed );


	//others
	std::string type2str(int type);
}


#endif
