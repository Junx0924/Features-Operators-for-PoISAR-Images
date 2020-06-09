#ifndef  CVFEATURES_HPP_
#define  CVFEATURES_HPP_
#include <opencv2/opencv.hpp>
#include "glcm.hpp"
#include "elbp.hpp"
#include "MPEG-7/Feature.h"
#include "mp.hpp"


using namespace std;
using namespace cv;


namespace cvFeatures {

	// get LBP feature for grayscale img 
	Mat GetLBP(const Mat &src, int radius =1, int neighbors =8, int histsize =32);

	// get GLCM features for grayscale img 
	Mat GetGLCM(const Mat &src,int winsize = 7, GrayLevel level = GrayLevel::GRAY_16, int histsize =32);

	// get color features for BGR img
	Mat GetMPEG7DCD(const Mat& src, int numOfColor =3);
	Mat GetMPEG7CSD(const Mat& src, int size =32);

	// get opening-closing by reconstruction profile for grayscale img
	Mat GetMP(const Mat& src, const array<int, 3>& morph_size = { 1,2,3 });

	// Compute min, max, mean, std, median for single channel mat
	Mat GetStatistic(const Mat& src);
	
	Mat GetHistOfMaskArea(const Mat& src, const Mat& mask, int minVal=0, int maxVal=255, int histSize =32, bool normed = true);
};

#endif
