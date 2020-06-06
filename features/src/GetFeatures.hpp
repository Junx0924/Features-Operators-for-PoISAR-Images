#ifndef  GETFEATURES_HPP_
#define  GETFEATURES_HPP_
#include <opencv2/opencv.hpp>
#include "glcm.hpp"
#include "elbp.hpp"
#include "MPEG-7/Feature.h"
#include "mp.hpp"


using namespace std;
using namespace cv;


class cvFeatures {

private:
	// 3 channel Mat, HH,HV,VV for full Pol, VV,VH,VV/VH for dual Pol
	Mat image; 
    unsigned char  class_type;

public:
//constructor
	cvFeatures(const Mat &src,const unsigned char &label) {
		 image = src;
		 class_type = label;
	}

	~cvFeatures() {
		 
	}

	// get texture features
	// get LBP feature of mask area 
	void GetLBP(vector<Mat>& features, vector<unsigned char> &classValue, int radius =1, int neighbors =8, int histsize =32);

	// get GLCM features for all channels of mask area 
	void GetGLCM(vector<Mat> &features, vector<unsigned char>& classValue,int winsize = 8, GrayLevel level = GrayLevel::GRAY_8, int histsize =32);

	// get color features 
	void GetMPEG7DCD(vector<Mat>& features, vector<unsigned char>& classValue, int numOfColor =3);
	void GetMPEG7CSD(vector<Mat>& features, vector<unsigned char>& classValue, int size =32);

	// get opening-closing by reconstruction profile for all channels (grayscale)
	void GetMP(vector<Mat>& features, vector<unsigned char>& classValue, const array<int, 3>& morph_size = { 1,2,3 });

	// Compute min, max, mean, std, median for all channels
	void GetStatistic(vector<Mat>& features, vector<unsigned char>& classValue);
	

private:
	Mat GetHistOfMaskArea(const Mat& src, const Mat& mask, int minVal=0, int maxVal=255, int histSize =32, bool normed = true);
};

#endif
