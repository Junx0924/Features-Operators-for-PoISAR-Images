#ifndef SEN12MS_HPP_
#define SEN12MS_HPP_ 

#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

using namespace cv;
namespace sen12ms {
	// Generate false color image from SAR data
	// R: VV, G:VH, B: VV/VH
	Mat GetFalseColorImage(const Mat& src, bool normed);

	// Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
	//Generate IGBP, LCCS from the ground truth 
	void GetClassCategory(const Mat& lc, Mat& IGBP, Mat& LCCS);

	// Create Masks for each class category
	void GetMask( const Mat& lc, vector<Mat> &IGBP_mask, vector<Mat> &LCCS_mask, vector<unsigned char> & IGBP_list,vector<unsigned char>&  LCCS_list);

	// get polarimetric min, max, mean, std, median of mask area 
	Mat GetPolStatistic(const Mat& src, const Mat& mask);

	// Caculate the historgram vector of a mat with mask
	Mat GetHistWithMask(const Mat& src, const Mat& mask, int minVal, int maxVal, int histSize, bool normed);

	//check if cetain class type existed in a class category
	bool FindLandClass(const Mat& src, vector<std::pair<int, int> >& ind, const int& landclass);
}
#endif