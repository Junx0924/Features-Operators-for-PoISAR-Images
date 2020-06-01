#ifndef SEN12MS_HPP
#define SEN12MS_HPP

#include <opencv2/opencv.hpp>
#include <string>
//#include "glcm.hpp"
//#include "elbp.hpp"
using namespace cv;
using namespace std;

// two types of masks
// choose in (IGBP, LCCS)
enum class MaskType
{
	IGBP = 0,
	LCCS = 1
};

class sen12ms {
public:
	 vector<std::string>  s1FileList;
	 vector<std::string>  lcFileList;
	 vector<Mat> * list_images;  // false color images(unnormalized)
	 vector<vector<Mat>>*  list_masks; // masks for each false color image
	 vector<vector<unsigned char>>* list_classValue; // class labels for each mask

public:
	// Constructor
	sen12ms(const string& s1FileListPath, const string & lcFileListPath) {
		LoadFileList(s1FileListPath, lcFileListPath);
		list_images = new vector<Mat>();
		list_masks = new vector<vector<Mat>>();
		list_classValue = new vector<vector<unsigned char>>();
		
	}
	sen12ms(vector<std::string>  &s1List, vector<std::string> & lcList) {
		s1FileList = s1List;
		lcFileList = lcList;
		list_images = new vector<Mat>();
		list_masks = new vector<vector<Mat>>();
		list_classValue = new vector<vector<unsigned char>>();
	}

	~sen12ms()
	{
		if(list_images){ delete list_images; }
		if(list_masks) { delete list_masks; }
		if(list_classValue) { delete list_classValue; }
	}

	void LoadDataToMemeory(int BatchSize , MaskType mask_type);
	
	// for torch dataloader
	void ProcessData(vector<Mat>& imageOfMaskArea, vector<unsigned char>& classValue);

	/*-----------still working---------*/
	// get feature vectors (LBP,GLCM, Statistic,MPEG-7 DCD,SCD)
	//  void GetFeatureLBP(vector<Mat>& features, vector<unsigned char>& classValue, int radius,int neighbors, int histsize);
	//  void GetFeatureGLCM(vector<Mat>& Energy, vector<Mat>& Contrast, vector<Mat>& Homogenity, vector<Mat>& Entropy,int winsize, GrayLevel level, int histsize);

	// Get PNG files for images and maskes
	void GeneratePNG(const string& outputpath, MaskType mask_type);

	// get polarimetric min, max, mean, std, median of mask area 
	Mat GetPolStatistic(const Mat& src, const Mat& mask);

	string GetClassName(unsigned char classValue, MaskType mask_type);

	// Caculate the historgram vector of a mat with mask
	Mat GetHistWithMask(const Mat& src, const Mat& mask, int minVal, int maxVal, int histSize, bool normed);

private:
	// Load tiff file list
	void LoadFileList(const string& s1FileListPath, const string& lcFileListPath);

	// Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
	//Generate IGBP, LCCS from the ground truth 
	void GetClassCategory(const Mat& lc, Mat& IGBP, Mat& LCCS);
	
	//check if cetain class type existed in a class category
	bool FindLandClass(const Mat& src, vector<std::pair<int, int> >& ind, const unsigned char& landclass);

	// Create Masks for each patch
	void GetMask(const Mat& lc, vector<Mat>& list_masks, vector<unsigned char>& list_classValue, MaskType mask_type);

	//read tiff file
	Mat ReadTiff(string filepath);

	// Generate false color image from SAR data
	// R: VV, G:VH, B: VV/VH
	Mat GetFalseColorImage(const Mat& src, bool normed);
};

#endif
 