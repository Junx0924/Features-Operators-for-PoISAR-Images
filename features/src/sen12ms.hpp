#ifndef SEN12MS_HPP
#define SEN12MS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include "glcm.hpp"
#include "elbp.hpp"
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
	

private:
	 vector<Mat>* list_images;  // false color images(unnormalized)
	 vector<vector<Mat>>* list_masks; // masks for each false color image
	 vector<vector<unsigned char>>* list_classValue; // class labels for each mask
	 int batchSize;
	 MaskType mask_type;

public:
	// Constructor
	sen12ms(const string& s1FileListPath, const string & lcFileListPath) {
		LoadFileList(s1FileListPath, lcFileListPath);
		list_images = new vector<Mat>();
		list_masks = new vector<vector<Mat>>();
		list_classValue = new vector<vector<unsigned char>>();
		batchSize = 0;
		mask_type = MaskType::IGBP;
	}

	sen12ms(vector<std::string>  &s1List, vector<std::string> & lcList) {
		s1FileList = s1List;
		lcFileList = lcList;
		list_images = new vector<Mat>();
		list_masks = new vector<vector<Mat>>();
		list_classValue = new vector<vector<unsigned char>>();
		batchSize = 0;
		mask_type = MaskType::IGBP;
	}

	~sen12ms()
	{
		if(list_images){ delete list_images; }
		if(list_masks) { delete list_masks; }
		if(list_classValue) { delete list_classValue; }
	}

	
	void SetMaskType(MaskType maskType) {
		mask_type = maskType;
	}

	void SetBatchSize(int size) {
		batchSize = size;

		if (!list_images) free(list_images); 
		if (!list_masks) free(list_masks);
		if (!list_classValue) free(list_classValue);

		list_images = new vector<Mat>(batchSize);
		list_masks = new vector<vector<Mat>>(batchSize);
		list_classValue = new vector<vector<unsigned char>>(batchSize);
	}

	// load current batch to memory 
	void LoadBatchToMemeory(int batch);

	// be careful to use this func
	void LoadAllToMemory();
	
	// get the image of mask area and its class
	void ProcessData(vector<Mat>& imageOfMaskArea, vector<unsigned char>& classValue);

	// get LBP feature of mask area 
	void GetFeatureLBP(vector<Mat>& features, vector<unsigned char>& classValue, int radius,int neighbors, int histsize);

	// get GLCM features on each channel of mask area
	void GetFeatureGLCM(vector<Mat>& features, vector<unsigned char>& classValue, int winsize, GrayLevel level, int histsize);

	// get statistic features of all channels of mask area
	void GetFeatureStatistic(vector<Mat>& features, vector<unsigned char>& classValue,int histsize);

	//  void GetMPEG7DCD(vector<Mat>& features, vector<unsigned char>& classValue, int numOfColor);

	// Get PNG files for images and maskes
	void GeneratePNG(const string& outputpath);

	string GetClassName(unsigned char classValue);

	

private:
	// Load tiff file list
	void LoadFileList(const string& s1FileListPath, const string& lcFileListPath);

	// Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
	//Generate IGBP, LCCS from the ground truth 
	void GetClassCategory(const Mat& lc, Mat& IGBP, Mat& LCCS);
	
	//check if cetain class type existed in a class category
	bool FindLandClass(const Mat& src, vector<std::pair<int, int> >& ind, const unsigned char& landclass);

	// Create Masks for each patch
	void GetMask(const Mat& lc, vector<Mat>& list_masks, vector<unsigned char>& list_classValue);

	//read tiff file
	Mat ReadTiff(string filepath);

	// Generate false color image from SAR data
	// R: VV, G:VH, B: VV/VH
	Mat GetFalseColorImage(const Mat& src, bool normed);

	// get polarimetric min, max, mean, std, median of mask area 
	Mat GetPolStatistic(const Mat& src, const Mat& mask);

	// Caculate the historgram vector of a mat with mask
	Mat GetHistOfMaskArea(const Mat& src, const Mat& mask, int minVal, int maxVal, int histSize, bool normed);
};

#endif
 