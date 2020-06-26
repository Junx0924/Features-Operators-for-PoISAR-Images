#ifndef SEN12MS_HPP_
#define SEN12MS_HPP_

#include <opencv2/opencv.hpp>
#include "Geotiff.hpp"
#include "Utils.h"
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

// two types of masks
// choose in (IGBP, LCCS)
enum class MaskType
{
	IGBP = 0,
	LCCS = 1
};



// to load sen12ms dataset
class sen12ms {
public:
	 vector<Mat>  list_images;  // false color images(unnormalized)
	 vector<Mat>  list_labelMaps; // label map
	 std::map<unsigned char, string> ClassName;

private:
	 vector<std::string>  s1FileList;
	 vector<std::string>  lcFileList;
	
	 int batchSize;
	 MaskType mask_type= MaskType::IGBP;
	 // draw samples from each image of mask area
	 int sampleSize = 10;
	 // Maximum sample points of each mask area
	 int samplePointNum = 100;
public:
	// Constructor
	sen12ms(const string& s1FileListPath, const string & lcFileListPath) {
		loadFileList(s1FileListPath, lcFileListPath);
		list_images =  vector<Mat>();
		list_labelMaps =   vector<Mat>();
	}

	sen12ms(vector<std::string>  &s1List, vector<std::string> & lcList) {
		s1FileList = s1List;
		lcFileList = lcList;
		list_images = vector<Mat>();
		list_labelMaps =  vector<Mat>();
		 
	}

	~sen12ms()
	{
		 
	}

	
	void SetMaskType(MaskType maskType) {
		this->mask_type = maskType;
		this->ClassName = GetClassName(maskType);
	}

	void SetBatchSize(int size) {
		batchSize = size;

		if (!list_images.empty())  list_images.clear(); 
		if (!list_labelMaps.empty())  list_labelMaps.clear();

		list_images =  vector<Mat>(batchSize);
		list_labelMaps =  vector<Mat>(batchSize);
	}

	// set the sample size and Maximum sample points of each mask area
	void SetSample(const int &size,const int & num) {
		sampleSize = size;
		samplePointNum = num;
	}

	// load current batch to memory 
	void LoadBatchToMemeory(int batch);

	// be careful to use this function
	void LoadAllToMemory();
	
	// get training data
	void GetPatches(vector<Mat>& patches, vector<unsigned char>& classValue);

	// Get PNG files for images and maskes
	void GeneratePNG(const string& outputpath);

	

private:
	// Load tiff file list
	void loadFileList(const string& s1FileListPath, const string& lcFileListPath);

	// Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
	//Generate IGBP, LCCS from the ground truth 
	void getLabelMap(const Mat& lc, Mat& labelMap);
	
	//check if cetain class type existed in a class category
	bool findLandClass(const Mat& labelMap, vector<std::pair<int, int> >& ind, const unsigned char& landclass);

	// Create Masks for each patch
	void getMask(const Mat& labelMap, vector<Mat>& list_masks, vector<unsigned char>& list_classValue);

	// Generate false color image from SAR data
	// R: VV, G:VH, B: VV/VH
	Mat getFalseColorImage(const Mat& src, bool normed);

	// Generate samples from each img
	void getSamples(const Mat& img, const Mat& mask, const unsigned char& mask_label, vector<Mat>& samples, vector<unsigned char>& sample_labels);

	std::map<unsigned char, string> GetClassName(MaskType maskType);
};

#endif
 