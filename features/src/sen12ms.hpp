#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

// two types of masks
// choose in (IGBP, LCCS,ALL)
enum class MaskType
{
	IGBP,
	LCCS 
};

class sen12ms {
public:
	 vector<std::string>  s1FileList;
	 vector<std::string>  lcFileList;
	 vector<Mat> * list_images; 
	 vector<vector<Mat>>*  list_masks;
	 vector<vector<unsigned char>>* list_classValue;

public:
	// Constructor
	sen12ms(const string& s1FileListPath, const string & lcFileListPath) {
		LoadFileList(s1FileListPath, lcFileListPath);
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

	
	// Load image and its masks, class value of the mask to memory
	void LoadDataToMemeory(int BatchSize , MaskType mask_type);

	// get data format for torch
	void ProcessData(vector<Mat>& imageOfMaskArea, vector<unsigned char>& classValue);

	// Get PNG files for images and maskes
	void GeneratePNG(const string& outputpath, MaskType mask_type);

	// get the class name
	string GetClassName(unsigned char classValue,MaskType mask_type);

	/*-----------still working---------*/
	// get feature vectors (LBP,GLCM, Statistic,MPEG-7 DCD,SCD)
	// void ProcessData(vector<vector<float>>& features, vector<unsigned char>& classValue);

private:
	// Load tiff file list
	void LoadFileList(const string& s1FileListPath, const string& lcFileListPath);

	// Generate false color image from SAR data
	// R: VV, G:VH, B: VV/VH
	Mat GetFalseColorImage(const Mat& src, bool normed);

	// Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
	//Generate IGBP, LCCS from the ground truth 
	void GetClassCategory(const Mat& lc, Mat& IGBP, Mat& LCCS);

	// Create Masks for each patch
	void GetMask(const Mat& lc, vector<Mat>& IGBP_mask, vector<Mat>& LCCS_mask, vector<unsigned char>& IGBP_list, vector<unsigned char>& LCCS_list);

	// get polarimetric min, max, mean, std, median of mask area 
	Mat GetPolStatistic(const Mat& src, const Mat& mask);

	// Caculate the historgram vector of a mat with mask
	Mat GetHistWithMask(const Mat& src, const Mat& mask, int minVal, int maxVal, int histSize, bool normed);

	//check if cetain class type existed in a class category
	bool FindLandClass(const Mat& src, vector<std::pair<int, int> >& ind, const unsigned char& landclass);
};
 