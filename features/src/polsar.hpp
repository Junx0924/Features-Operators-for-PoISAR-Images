#pragma once
#ifndef  POLSAR_HPP_
#define  POLSAR_HPP_
#include <opencv2/opencv.hpp>
/*defining compiler versions for some
compiler specific includes*/
#define VC				//GCC/VC

using namespace std;
using namespace cv;

//load data from Oberpfaffenhofen
class polsar {

private:
	// data = complex mat with values [HH, VV, HV]
	vector<Mat> data;
	vector<unsigned  char> labels;
	vector<Mat> masks;
	// record sample Points of each mask area
	vector<vector<Point>> samplePoints;

	// draw samples from each image of mask area
	int sampleSize = 64;
	// Maximum sample points of each mask area
	int samplePointNum = 1000;

	std::map<string, unsigned char>classNames;
	unsigned border = 3;

public:
	// constructor
	// input: rat file folder, label file folder, sample size, maximum sample points of each mask area
	polsar(const string & RATfileFolder, const string & labelFolder, const int &patchSize, const int &pointNum) {

		sampleSize = patchSize;
		samplePointNum = pointNum;

		loadData(RATfileFolder);
		vector<string>  labelNames;
		ReadClassLabels(labelFolder, labelNames, masks);

		// read the labelNames to dict
		for (int i = 0; i < labelNames.size(); i++) {
			classNames.insert(pair<string, unsigned char>(labelNames[i], i+1));
			labels.push_back(i + 1);

			// get the sample points in each mask area
			vector<Point> pts;
			getSafeSamplePoints(masks[i], pts);
			cout << "Get " << pts.size() << " sample points for class " << labelNames[i] << endl;
			samplePoints.push_back(pts);
		}
		
	}

	~polsar() {}

	  // data for torch
	 // get patches of 3 channel (HH+VV,HV,HH-VV) intensity(dB)
	 void GetPauliColorPatches(vector<Mat>& patches, vector<unsigned char>& classValue);
	 // get patches of 3 channel (HH,HV,VV) intensity(dB)
	 void GetPatches(vector<Mat>& patches, vector<unsigned char>& classValue);


	 // get statistical features(min,max,mean,median,std) on polsar parameters, size 1*35
	 void GetStatisticFeature(vector<Mat>& features, vector<unsigned char>& classValue);
	 // get texture features(LBP and GLCM) on HH,VV,VH, default size 1*64
	 void GetTextureFeature(vector<Mat>& features, vector<unsigned char>& classValue);
	 // get color features(MPEG-7 DCD,CSD) on Pauli Color image, default size 1*44
	 void GetColorFeature(vector<Mat>& features, vector<unsigned char>& classValue);
	 // get MP features on HH,VV,VH, default size (sampleSize*3,sampleSize)
	 void GetMPFeature(vector<Mat>& features, vector<unsigned char>& classValue);

	 //---------still working---------
	 // void GetTargetDecomposition(vector<Mat>& features, vector<unsigned char>& classValue);
	 // get elements from covariance matrix C and coherent matrix T
	 // void GetElementCT(vector<Mat>& features, vector<unsigned char>& classValue);

private:
	//Generate false color image
	// R:HH, G:HV, B:VV
	Mat getFalseColorImg(const Mat& hh, const Mat& vv, const Mat& hv,  bool normed = false);
	//R: HH+VV, G:HV, B: HH-VV
	Mat getPauliColorImg(const Mat& hh, const Mat& vv, const Mat& hv);

	// Generate samples from each img
	void getSafeSamplePoints(const Mat& mask, vector<Point>& pts);


	// process complex scattering values
	Mat getComplexAmpl(const Mat& in);
	Mat getComplexAngle(const Mat& in);
	Mat logTransform(const Mat& in);  //intensity in dB
	
	Mat getPhaseDiff(const Mat& hh, const Mat& vv);

	 
	


	/***Author: Anupama Rajkumar***/
	void loadData(string RATfolderPath);
	void ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages);
	Size loadRAT(string fname, vector<Mat>& data, bool metaOnly = false);
	Size loadRAT2(string fname, vector<Mat>& data, bool metaOnly = false);
	void getTileInfo(cv::Size size, unsigned border, unsigned& tile, vector<unsigned>& tileSize, vector<unsigned>& tileStart);
};

#endif