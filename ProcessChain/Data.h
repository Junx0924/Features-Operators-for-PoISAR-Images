#pragma once

#ifndef DATA_H
#define DATA_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Data {
public:
	/*constructors and desctructors*/
	 Data(void);
	~Data(void);

	/***********Functions***************/
	void loadData(string folder);
	void loadImage(string fname);
	void loadLabels(const string &folderPath, int numOfClasses);
	void loadPolSARData(std::vector<std::string> const& fname);
	void getTileInfo(cv::Size size, unsigned border, unsigned &tile, vector<unsigned> &tileSize, vector<unsigned> &tileStart);
	cv::Size loadRAT(string fname, vector<Mat> &data, bool metaOnly = false);
	cv::Size loadRAT2(string fname, vector<Mat> &data, bool metaOnly = false);
	/***********Functions***************/
private:
	/***********Variables***************/
	// data = scattering vector with values [HH, VV, HV]
	std::vector<cv::Mat> data;
	unsigned border = 3;
	/***********Variables***************/
};


#endif