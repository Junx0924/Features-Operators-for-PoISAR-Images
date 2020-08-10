#pragma once
#ifndef  DATA_HPP_
#define  DATA_HPP_
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <iostream>
#include <fstream>
#include <random>


/*defining compiler versions for some
compiler specific includes*/
#define VC				//GCC/VC



//load data from Oberpfaffenhofen
class Data{

private:
	unsigned border = 3;

public:
	// data = complex mat with values [HH, VV, HV]
	std::vector<cv::Mat> data;
	cv::Mat LabelMap;
	// record the class name of each label
	std::map<unsigned char, std::string>classNames;

	// constructor
	// input: rat file folder, label file folder 
	Data(const std::string& RATfileFolder, const std::string& labelFolder) {

		// read rat data, can't save them directly to hdf5, it will lost precision
		loadData(RATfileFolder);

		// read labels
		std::vector<cv::Mat> masks;
		std::vector<std::string>  labelNames;
		ReadClassLabels(labelFolder, labelNames, masks);
		this->LabelMap = this->generateLabelMap(masks);

		// read the labelNames to dict
		classNames[unsigned char(0)] = "Unclassified";
		std::cout << "Unclassified" << " label : " << std::to_string(0) << std::endl;
		for (int i = 0; i < labelNames.size(); i++) {
			classNames.insert(std::pair<unsigned char, std::string>(i + 1, labelNames[i]));
			std::cout << labelNames[i] << " label : " << std::to_string(i+1) << std::endl;
		}
	}
	~Data() {
		
	}

private:
	/***Author: Anupama Rajkumar***/
	cv::Mat generateLabelMap(const std::vector<cv::Mat>& masks);
	void loadData(std::string RATfolderPath);
	void ReadClassLabels(std::string labelPath, std::vector<std::string>& labelNames, std::vector<cv::Mat>& labelImages);
	cv::Size loadRAT(std::string fname, std::vector<cv::Mat>& data, bool metaOnly = false);
	cv::Size loadRAT2(std::string fname, std::vector<cv::Mat>& data, bool metaOnly = false);
	void getTileInfo(cv::Size size, unsigned border, unsigned& tile, std::vector<unsigned>& tileSize, std::vector<unsigned>& tileStart);
};

#endif