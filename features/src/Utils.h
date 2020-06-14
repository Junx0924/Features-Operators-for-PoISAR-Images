#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace Utils {
		void WriteToFile(Mat& labelMap, string& fileName);
		map<string, Vec3f> loadLabelsMetadata();
		Mat_<Vec3f> visualiseLabels(Mat &image, string& imageName);
		void GetLabelPatchIndex(int sizeOfPatch, Point2i samplePoints, Mat& LabelMap, int& pStart_r, int& pStart_c, int& pEnd_r, int& pEnd_c);
		void Visualization(string& fileName, string& imageName, Size size);
		void DisplayClassName(int finalClass);

		//by Jun
		void getSafeSamplePoints(const Mat& img, const int & samplePointNum, const int& sampleSize, vector<Point>& pts);
		Mat writeImgToPNG(const Mat& src);
};
#endif
