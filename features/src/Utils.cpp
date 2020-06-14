#include "Utils.h"
#include "Ober.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/opencv.hpp>
#include "sarFeatures.hpp"

using namespace std;
using namespace cv;

/***********************************************************************
A helper function to store the classification data
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Using this function store classification values in csv files
that can be used later for data analysis
*************************************************************************/

void Utils::WriteToFile(Mat& labelMap, string& fileName) {
	ofstream distance_list;
	distance_list.open(fileName);

	for (int row = 0; row < labelMap.rows; row++) {
		for (int col = 0; col < labelMap.cols; col++) {
			distance_list << labelMap.at<float>(row, col) << ",";
		}
		distance_list << endl;
	}
}



/***********************************************************************
A helper function containing color metadata to be used when visualizing
Author : Eli Ionescu
Date : 27.05.2020
Description: Using this function creates a map of the colors and the labels
they correspond to. To be used with visualization
*************************************************************************/
map<string, Vec3f> Utils::loadLabelsMetadata()
{
	map<string, Vec3f> name_color;

	// Color is BGR not RGB!
	Vec3f red = Vec3f(49.0f, 60.0f, 224.0f);
	Vec3f blue = Vec3f(164.0f, 85.0f, 50.0f);
	Vec3f yellow = Vec3f(0.0f, 190.0f, 246.0f);
	Vec3f dark_green = Vec3f(66.0f, 121.0f, 79.0f);
	Vec3f light_green = Vec3f(0.0f, 189.0f, 181.0f);
	Vec3f black = Vec3f(0.0f, 0.0f, 0.0f);

	name_color["city"] = red;
	name_color["field"] = yellow;
	name_color["forest"] = dark_green;
	name_color["grassland"] = light_green;
	name_color["street"] = blue;
	name_color["unclassified"] = black;

	return name_color;
}

/***********************************************************************
A helper function to visualize the maps (label or classified)
Author : Eli Ionescu
Date : 27.05.2020
Description: Using this function to assign colors to maps (label and classified)
*************************************************************************/
Mat_<Vec3f> Utils::visualiseLabels(Mat &image, string& imageName)
{
	map<string, Vec3f> colors = loadLabelsMetadata();

	Mat result = Mat(image.rows, image.cols, CV_32FC3, Scalar(255.0f, 255.0f, 255.0f));
	// Create the output result;
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			Vec3f color;
			// Take every point and assign the right color in the result Mat
			int val = image.at<float>(row, col);
			switch (val) {
			case 0:
				color = colors["unclassified"];
				break;
			case 1:
				color = colors["city"];
				break;
			case 2:
				color = colors["field"];
				break;
			case 3:
				color = colors["forest"];
				break;
			case 4:
				color = colors["grassland"];
				break;
			case 5:
				color = colors["street"];
				break;
			default:
				cout << "Wrong value" << endl;
				break;
			}
			result.at<Vec3f>(row, col) = color;
		}
	}
	imwrite(imageName, result);

	return result;
}

/***********************************************************************
A helper function to visualize the maps (label or classified)
Author : Anupama Rajkumar
Date : 01.06.2020
Description: Using this function to create visualizations by reading the stored
csv files
*************************************************************************/

void Utils::Visualization(string& fileName, string& imageName, Size size) {

	cv::Mat img;
	//reading data from csv
	cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(fileName, 0, -1, -1);
	cv::Mat data = raw_data->getSamples();
	// optional if you have a color image and not just raw data
	data.convertTo(img, CV_32FC1);
	// set the image size
	cv::resize(img, img, size);
	//visualize the map
	visualiseLabels(img, imageName);
	cv::waitKey(0);
}

/***********************************************************************
Helper function to get the patch start and end index from label map
Author : Anupama Rajkumar
Date : 29.05.2020
Description: This function provides the start and end positions of the patch
from which the nearest neighbour needs to be found in the labelMap
*************************************************************************/

void Utils::GetLabelPatchIndex(int sizeOfPatch, Point2i samplePoints, Mat& LabelMap, int& pStart_r, int& pStart_c, int& pEnd_r, int& pEnd_c) {
	/*Ensure that the patch size is even*/
	if ((sizeOfPatch % 2) != 0) {
		cout << "Please ensure that the patch size is even. Changing patch dimension to next lower even number" << endl;
		sizeOfPatch = -1;
	}
	int rowStart, rowEnd, colStart, colEnd;
	pStart_r = samplePoints.x - (sizeOfPatch / 2.);
	pStart_c = samplePoints.y - (sizeOfPatch / 2.);

	pEnd_r = samplePoints.x + (sizeOfPatch / 2.);
	pEnd_c = samplePoints.y + (sizeOfPatch / 2.);

	if ((pStart_r < 0) || (pStart_c < 0))
	{
		pStart_r = 0;
		pStart_c = 0;
		pEnd_r = sizeOfPatch;
		pEnd_c = sizeOfPatch;
	}
	if ((pEnd_r > LabelMap.rows) || (pEnd_c > LabelMap.cols))
	{
		pEnd_r = LabelMap.rows - 1;
		pEnd_c = LabelMap.cols - 1;
		pStart_r = LabelMap.rows - 1 - sizeOfPatch;
		pStart_c = LabelMap.cols - 1 - sizeOfPatch;
	}
}


/*===================================================================
 * Function: getSafeSamplePoints
 * Author: Jun Xiang
 *
 * Summary:
 *   Extract sample points from mask area
 *
 * Arguments:
 *   Mat& mask  -- binarized image mask, zeros are background
 *   const int& samplePointNum -- maximum number of sample points for mask area
 *   const int& sampleSize  -- patch size at the sample point
 *	 vector<Point>& pts  --- to record the index of the sample points
 *
 * Returns:
 *   void
=====================================================================
*/
void Utils::getSafeSamplePoints(const Mat& img, const int& samplePointNum, const int& sampleSize, vector<Point>& pts) {
	
	// to draw samples from mask area
	if (img.channels() == 1) {
		Mat mask = img;
		vector<Point> ind;
		cv::findNonZero(img, ind);
		int nonZeros = static_cast<int>(ind.size());

		if (nonZeros > 0) {
			std::random_device random_device;
			std::mt19937 engine{ random_device() };
			std::uniform_int_distribution<int> dist(0, nonZeros - 1);

			int count = 0; // to record how many right sample points are found
			int iter = 0; // to record how many random points are tried out

			int N = nonZeros;
			if (nonZeros > samplePointNum) { N = samplePointNum; }

			std::set<pair<int, int>> new_ind;

			while (count < N) {
				Point  p = ind[dist(engine)];
				//check if the sample corners are on the border
				int x_min = p.x - int(sampleSize / 2); // (x,y) -> (col,row)
				int x_max = p.x + int(sampleSize / 2);
				int y_min = p.y - int(sampleSize / 2);
				int y_max = p.y + int(sampleSize / 2);
				// get rid of the points on the borders
				if (x_max < mask.cols && y_max < mask.rows && y_min >= 0 && x_min >= 0) {
					// get rid of points which are half patch size away from the mask zero area
					// (row,col) ->(y,x)
					if (mask.at<unsigned char>(y_min, x_min) != unsigned char(0) &&
						mask.at<unsigned char>(y_min, x_max) != unsigned char(0) &&
						mask.at<unsigned char>(y_max, x_min) != unsigned char(0) &&
						mask.at<unsigned char>(y_max, x_max) != unsigned char(0)) {
						//pts.push_back(p);
						new_ind.insert(pair<int, int>(p.x, p.y));
						count = count + 1;
					}
				}
				iter = iter + 1;
				if (iter > nonZeros) { break; }
			}

			for (auto it = new_ind.begin(); it != new_ind.end(); ++it)
			{
				pts.push_back(Point(it->first, it->second));
			}
		}
	} // draw sample points from rgb img
	else if (img.channels() == 3) {

		int totalsize = img.rows * img.cols;
		std::random_device random_device;
		std::mt19937 engine{ random_device() };
		std::uniform_int_distribution<int> distX(0, img.cols-1);
		std::uniform_int_distribution<int> distY(0, img.rows-1);

		int count = 0; // to record how many right sample points are found
		int iter = 0; // to record how many random points are tried out

		int N = totalsize;
		if (totalsize > samplePointNum) { N = samplePointNum; }

		std::set<pair<int, int>> new_ind;

		while (count < N) {
			Point  p = Point(distX(engine), distY(engine));
			//check if the sample corners are on the border
			int x_min = p.x - int(sampleSize / 2); // (x,y) -> (col,row)
			int x_max = p.x + int(sampleSize / 2);
			int y_min = p.y - int(sampleSize / 2);
			int y_max = p.y + int(sampleSize / 2);
			// get rid of the points on the borders
			if (y_max < img.rows && x_max < img.cols && y_min >= 0 && x_min >= 0) {
				 
					new_ind.insert(pair<int, int>(p.x, p.y));
					count = count + 1;
				
			}
			iter = iter + 1;
			if (iter > totalsize) { break; }
		}

		for (auto it = new_ind.begin(); it != new_ind.end(); ++it)
		{
			pts.push_back(Point(it->first, it->second));
		}
	}
}

// convert img data to png file
Mat Utils::writeImgToPNG(const Mat& src) {
	Mat dst;
	dst = src.clone();

	if (dst.channels() == 2) {
		dst = polsar::logTransform(dst);
	}
	else if (dst.channels() == 3) {
		cvtColor(dst, dst, COLOR_BGR2GRAY);
	}
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);
	cv::equalizeHist(dst, dst);
	return dst;
}

/***********************************************************************
Helper function to print the name of the identified class
Author : Anupama Rajkumar
Date : 29.05.2020
Description: This function prints the  name of the class identified
*************************************************************************/
void Utils::DisplayClassName(int finalClass) {
	switch (finalClass) {
	case 1:
		cout << "Point classified as City" << endl;
		break;
	case 2:
		cout << "Point classified as Field" << endl;
		break;
	case 3:
		cout << "Point classified as Forest" << endl;
		break;
	case 4:
		cout << "Point classified as Grassland" << endl;
		break;
	case 5:
		cout << "Point classified as Street" << endl;
		break;
	default:
		cout << finalClass << endl;
		cout << "Something went wrong..can't be classified" << endl;
		break;
	}

}





