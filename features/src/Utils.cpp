#include "Utils.h"



/**********************************
Generating a label map
Author : Anupama Rajkumar
Date : 27.05.2020
Modified by: Jun Xiang 22,06,2020
Description : Idea is to create a single label map from a list of various
label classes.This map serves as points of reference when trying to classify
patches
* ************************************************************************/

cv::Mat Utils::generateLabelMap(const std::vector<cv::Mat> & masks) {
	size_t NUMOFCLASSES = masks.size();
	int rows = masks[0].rows;
	int cols = masks[0].cols;
	cv::Mat labelMap = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (size_t cnt = 0; cnt < NUMOFCLASSES; cnt++) {
		cv::Mat mask = masks[cnt];
		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				if (labelMap.at<unsigned char>(row, col) == (unsigned char)(0)) {
					if (mask.at<unsigned char>(row, col) > (unsigned char)(0)) {
						labelMap.at<unsigned char>(row, col) = static_cast<unsigned char>(cnt + 1);		    //class of label
					}
				}
			}
		}
	}
	return labelMap;
}



/***********************************************************************
input: the ground_truth label and the label from classifiers
return: the color
*************************************************************************/
cv::Vec3b Utils::getLabelColor(unsigned char class_result)
{
	cv::Vec3b labelColor;

	// Color is BGR not RGB!
	cv::Vec3b black = cv::Vec3b(0, 0, 0);// unclassified, class 0

	cv::Vec3b red = cv::Vec3b(49, 60, 224); //city, class 1

	cv::Vec3b blue = cv::Vec3b(164, 85, 50); //street, class 2

	cv::Vec3b yellow = cv::Vec3b(0, 190, 246); //field, class 3

	cv::Vec3b dark_green = cv::Vec3b(66, 121, 79); //forest, class 4

	cv::Vec3b light_green = cv::Vec3b(0, 189, 181); // grassland, class 5


	std::vector<cv::Vec3b> right_color = { black, red,  yellow, dark_green, light_green,blue,};
	
	
	labelColor = right_color[int(class_result)];
	 
	return labelColor;
}




void Utils::generateColorMap(const std::string& hdf5_fileName, const std::string& feature_name, const std::string & classifier_type, int filterSize, int patchSize,int batchSize) {
	std::vector<unsigned char> labels;
	std::cout << std::endl;
	std::cout << "start to generate the color map of classified results"<<std::endl;
	std::string dataset = "/"+classifier_type;
	int totalrows = getRowSize(hdf5_fileName, feature_name, dataset, filterSize, patchSize);

	std::string parent = feature_name;
	if (patchSize != 0) { dataset = dataset + "_patchSize_" + std::to_string(patchSize); }
	if (filterSize != 0) { parent = feature_name + "_filterSize_" + std::to_string(filterSize); }

	cv::Mat labelMap;
	Utils::readDataFromHDF(hdf5_fileName, "/masks", "/labelMap", labelMap);
	if(totalrows>0){
		std::vector<unsigned char> class_results;
		std::vector<unsigned char> ground_truth_labels;
		cv::Mat colorResultMap = cv::Mat::zeros(cv::Size(labelMap.size()), CV_8UC3);
		cv::Mat groundTruthMap = cv::Mat::zeros(cv::Size(labelMap.size()), CV_8UC3);

		int offset_row = 0;
		int partSize;
		if (batchSize > totalrows) { batchSize = totalrows; }
		int partsCount = totalrows / batchSize;
		for (int i = 0; i < partsCount; ++i) {
			partSize = totalrows / (partsCount - i);
			totalrows -= partSize;
			if (totalrows < 0) { break; }
			cv::Mat pts;
			Utils::readDataFromHDF(hdf5_fileName, feature_name, dataset, pts, offset_row, partSize);

			for (int j = 0; j < pts.rows; j++) {
				int row = pts.at<int>(j, 1);
				int col = pts.at<int>(j, 2);
				unsigned char label = unsigned char(pts.at<int>(j, 0));
				unsigned char ground_truth = labelMap.at<unsigned char>(row, col);

				colorResultMap.at<cv::Vec3b>(row, col) = getLabelColor(label);
				groundTruthMap.at<cv::Vec3b>(row, col) = getLabelColor(ground_truth);

				class_results.push_back(label);
				ground_truth_labels.push_back(ground_truth);
			}
			offset_row = offset_row + partSize;
		}
		
		std::cout << "generate " << feature_name.substr(1) + "_colormap.png" << std::endl;
		cv::imwrite(feature_name.substr(1) + "_colormap.png", colorResultMap);
		cv::imwrite("groundTruthMap.png", groundTruthMap);
		featureProcess::calculatePredictionAccuracy(feature_name.substr(1), class_results, ground_truth_labels);
	}
	else {
		std::cout << "can't find " << parent + dataset << " in " << hdf5_fileName << std::endl;
	}
}



// classifier_type: choose from {"KNN", "opencvKNN","opencvRF","opencvFLANN"}
void Utils::classifyFeaturesML(const std::string& hdf5_fileName, const std::string& feature_name, const std::string classifier_type, int trainPercent, int filterSize, int patchSize,int batchSize) {
	std::vector<std::string> dataset_name = { "/feature" ,"/patchLabel" };

	if (Utils::checkExistInHDF(hdf5_fileName, feature_name, { "/" + classifier_type }, filterSize, patchSize)) {
		Utils::deleteDataFromHDF(hdf5_fileName, feature_name, { "/" + classifier_type }, filterSize, patchSize);
	}

	int fullSize = getRowSize(hdf5_fileName, feature_name, dataset_name[0], filterSize, patchSize);
	std::cout << "get " << fullSize << " rows for " << feature_name << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << std::endl;
	
	int offset_row = 0;
	int partSize;
	if (batchSize > fullSize) { batchSize = fullSize; }
	int partsCount = fullSize / batchSize;
	if (fullSize != 0) {
		for (int i = 0; i < partsCount; ++i) {
			partSize = fullSize / (partsCount - i);
			fullSize -= partSize;
			if (fullSize < 0) { break; }
			std::vector<cv::Mat> features;
			std::vector<cv::Point> labelPoints;
			std::vector<unsigned char> labels;
			Utils::getFeaturesFromHDF(hdf5_fileName, feature_name, dataset_name, features, labels, labelPoints, filterSize, patchSize, offset_row, partSize);
			std::cout << "get " << features.size() << " rows for " << feature_name << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << std::endl;

			std::vector<unsigned char> class_results;
			featureProcess::applyML(features, labels, 80, classifier_type, class_results);
			featureProcess::calculatePredictionAccuracy("", class_results, labels);
			saveClassResultToHDF(hdf5_fileName, feature_name, "/" + classifier_type, class_results, labelPoints, filterSize, patchSize);
			offset_row = offset_row + partSize;
			std::cout << "classifiy " << feature_name.substr(1) << " progress: " << float(i + 1) / float(partsCount) * 100.0 << "% \n" << std::endl;
		}
	}
	else {
		std::cout << feature_name << " with filterSize " << filterSize << " , patchSize " << patchSize << " is not existed in hdf5 file " << std::endl;
	}
}

// save the classify result to hdf5
void Utils::saveClassResultToHDF(const std::string& hdf5_fileName, const std::string& parent_name, const std::string& dataset_name, const std::vector<unsigned char>& class_result, const std::vector<cv::Point> & points,int filterSize,int patchSize) {
	cv::Mat pts = cv::Mat(points.size(), 3, CV_32SC1);
	for (size_t i = 0; i < points.size(); ++i) {
		pts.at<int>(i, 0) = (int)(class_result[i]);
		pts.at<int>(i, 1) = points[i].y; //row
		pts.at<int>(i, 2) = points[i].x; //col
	}
	std::string dataset = dataset_name;
	std::string parent = parent_name;
	if (patchSize != 0) { dataset = dataset_name + "_patchSize_" + std::to_string(patchSize); }
	if (filterSize != 0) { parent = parent_name + "_filterSize_" + std::to_string(filterSize); }
	Utils::insertDataToHDF(hdf5_fileName, parent_name, dataset, pts);
}


// get features data from hdf5
// features and featureLabels for train and test
// labelPoints for the location in image
void Utils::getFeaturesFromHDF(const std::string& hdf5_fileName, const std::string& parent_name, std::vector<std::string>& dataset_name,
	std::vector<cv::Mat>& features,std::vector<unsigned char>& featureLabels, std::vector<cv::Point> & labelPoints, int filterSize, int patchSize, int offset_row, int counts_rows) {
	
	std::vector<cv::Mat> data;
	if (Utils::checkExistInHDF(hdf5_fileName, parent_name, dataset_name, filterSize, patchSize)) {
		
		Utils::readDataFromHDF(hdf5_fileName, parent_name, dataset_name, data, filterSize, patchSize,offset_row,counts_rows);
		cv::Mat feature = data[0];
		cv::Mat pts = data[1]; //labelPoint
		for (int i = 0; i < feature.rows; ++i) {
			features.push_back(feature.row(i));
			featureLabels.push_back((unsigned char)(pts.at<int>(i, 0)));
			cv::Point p;
			p.y = pts.at<int>(i, 1); //row
			p.x = pts.at<int>(i, 2); //col
			labelPoints.push_back(p);
		}
	}
}



void Utils::deleteDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, int filterSize, int patchSize) {
	std::string parent = parent_name;

	if (filterSize != 0) { parent = parent + "_filterSize_" + std::to_string(filterSize); }
	for (int i = 0; i < dataset_name.size(); ++i) {
		if (patchSize != 0) {
			deleteDataFromHDF(filename, parent, dataset_name[i] + "_patchSize_" + std::to_string(patchSize));
		}
		else {
			deleteDataFromHDF(filename, parent, dataset_name[i]);
		}
	}
}


void Utils::deleteDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	std::string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		std::cout << parent_name << " is not existed." << std::endl;
	}else {
		if (!h5io->hlexists(datasetName)) {
			std::cout << datasetName << " is not existed." << std::endl;
		}else {
	        int result = h5io->dsdelete(datasetName);
			if (!result) {
				std::cout << "delete dataset " << datasetName << " success." << std::endl;
			}
			else {
				std::cout << "Failed to delete " << datasetName << std::endl;
			}
		}
	}
}



void Utils::writeDataToHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, const std::vector<cv::Mat>& data, int filterSize , int patchSize) {
	std::string parent = parent_name;
	if (filterSize != 0) { parent = parent + "_filterSize_" + std::to_string(filterSize); }

	if (data.size() == dataset_name.size()) {
		for (int i = 0; i < data.size(); ++i) {
			if(patchSize !=0){
			writeDataToHDF(filename, parent, dataset_name[i]+ "_patchSize_"+ std::to_string(patchSize), data[i]);
			}
			else {
				writeDataToHDF(filename, parent, dataset_name[i] , data[i]);
			}
		}
	}
	else {
		std::cout << "the size of dataset_name doesn't match that of data" << std::endl;
	}
}

void Utils::writeDataToHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& src) {
	if(!src.empty()){
		cv::Mat data = src.clone();
		if (data.channels() > 1) {
			for (size_t i = 0; i < data.total() * data.channels(); ++i)
				((int*)data.data)[i] = (int)i;
		}

		cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

		// first we need to create the parent group
		if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

		// create the dataset if it not exists
		std::string datasetName = parent_name + dataset_name;
		if (!h5io->hlexists(datasetName)) {
			h5io->dscreate(data.rows, data.cols, data.type(), datasetName);
			h5io->dswrite(data, datasetName);

			// check if the data are correctly write to hdf file
			cv::Mat expected = cv::Mat(cv::Size(data.size()), data.type());
			h5io->dsread(expected, datasetName);
			float diff = norm(data - expected);
			CV_Assert(abs(diff) < 1e-10);

			if (h5io->hlexists(datasetName))
			{
				//std::cout << "write " << datasetName << " to " << filename << " success." << std::endl;
			}
			else {
				std::cout << "Failed to write " << datasetName << " to " << filename << std::endl;
			}
		}
		else {
			std::cout << datasetName << " is already existed." << std::endl;
		}
		h5io->close();
	}
}

void Utils::readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, std::vector<cv::Mat>& data, int filterSize, int patchSize, int offset_row, int counts_rows) {
	std::string parent = parent_name;
	if (!data.empty()) { data.clear(); }

	if (filterSize != 0) { parent = parent + "_filterSize_" + std::to_string(filterSize); }
		for (int i = 0; i < dataset_name.size(); ++i) {
			cv::Mat temp;
			if (patchSize != 0) {
				readDataFromHDF(filename, parent, dataset_name[i] + "_patchSize_" + std::to_string(patchSize), temp, offset_row, counts_rows);
			}
			else {
				readDataFromHDF(filename, parent, dataset_name[i], temp, offset_row, counts_rows);
			}
			if (!temp.empty()) { data.push_back(temp); }
		}
}



void Utils::readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, cv::Mat& data, int offset_row, int counts_rows) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

	std::string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		//std::cout << parent_name << " is not existed" << std::endl;
		data = cv::Mat();
	}
	else if (!h5io->hlexists(datasetName)) {
		//std::cout << datasetName << " is not existed" << std::endl;  
		data = cv::Mat();
	}
	else {

		std::vector<int> data_size = h5io->dsgetsize(datasetName);

		if(counts_rows !=0 && counts_rows <= data_size[0]){
			data = cv::Mat(counts_rows, data_size[1], h5io->dsgettype(datasetName));
			std::vector<int> dims_offset(2), dims_count(2);
			dims_offset = { offset_row,0 };
			dims_count = { counts_rows, data.cols };
			h5io->dsread(data, datasetName, dims_offset, dims_count);
		}
		else if (counts_rows == 0) {
			data = cv::Mat(data_size[0], data_size[1], h5io->dsgettype(datasetName));

			h5io->dsread(data, datasetName);
		}
	}

	h5io->close();
}

void readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, std::vector<cv::Mat>& data, int filterSize, int patchSize, int start_row , int total_rows) {

}

int Utils::getRowSize(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, int filterSize, int patchSize) {
	int rows;

	std::string parent = parent_name;
	std::string dataset = dataset_name;
	if (filterSize != 0) { parent = parent + "_filterSize_" + std::to_string(filterSize); }
	if (patchSize != 0) { dataset = dataset + "_patchSize_" + std::to_string(patchSize);}
	
	std::vector<int> data_size;
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

	std::string datasetName = parent_name + dataset;

	if (!h5io->hlexists(parent_name)) {
		//std::cout << parent_name << " is not existed" << std::endl;
		rows =0;
	}
	else if (!h5io->hlexists(datasetName)) {
		//std::cout << datasetName << " is not existed" << std::endl;  
		rows = 0;
	}
	else {

		 data_size = h5io->dsgetsize(datasetName);
		 rows = data_size[0];
	}

	h5io->close();
	return rows;
}

void Utils::writeAttrToHDF(const std::string& filename,const std::string& attribute_name,const int &attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		std::cout << " already existed" << std::endl;
	}
	h5io->close();
}

void Utils::writeAttrToHDF(const std::string& filename, const std::string& attribute_name, const std::string &attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		std::cout << " already existed" << std::endl;
	}
	h5io->close();
}

void Utils::readAttrFromHDF(const std::string& filename, const std::string& attribute_name,  std::string &attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		std::cout << attribute_name<<" is not existed" << std::endl;
	}
	else {
		h5io->atread(&attribute_value, attribute_name);
	}
	h5io->close();
}

void Utils::readAttrFromHDF(const std::string& filename, const std::string& attribute_name, int& attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		std::cout << attribute_name << " is not existed" << std::endl;
	}
	else {
		h5io->atread(&attribute_value, attribute_name);
	}
	h5io->close();
}



bool Utils::checkExistInHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	bool flag = true;
	
	if (!h5io->hlexists(parent_name)) {
		flag = false;
		//std::cout << parent_name << " is not existed" << std::endl;
	}else if (!h5io->hlexists(parent_name + dataset_name)) {
		flag = false;
		//std::cout << parent_name + dataset_name << " is not existed" << std::endl;
	}
	h5io->close();
	return flag;
}


bool Utils::checkExistInHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, int filterSize, int patchSize ) {
	bool flag = true;
	std::string parent = parent_name;
	std::vector<std::string> dataset = dataset_name;
	if (filterSize != 0) {
		parent = parent_name + "_filterSize_" + std::to_string(filterSize);
	}

	for (auto& n : dataset) {
		if(patchSize !=0){
		   n =  n + "_patchSize_" + std::to_string(patchSize);
		}
		bool temp = checkExistInHDF(filename, parent, n);
		flag = flag && temp;
	}
	return	flag;
}


bool Utils::insertDataToHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data) {
	bool flag = true;
	if(!data.empty()){
		cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

		if (checkExistInHDF(filename, parent_name, dataset_name)) {
			std::string dataset = parent_name + dataset_name;
			std::vector<int> data_size = h5io->dsgetsize(dataset);
			// expand the dataset at row direction
			int offset[2] = { data_size[0],0 };

			if ((h5io->dsgettype(dataset) == data.type()) && (data_size[1] == data.cols)) {
				h5io->dsinsert(data, dataset, offset);

				//check if insert success
				//std::cout << std::endl;
				//std::cout << "insert " << data.rows << " rows to " << dataset << " success " << std::endl;
				//std::cout << dataset << " rows in total: " << data.rows + offset[0] << std::endl;
			}

			else {
				flag = false;
				std::cout << std::endl;
				std::cout << " the new data has different size and type with the existed data" << std::endl;
				std::cout << dataset << " insert failed" << std::endl;
			}
		}
		else {
			// first we need to create the parent group
			if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

			std::string dataset = parent_name + dataset_name;
			int chunks[2] = { 1, data.cols };
			// create Unlimited x data.cols, data.type() space, dataset can be expanded unlimted on the row direction
			h5io->dscreate(hdf::HDF5::H5_UNLIMITED, data.cols, data.type(), dataset, hdf::HDF5::H5_NONE, chunks);
			// the first time to write data, offset at row,col direction is 0
			int offset[2] = { 0, 0 };
			h5io->dsinsert(data, dataset, offset);
			std::cout << std::endl;
			//std::cout << "insert " << data.rows << " rows to" << dataset << " success " << std::endl;
			//std::cout << dataset << " rows in total: " << data.rows + offset[0] << std::endl;
		}
	}
	return flag;
}

bool Utils::insertDataToHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, const std::vector<cv::Mat>& data, int filterSize, int patchSize) {
	bool flag = true;
	std::string parent = parent_name;
	if (filterSize != 0) { parent = parent + "_filterSize_" + std::to_string(filterSize); }

	if (data.size() == dataset_name.size()) {
		for (int i = 0; i < data.size(); ++i) {
			bool temp;
			if (patchSize != 0) {
				temp =insertDataToHDF(filename, parent, dataset_name[i] + "_patchSize_" + std::to_string(patchSize), data[i]);
			}
			else {
				temp =insertDataToHDF(filename, parent, dataset_name[i], data[i]);
			}
			flag = flag && temp;
		}
	}
	else {
		flag = false;
		std::cout << "the size of dataset_name doesn't match that of data"<<std::endl;
	}
	return flag;
}


std::vector<cv::Point> Utils::generateSamplePoints(const cv::Mat& labelMap, const int& patchSize, const int& stride) {

	std::vector<cv::Point> samplePoints;
	for (int row = 0; row < labelMap.rows - patchSize; row += stride) {
		for (int col = 0; col < labelMap.cols - patchSize; col += stride) {
			cv::Rect cell = cv::Rect(col, row, patchSize, patchSize);

			int halfsize = patchSize / 2;
			
			//record the central points of each patch
			samplePoints.push_back(cv::Point(col + halfsize, row + halfsize));
		}
	}
	return samplePoints;
}

void Utils::getRandomSamplePoint(const cv::Mat& labelMap, std::vector<cv::Point> & samplePoints, const unsigned char &sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass) {
	 
	std::vector<cv::Point> temp = generateSamplePoints(labelMap, sampleSize, stride);
	std::map<unsigned char, std::vector<cv::Point> > count;
	for (auto& p : temp) {
		unsigned char label = labelMap.at<unsigned char>(p.y, p.x);
		if (label == sampleLabel) {
			count[sampleLabel].push_back(p);
		}
	}
	std::vector<cv::Point> pts = count[sampleLabel];

	if (numOfSamplePointPerClass > 0) {
		std::random_device random_device;
		std::mt19937 engine{ random_device() };
		std::uniform_int_distribution<int> pt(0, pts.size() - 1);
		size_t num = 0;
		size_t iter = 0;

		while (num < numOfSamplePointPerClass) {
			cv::Point p = pts[pt(engine)];

			// get samples in homogeneous areas 
			// this is only for checking the accuracy of features
			unsigned char label = labelMap.at<unsigned char>(p.y, p.x);
			unsigned char sample_upcorner = labelMap.at<unsigned char>(p.y - sampleSize / 2, p.x - sampleSize / 2);
			unsigned char sample_downcorner = labelMap.at<unsigned char>(p.y + sampleSize / 2, p.x + sampleSize / 2);
			unsigned char sample_leftcorner = labelMap.at<unsigned char>(p.y + sampleSize / 2, p.x - sampleSize / 2);
			unsigned char sample_rightcorner = labelMap.at<unsigned char>(p.y - sampleSize / 2, p.x + sampleSize / 2);
			if ((label == sample_upcorner) && (label == sample_downcorner) &&
				(label == sample_leftcorner) && (label == sample_rightcorner)) {
				samplePoints.push_back(p);
				++num;
			}
			 ++iter;
			if (iter > pts.size()) { break; }
		}
	 }
	 else {
		 std::cout << "load all the sample points" << std::endl;
		 copy(pts.begin(), pts.end(), back_inserter(samplePoints));
	 }
}






void Utils::featureDimReduction(const std::string& hdf5_fileName, const std::string& feature_name, int numSamples, int filterSize, int patchSize) {
	std::vector<std::string> dataset_name = { "/feature" ,"/patchLabel" };

	int totalrows = getRowSize(hdf5_fileName, feature_name, dataset_name[0], filterSize, patchSize);
	if (numSamples > totalrows) { numSamples = totalrows; }

	std::random_device random_device;
	std::mt19937 engine{ random_device() };
	std::uniform_int_distribution<int> rows(0, totalrows - 1);

	std::vector<cv::Mat> feature_temp(numSamples), patchLabels_temp(numSamples);
	for(int i =0; i< numSamples; i++){
		int offset_row = rows(engine);
		int counts_row = 1;
		std::vector<cv::Mat> data;
		Utils::readDataFromHDF(hdf5_fileName, feature_name, dataset_name, data, filterSize, patchSize, offset_row, counts_row);
		feature_temp[i] = data[0];
		patchLabels_temp[i] = data[1];
	}

	cv::Mat feature, patchLabels;
	cv::vconcat(feature_temp, feature);
	cv::vconcat(patchLabels_temp, patchLabels);

	int new_dims = 2;
	cv::Mat reduced_feature = featureProcess::featureDimReduction(feature, new_dims);

	//save dimension reduced features and labels to txt file
	std::cout << "save dimension reduced "<< feature_name.substr(1)<<" feature to txt" << std::endl;
	std::string dim_reduce =  feature_name.substr(1) + "_dimReduced" + ".txt";
	std::ofstream fout(dim_reduce);
	for(int i =0; i< reduced_feature.rows; i++){
	   fout << patchLabels.at<int>(i, 0) << "," << reduced_feature.at<float>(i, 0) << "," << reduced_feature.at<float>(i, 1) << std::endl;
	}
	
	//check the KNN accuracy of dim reduced features
	std::vector<cv::Mat> newfeatures(reduced_feature.rows);
	std::vector<unsigned char> labels(reduced_feature.rows);
	for (int i = 0; i < reduced_feature.rows; i++) {
		labels[i] = unsigned char(patchLabels.at<int>(i, 0));
		cv::Mat temp(1, 2, CV_32FC1);
		temp.at<float>(0, 0) = reduced_feature.at<float>(i, 0);
		temp.at<float>(0, 1) = reduced_feature.at<float>(i, 1);
		newfeatures[i] = temp;
	}
	std::vector<unsigned char> results;
	featureProcess::applyML(newfeatures, labels, 80, "opencvFLANN", results);
	featureProcess::calculatePredictionAccuracy("", results, labels);
}


void Utils::splitVec(const std::vector<unsigned char>& labels, std::vector<std::vector<int>>& subInd, int batchSize) {

	// To regulate count of parts
	if (batchSize > labels.size()) { batchSize = labels.size(); }
	int partsCount = labels.size()/ batchSize;

	if (subInd.size() == 0) { subInd = std::vector<std::vector<int>>(partsCount); }

	std::map<unsigned char, std::vector<int>> count;
	for (int ind = 0; ind < labels.size(); ind++) {
		count[labels[ind]].push_back(ind);
	}

	for (const auto& c : count) {
		std::vector<int> inds = c.second;
		// Variable to control size of non divided elements
		int fullSize = inds.size();
		int start = 0;
		for (int i = 0; i < partsCount; ++i) {
			int partSize = fullSize / (partsCount - i);
			fullSize -= partSize;
			for (int j = 0; j < partSize; j++) {
				subInd[i].push_back(inds[start + j]);
			}
			start = start + partSize;
		}
	}

	//shuffle the index
	// obtain a time-based seed
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine e(seed);
	for (auto i = 0; i < subInd.size(); i++) {
		std::shuffle(subInd[i].begin(), subInd[i].end(), e);
	}
}