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
	cv::Vec3b red = cv::Vec3b(49, 60, 224); //city

	cv::Vec3b blue = cv::Vec3b(164, 85, 50); //street

	cv::Vec3b yellow = cv::Vec3b(0, 190, 246); //field

	cv::Vec3b dark_green = cv::Vec3b(66, 121, 79); //forest

	cv::Vec3b light_green = cv::Vec3b(0, 189, 181); // grassland

	cv::Vec3b black = cv::Vec3b(0, 0, 0);

	std::vector<cv::Vec3b> right_color = { red,  yellow, dark_green, light_green,blue, black };
	
	
	labelColor = right_color[int(class_result)-1];
	 
	return labelColor;
}



/************************************************************
Dividing the data samples into training and test samples
eg: make sure each class is divided 80% as train, 20% as test
int fold: the cross validation fold number, an integer between {1, 100 / (100 - percentOfTrain)}
Modified by: Jun 15.06.2020
return: the index of test data in the original data 
*************************************************************/
std::vector<int> Utils::DivideTrainTestData(const std::vector<cv::Mat> &data, const std::vector<unsigned char> & data_label, int percentOfTrain,
	std::vector<cv::Mat> & train_img,  std::vector<unsigned char> &train_label, std::vector<cv::Mat>& test_img, std::vector<unsigned char> & test_label,int fold) {
	
	std::map<unsigned char, std::vector<int>> numPerClass;
	int index = 0;
	for (auto c : data_label) { 
		numPerClass[c].push_back(index); 
		index++;
	}
	std::vector<int> train_index, test_index;
	int total_folds = 100 / (100 - percentOfTrain);

	// make sure each class is divided 80% as train, 20% as test
	for (auto it = numPerClass.begin(); it != numPerClass.end(); it++)
	{
		size_t train_size = it->second.size() * percentOfTrain / 100;
		size_t test_size = it->second.size() - train_size;

		std::vector<int> indOfClass;
		// expand indOfClass twice
		copy(it->second.begin(), it->second.end(), back_inserter(indOfClass));
		copy(it->second.begin(),it->second.end(), back_inserter(indOfClass));

		std::vector<int> train_temp, test_temp;
		int train_temp_size = 0;
		int test_temp_size = 0;
		for (size_t i = 0; i < indOfClass.size(); ++i) {
			if (train_temp_size < test_size) {
				test_temp.push_back(indOfClass[i+ (fold-1)*test_size]);
				train_temp_size++;
			}
			if (test_temp_size < train_size) {
				train_temp.push_back(indOfClass[i + fold * test_size]);
				test_temp_size++;
			}
		}

		copy(train_temp.begin(), train_temp.end(), back_inserter(train_index));
		copy(test_temp.begin(), test_temp.end(), back_inserter(test_index));

	}
	for (auto i : train_index) {
		train_img.push_back(data[i]);
		train_label.push_back(data_label[i]);
	}
	for (auto i : test_index) {
		test_img.push_back(data[i]);
		test_label.push_back(data_label[i]);
	}

	return test_index;
}

void Utils::generateColorMap(const std::string& hdf5_fileName, const std::string& feature_name, const std::string & classifier_type, int filterSize, int patchSize) {
	std::vector<unsigned char> labels;
	cv::Mat pts;
	cv::Mat labelMap;

	std::string dataset = "/"+classifier_type;
	std::string parent = feature_name;
	if (patchSize != 0) { dataset = dataset + "_patchSize_" + std::to_string(patchSize); }
	if (filterSize != 0) { parent = feature_name + "_filterSize_" + std::to_string(filterSize); }

	Utils::readDataFromHDF(hdf5_fileName, feature_name, dataset,pts);
	Utils::readDataFromHDF(hdf5_fileName, "/masks", "/labelMap", labelMap);
	if(!pts.empty()){
		cv::Mat colorResultMap = cv::Mat::zeros(cv::Size(labelMap.size()), CV_8UC3);
		cv::Mat colorLabelMap = cv::Mat::zeros(cv::Size(labelMap.size()), CV_8UC3);
		for (int i = 0; i < pts.rows; ++i) {
			int row = pts.at<int>(i, 1);
			int col = pts.at<int>(i, 2);
			unsigned char label = unsigned char(pts.at<int>(i, 0));
			 unsigned char ground_truth = labelMap.at<unsigned char>(row, col);
			 colorResultMap.at<cv::Vec3b>(row, col) = getLabelColor( label);
			colorLabelMap.at<cv::Vec3b>(row, col) = getLabelColor(ground_truth);
		}
		std::cout << "generate " << feature_name.substr(1) + "_colormap.png" << std::endl;
		cv::imwrite(feature_name.substr(1) + "_colormap.png", colorResultMap);
		cv::imwrite("colorLabelMap.png", colorLabelMap);
	}
	else {
		std::cout << "can't find " << parent + dataset << " in " << hdf5_fileName << std::endl;
	}
}

void Utils::classifyFeaturesKNN(const std::string& hdf5_fileName,  const std::string& feature_name, int k, int trainPercent,int filterSize, int patchSize) {
	KNN* knn = new KNN();
	std::vector<std::string> dataset_name = { "/feature_dimReduced" ,"/patchLabel" };
	std::vector<cv::Mat> features;
	std::vector<cv::Point> labelPoints;
	std::vector<unsigned char> labels;
	std::vector<unsigned char> class_results;
	Utils::getFeaturesFromHDF(hdf5_fileName, feature_name, dataset_name,features, labels, labelPoints, filterSize, patchSize);
	if (!features.empty()) {
		std::cout << "get " << features.size() << " rows for " << feature_name << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << std::endl;
		knn->applyKNN(features, labels, k, 80, class_results);

		if (Utils::checkExistInHDF(hdf5_fileName, feature_name, { "/knn" }, filterSize, patchSize)) {
			Utils::deleteDataFromHDF(hdf5_fileName, feature_name, { "/knn" }, filterSize, patchSize);
		}

		Utils::saveClassResultToHDF(hdf5_fileName, feature_name, "/knn", class_results, labelPoints, filterSize, patchSize);
		features.clear();
		labelPoints.clear();
		labels.clear();
		class_results.clear();
	}
	else {
		std::cout << feature_name << " with filterSize " << filterSize << " , patchSize " << patchSize << " is not existed in hdf5 file " << std::endl;
	}
	 
	delete knn;
}

// classifier_type: choose from {"KNN", "RF"}
void Utils::classifyFeaturesML(const std::string& hdf5_fileName, const std::string& feature_name, const std::string classifier_type, int trainPercent, int filterSize, int patchSize) {

	std::vector<std::string> dataset_name = { "/feature" ,"/patchLabel" };
	std::cout << std::endl;
	
	std::map<unsigned char, int> accuracy;
	std::vector<cv::Mat> features;
	std::vector<cv::Point> labelPoints;
	std::vector<unsigned char> labels;
	Utils::getFeaturesFromHDF(hdf5_fileName, feature_name, dataset_name, features, labels, labelPoints, filterSize, patchSize);
	if (!features.empty()) {
		if (Utils::checkExistInHDF(hdf5_fileName, feature_name, { "/" + classifier_type }, filterSize, patchSize)) {
			Utils::deleteDataFromHDF(hdf5_fileName, feature_name, { "/" + classifier_type }, filterSize, patchSize);
		}
		std::cout << "get " << features.size() << " rows for " << feature_name << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << std::endl;

		int n = features.size() / 5000;
		if (n == 0) { n = 1; }
		std::vector<std::vector<cv::Mat>> subFeatures(n);
		std::vector<std::vector<unsigned char>> subLabels(n);
		std::vector<std::vector<cv::Point> > subLabelPoints(n);
		splitVec(features, labels, labelPoints, subFeatures, subLabels, subLabelPoints, n);

		
		for (int j = 0; j < n; j++) {
			std::vector<unsigned char> class_results;
			applyML(subFeatures[j], subLabels[j], 80, classifier_type, class_results);
			Utils::saveClassResultToHDF(hdf5_fileName, feature_name, "/" + classifier_type, class_results, subLabelPoints[j], filterSize, patchSize);
			
			calculatePredictionAccuracy(feature_name,class_results, subLabels[j]);
			class_results.clear();
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
	std::vector<cv::Mat>& features,std::vector<unsigned char>& featureLabels, std::vector<cv::Point> & labelPoints, int filterSize, int patchSize) {
	
	std::vector<cv::Mat> data;
	if (Utils::checkExistInHDF(hdf5_fileName, parent_name, dataset_name, filterSize, patchSize)) {
		Utils::readDataFromHDF(hdf5_fileName, parent_name, dataset_name, data, filterSize, patchSize);
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

// shuffle the data, and record the original index of the shuffled data
std::vector<int> Utils::shuffleDataSet(std::vector<cv::Mat>& data, std::vector<unsigned char>& data_label) {
	int size = data.size();
	std::vector<int> ind(size);
	std::random_device random_device;
	std::mt19937 engine{ random_device() };
	std::uniform_int_distribution<int> rnd(0, size - 1);
	for (int i = 0; i < size; ++i) {
		cv::Mat temp = data[i];
		signed char temp_c = data_label[i];
		int swap = rnd(engine);
		if (swap == i) { continue; }
		else {
			data[i] = data[swap];
			data[swap] = temp;
			data_label[i] = data_label[swap];
			data_label[swap] = temp_c;
		}
		ind[i] = swap;
		ind[swap] = i;
	}
	return ind;
}

float Utils::calculatePredictionAccuracy(const std::string & feature_name,const std::vector<unsigned char>& classResult, const std::vector<unsigned char>& testLabels)
{
	std::string overall_accuracy = "oa_"+ feature_name.substr(1)+".txt";
	std::ofstream fout(overall_accuracy);
	fout << feature_name.substr(1) << std::endl;
	float accuracy = 0.0;
	if (classResult.size() != testLabels.size()) {
		std::cerr << "Predicted and actual label vectors differ in length. Somethig doesn't seem right." << std::endl;
		exit(-1);
	}
	else {
		std::map<unsigned char, float> hit;
		std::map<unsigned char, float> total;

		int dim = classResult.size();

		for (int i = 0; i < dim; ++i) {
			if (classResult[i] == testLabels[i]) {
				hit[classResult[i]]++;
			}
			total[testLabels[i]]++;
		}

		float a = 0.0;
		for (auto& h : hit) {
			unsigned char label = h.first;
			float correct = h.second;
			float totalNum = total[label];
			float class_accuracy = correct / totalNum;
			fout<< "accuracy for class " << std::to_string(label) << ": " << class_accuracy << std::endl;
			std::cout << "accuracy for class " << std::to_string(label) << ": " << class_accuracy << std::endl;
			a = correct + a;
		}
		accuracy = a / testLabels.size();
		std::cout << "overall accuracy: " << accuracy << std::endl;
		fout << "oa: " << accuracy << std::endl;
	}
	return  accuracy;
}

cv::Mat Utils::getConfusionMatrix(const std::map<unsigned char, std::string>& className, std::vector<unsigned char>& classResult, std::vector<unsigned char>& testLabels) {
	std::map<std::pair<unsigned char, signed char>, int> testCount;

	for (int i = 0; i < testLabels.size(); ++i) {
		for (int j = 0; j < classResult.size(); ++j) {
			std::pair temp = std::make_pair(testLabels[i], classResult[j]);
			testCount[temp]++;
		}
	}

	int numOfClass = className.size();
	std::vector<unsigned char> classList(numOfClass);
	for (auto it = className.begin(); it != className.end(); it++) {
		classList.push_back(it->first);
	}

	cv::Mat count = cv::Mat(className.size(), className.size(), CV_8UC1);
	for (int i = 0; i < numOfClass; ++i) {
		for (int j = 0; j < numOfClass; ++j) {
			std::pair temp = std::make_pair(classList[i], classList[j]);
			count.at<unsigned char>(i, j) = testCount[temp];
		}
	}
	return count;
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

void Utils::readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::vector<std::string>& dataset_name, std::vector<cv::Mat>& data, int filterSize, int patchSize) {
	std::string parent = parent_name;
	if (!data.empty()) { data.clear(); }

	if (filterSize != 0) { parent = parent + "_filterSize_" + std::to_string(filterSize); }
		for (int i = 0; i < dataset_name.size(); ++i) {
			cv::Mat temp;
			if (patchSize != 0) {
				readDataFromHDF(filename, parent, dataset_name[i] + "_patchSize_" + std::to_string(patchSize), temp);
			}
			else {
				readDataFromHDF(filename, parent, dataset_name[i], temp);
			}
			if (!temp.empty()) { data.push_back(temp); }
		}
}

void Utils::readDataFromHDF(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, cv::Mat& data) {

	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

	std::string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		//std::cout << parent_name << " is not existed" << std::endl;
		data = cv::Mat();
	}
	else if (!h5io->hlexists(datasetName) ) { 
		//std::cout << datasetName << " is not existed" << std::endl;  
		data = cv::Mat(); 
	} else {
		std::vector<int> data_size = h5io->dsgetsize(datasetName);

		data = cv::Mat(data_size[0],data_size[1],h5io->dsgettype(datasetName));

	    h5io->dsread(data, datasetName);
		//std::cout << "get " <<  datasetName  << " success" << std::endl;
	}

	h5io->close();
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


void Utils::applyML(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_labels,int trainPercent, const std::string & classifier_type, std::vector<unsigned char>& results) {

	std::cout << "start to classify data with classifier :" << classifier_type << std::endl;

	// classify result
	 results = std::vector<unsigned char>(data_labels.size());

	//copy the original data
	std::vector<cv::Mat> temp(data.begin(), data.end());
	std::vector<unsigned char> temp_labels(data_labels.begin(), data_labels.end());

	std::vector<cv::Mat> train;
	std::vector<unsigned char> train_labels;
	std::vector<cv::Mat> test;
	std::vector<unsigned char> test_labels;

	int total_folds = 100 / (100 - trainPercent);
	for (int fold = 1; fold < total_folds + 1; ++fold) {
		std::vector<int> test_ind = Utils::DivideTrainTestData(temp, temp_labels, trainPercent, train, train_labels, test, test_labels, fold);
		std::vector<unsigned char> test_result;
		cv::Mat traindata, traindata_label, testdata;
		vconcat(train, traindata);
		vconcat(test, testdata);
		vconcat(train_labels, traindata_label);
		traindata_label.convertTo(traindata_label, CV_32SC1);
		traindata.convertTo(traindata, CV_32FC1);
		testdata.convertTo(testdata, CV_32FC1);

		if (classifier_type == "KNN") {
			cv::Ptr<cv::ml::TrainData> cv_data = cv::ml::TrainData::create(traindata, 0, traindata_label);
			cv::Ptr<cv::ml::KNearest>  knn(cv::ml::KNearest::create());
			knn->setDefaultK(20);
			knn->setIsClassifier(true);
			knn->train(cv_data);
			for (auto & x_test : test) {
				x_test.convertTo(x_test, CV_32FC1);
				auto knn_result = knn->predict(x_test);
				test_result.push_back(unsigned char(knn_result));
			}
		}
		else if (classifier_type == "FLANN") {
			int K = 20;
			cv::flann::Index flann_index(
				traindata,
				cv::flann::KDTreeIndexParams(4),
				cvflann::FLANN_DIST_EUCLIDEAN
			);
			cv::Mat indices(testdata.rows, K, CV_32S);
			cv::Mat dists(testdata.rows, K, CV_32F);
			flann_index.knnSearch(testdata, indices, dists,K, cv::flann::SearchParams(200));
			KNN* knn = new KNN();
			for (int i = 0; i < testdata.rows; i++) {
				std::vector<std::pair<float, unsigned char>> dist_vec(K);
				for (int j = 0; j < K; j++) {
					unsigned char temp = train_labels[indices.at<int>(i, j)];
					float distance = dists.at<float>(i, j);
					dist_vec[j] = std::make_pair(distance, temp);
				}
				// voting 
				test_result.push_back(knn->Classify(dist_vec,K));
				dist_vec.clear();
			}
			delete knn;
		}
		else if (classifier_type == "RF") {
			cv::Ptr<cv::ml::TrainData> cv_data = cv::ml::TrainData::create(traindata, cv::ml::ROW_SAMPLE, traindata_label);
			cv::Ptr<cv::ml::RTrees>  randomForest(cv::ml::RTrees::create());
			auto criterRamdomF = cv::TermCriteria();
			criterRamdomF.type = cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS;
			criterRamdomF.epsilon = 1e-8;
			criterRamdomF.maxCount = 500;
			randomForest->setTermCriteria(criterRamdomF);
			//randomForest->setMaxCategories(15);
			//randomForest->setMaxDepth(25);
			//randomForest->setMinSampleCount(1);
			//randomForest->setTruncatePrunedTree(false);
			//randomForest->setUse1SERule(false);
			//randomForest->setUseSurrogates(false);
			//randomForest->setPriors(cv::Mat());
			//randomForest->setCVFolds(1);
			 
			randomForest->train(cv_data);
			for (auto & x_test : test) {
				x_test.convertTo(x_test, CV_32FC1);
				test_result.push_back(unsigned char(randomForest->predict(x_test)));
			}
		}

		float count = 0.0;
		for (int i = 0; i < test.size(); ++i) {
			unsigned char y_test = test_labels[i];
			unsigned char y_result = test_result[i];
			if (y_test == y_result) {
				count = count + 1.0;
			}
			results[test_ind[i]] = y_result;
		}
		 
		train.clear();
		train_labels.clear();
		test.clear();
		test_labels.clear();
		test_result.clear();
	}
}

// put data in batches, make sure each class has balanced proportion in each batch
// n: number of batches
void Utils::splitVec(const std::vector<cv::Mat>& features, const std::vector<unsigned char>& labels, const std::vector<cv::Point> & labelPoints, std::vector<std::vector<cv::Mat>>& subFeatures,
	std::vector<std::vector<unsigned char>>& subLables, std::vector<std::vector<cv::Point> >& subLabelPoints, int n) {
	if ((features.size() == labels.size()) && (labels.size() == labelPoints.size())) {

		if (subFeatures.size() == 0) { subFeatures = std::vector<std::vector<cv::Mat>>(n); }
		if (subLables.size() == 0) { subLables = std::vector<std::vector<unsigned char>>(n); }
		if (subLabelPoints.size() == 0) { subLabelPoints = std::vector<std::vector<cv::Point> >(n); }

		
		// To regulate count of parts
		int partsCount = n;

		std::map<unsigned char, std::vector<int>> count;
		for (int ind = 0; ind < labels.size(); ind++) {
			count[labels[ind]].push_back(ind);
		}

		for (const auto& c : count) {
			std::vector<int> inds = c.second;
			// Variable to control size of non divided elements
			int fullSize = inds.size();
			int start = 0;
			std::vector<std::vector<int>> subInd(n);
			for (int i = 0; i < partsCount; ++i) {
				int partSize = fullSize / (partsCount - i);
				fullSize -= partSize;
				std::vector<int> temp;
				for (int j = 0; j < partSize; j++) {
					temp.push_back(inds[start + j]);
				}
				subInd[i] = temp;
				start = start + partSize;
			}
			for (int i = 0; i < subInd.size(); i++) {
				for (const auto& ind : subInd[i]) {
					subFeatures[i].push_back(features[ind]);
					subLables[i].push_back(labels[ind]);
					subLabelPoints[i].push_back(labelPoints[ind]);
				}
			}
		}
	}
}

void Utils::featureDimReduction(const std::string& hdf5_fileName, const std::string& feature_name, int filterSize, int patchSize) {
	std::vector<std::string> dataset_name = { "/feature" ,"/patchLabel" };

	std::vector<cv::Mat> data;
	Utils::readDataFromHDF(hdf5_fileName, feature_name, dataset_name, data, filterSize, patchSize);
	cv::Mat feature = data[0];
	cv::Mat patchLabels = data[1];
	
	int new_dims = 2;
	cv::Mat reduced_feature = featureDimReduction(feature, new_dims);

	//save dimension reduced features and labels to txt file
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
	applyML(newfeatures, labels, 80, "KNN", results);
	calculatePredictionAccuracy(feature_name, results, labels);
}

//input: Mat& feature, one sample per row
//new_dims: default 2
cv::Mat Utils::featureDimReduction(const cv::Mat& features, int new_dims) {
	cv::Mat feature;
	features.convertTo(feature, CV_64FC1);
	// Define some variables
	int N = feature.rows;
	int D = feature.cols;
	int perplexity = 40;
	int max_iter = 1000;
	double* X = (double*)malloc(feature.total() * sizeof(double)); // data
	double* Y = (double*)malloc(N * new_dims * sizeof(double));//output
	
	//data
	for (int i = 0; i < feature.rows; i++) {
		for (int j = 0; j < feature.cols; j++) {
			X[i * feature.cols + j] = feature.at<double>(i, j);
		}
	}
	 TSNE::run(X, N, D, Y,new_dims,perplexity,0.5,-1,false,max_iter,250,250);


	cv::Mat  reduced_feature(N, new_dims, CV_32FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < new_dims; j++) {
			reduced_feature.at<float>(i, j) = float(Y[i * new_dims + j]);
		}
	}
	
	// Clean up the memory
	free(X); X = NULL;
	free(Y); Y = NULL;
	return reduced_feature;
}