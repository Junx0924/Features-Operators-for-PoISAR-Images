#include "Utils.h"

using namespace std;
using namespace cv;

/**********************************
Generating a label map
Author : Anupama Rajkumar
Date : 27.05.2020
Modified by: Jun Xiang 22,06,2020
Description : Idea is to create a single label map from a list of various
label classes.This map serves as points of reference when trying to classify
patches
* ************************************************************************/

Mat Utils::generateLabelMap(const vector<Mat> & masks) {
	size_t NUMOFCLASSES = masks.size();
	int rows = masks[0].rows;
	int cols = masks[0].cols;
	Mat labelMap = Mat::zeros(rows, cols, CV_8UC1);
	for (size_t cnt = 0; cnt < NUMOFCLASSES; cnt++) {
		Mat mask = masks[cnt];
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
Vec3b Utils::getLabelColor(unsigned char class_result)
{
	 Vec3b labelColor;

	// Color is BGR not RGB!
	 Vec3b red = Vec3b(49, 60, 224); //city

	 Vec3b blue = Vec3b(164, 85, 50); //street

	 Vec3b yellow = Vec3b(0, 190, 246); //field

	 Vec3b dark_green = Vec3b(66, 121, 79); //forest

	 Vec3b light_green = Vec3b(0, 189, 181); // grassland

	 Vec3b black = Vec3b(0, 0, 0);

	vector<Vec3b> right_color = { red,  yellow, dark_green, light_green,blue, black };
	
	
	labelColor = right_color[int(class_result)-1];
	 
	return labelColor;
}



/***********************************************************************
Helper function to get the patch start and end index from label map
make sure the patch stays in one type of class areas
Author : Anupama Rajkumar
Date : 29.05.2020
Modified by: Jun Xiang
*************************************************************************/

void Utils::GetPatchIndex(int sizeOfPatch, Point2i &samplePoint,const Mat& LabelMap, int& start_col, int& start_row, int& end_col, int& end_row) {
	// (x,y)->(col,row)
	start_col = samplePoint.x - (sizeOfPatch / 2.);
	start_row = samplePoint.y - (sizeOfPatch / 2.);

	end_col = samplePoint.x + (sizeOfPatch / 2.);
	end_row = samplePoint.y + (sizeOfPatch / 2.);

	if ((start_col < 0) || (start_row < 0))
	{
		start_col = 0;
		start_row = 0;
		end_col = sizeOfPatch;
		end_row = sizeOfPatch;
	}
	if ((end_row > LabelMap.rows) || (end_col > LabelMap.cols))
	{
		start_row = LabelMap.rows - 1 - sizeOfPatch;
		start_col = LabelMap.cols - 1 - sizeOfPatch;
		end_row = LabelMap.rows - 1;
		end_col = LabelMap.cols - 1;
	}
}




/************************************************************
Dividing the data samples into training and test samples
eg: make sure each class is divided 80% as train, 20% as test
int fold: the cross validation fold number, an integer between {1, 100 / (100 - percentOfTrain)}
Modified by: Jun 15.06.2020
return: the index of test data in the original data 
*************************************************************/
vector<int> Utils::DivideTrainTestData(const vector<Mat> &data, const vector<unsigned char> & data_label, int percentOfTrain,
	vector<Mat> & train_img,  vector<unsigned char> &train_label, vector<Mat>& test_img, vector<unsigned char> & test_label,int fold) {
	
	std::map<unsigned char, vector<int>> numPerClass;
	int index = 0;
	for (auto c : data_label) { 
		numPerClass[c].push_back(index); 
		index++;
	}
	vector<int> train_index, test_index;
	int total_folds = 100 / (100 - percentOfTrain);

	// make sure each class is divided 80% as train, 20% as test
	for (auto it = numPerClass.begin(); it != numPerClass.end(); it++)
	{
		size_t train_size = it->second.size() * percentOfTrain / 100;
		size_t test_size = it->second.size() - train_size;

		vector<int> indOfClass;
		// expand indOfClass twice
		copy(it->second.begin(), it->second.end(), back_inserter(indOfClass));
		copy(it->second.begin(),it->second.end(), back_inserter(indOfClass));

		vector<int> train_temp, test_temp;
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

void Utils::generateColorMap(const String& hdf5_fileName, const string& feature_name, const string & result, int filterSize, int patchSize) {
	vector<unsigned char> labels;
	Mat pts;
	Mat labelMap;

	string dataset = result;
	string parent = feature_name;
	if (patchSize != 0) { dataset = dataset + "_patchSize_" + to_string(patchSize); }
	if (filterSize != 0) { parent = feature_name + "_filterSize_" + to_string(filterSize); }

	Utils::readDataFromHDF(hdf5_fileName, feature_name, dataset,pts);
	Utils::readDataFromHDF(hdf5_fileName, "/masks", "/labelMap", labelMap);
	if(!pts.empty()){
		Mat colormap = Mat::zeros(Size(labelMap.size()), CV_8UC3);
		Mat colorLabelMap = Mat::zeros(Size(labelMap.size()), CV_8UC3);
		for (int i = 0; i < pts.rows; ++i) {
			int row = pts.at<int>(i, 1);
			int col = pts.at<int>(i, 2);
			unsigned char label = unsigned char(pts.at<int>(i, 0));
			 unsigned char ground_truth = labelMap.at<unsigned char>(row, col);
			colormap.at<Vec3b>(row, col) = getLabelColor( label);
			colorLabelMap.at<Vec3b>(row, col) = getLabelColor(ground_truth);
			if (patchSize >0) {
				for (int r = row - patchSize / 2; r < row + patchSize / 2; ++r) {
					for (int c = col - patchSize / 2; c < col + patchSize / 2; ++c){
						colormap.at<Vec3b>(r, c) = getLabelColor(label);
					    colorLabelMap.at<Vec3b>(r, c) = getLabelColor(ground_truth);
					}
				}
			}
		}
		cout << "generate " << feature_name.substr(1) + "_colormap.png" << endl;
		cv::imwrite(feature_name.substr(1) + "_colormap.png", colormap);
		cv::imwrite("colorLabelMap.png", colorLabelMap);
	}
	else {
		cout << "can't find " << parent + dataset << " in " << hdf5_fileName << endl;
	}
}

void Utils::classifyFeaturesKNN(const String& hdf5_fileName,  const string& feature_name, int k, int trainPercent,int filterSize, int patchSize) {
	KNN* knn = new KNN();
	
	vector<string> feature_type = { "/texture","/color" ,"/CTelememts" ,"/polStatistic","/decomp" ,"/MP" };
	vector<string> dataset_name = { "/feature" ,"/patchLabel" };
	cout << endl;
	for (size_t i = 0; i < feature_type.size(); ++i) {
		if( feature_name == feature_type[i]){
			vector<Mat> features;
			vector<Point> labelPoints;
			vector<unsigned char> labels;
			vector<unsigned char> class_results;
			Utils::getFeaturesFromHDF(hdf5_fileName, feature_type[i], dataset_name, features, labels, labelPoints, filterSize, patchSize);
			if(!features.empty()){
				cout << "get " << features.size()<<" rows for "<< feature_type[i] << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << endl;
			    knn->applyKNN(features, labels, k, 80, class_results);

				if (Utils::checkExistInHDF(hdf5_fileName, feature_type[i], { "/knn" }, filterSize, patchSize)){
					Utils::deleteDataFromHDF(hdf5_fileName, feature_type[i], { "/knn" },filterSize, patchSize);
				}

				Utils::saveClassResultToHDF(hdf5_fileName, feature_type[i], "/knn", class_results, labelPoints, filterSize, patchSize);
				features.clear();
				labelPoints.clear();
				labels.clear();
				class_results.clear();
			}
			else {
				cout << feature_type[i] << " with filterSize " << filterSize << " , patchSize " << patchSize << " is not existed in hdf5 file " << endl;
			}
		}
	}
	delete knn;
}

// classifier_type: choose from {"KNN", "RF"}
void Utils::classifyFeaturesML(const String& hdf5_fileName, const string& feature_name, const string classifier_type, int trainPercent, int filterSize, int patchSize) {

	vector<string> feature_type = { "/texture","/color" ,"/CTelememts" ,"/polStatistic","/decomp" ,"/MP" };
	vector<string> dataset_name = { "/feature" ,"/patchLabel" };
	cout << endl;
	for (size_t i = 0; i < feature_type.size(); ++i) {
		if (feature_name == feature_type[i]) {
			vector<Mat> features;
			vector<Point> labelPoints;
			vector<unsigned char> labels;
			vector<unsigned char> class_results;
			Utils::getFeaturesFromHDF(hdf5_fileName, feature_type[i], dataset_name, features, labels, labelPoints, filterSize, patchSize);
			if (!features.empty()) {
				cout << "get " << features.size() << " rows for " << feature_type[i] << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << endl;
				applyML(features, labels,  80, classifier_type, class_results);

				if (Utils::checkExistInHDF(hdf5_fileName, feature_type[i], { "/"+ classifier_type }, filterSize, patchSize)) {
					Utils::deleteDataFromHDF(hdf5_fileName, feature_type[i], { "/" + classifier_type }, filterSize, patchSize);
				}

				Utils::saveClassResultToHDF(hdf5_fileName, feature_type[i], "/" + classifier_type, class_results, labelPoints, filterSize, patchSize);
				features.clear();
				labelPoints.clear();
				labels.clear();
				class_results.clear();
			}
			else {
				cout << feature_type[i] << " with filterSize " << filterSize << " , patchSize " << patchSize << " is not existed in hdf5 file " << endl;
			}
		}
	}
}

// save the classify result to hdf5
void Utils::saveClassResultToHDF(const String& hdf5_fileName, const String& parent_name, const string& dataset_name, const vector<unsigned char>& class_result, const vector<Point>& points,int filterSize,int patchSize) {
	Mat pts = Mat(points.size(), 3, CV_32SC1);
	for (size_t i = 0; i < points.size(); ++i) {
		pts.at<int>(i, 0) = (int)(class_result[i]);
		pts.at<int>(i, 1) = points[i].y; //row
		pts.at<int>(i, 2) = points[i].x; //col
	}
	string dataset = dataset_name;
	string parent = parent_name;
	if (patchSize != 0) { dataset = dataset_name + "_patchSize_" + to_string(patchSize); }
	if (filterSize != 0) { parent = parent_name + "_filterSize_" + to_string(filterSize); }
	Utils::insertDataToHDF(hdf5_fileName, parent_name, dataset, pts);
}


// get features data from hdf5
// features and featureLabels for train and test
// labelPoints for the location in image
void Utils::getFeaturesFromHDF(const String& hdf5_fileName, const String& parent_name, const vector<String>& dataset_name,
	vector<Mat>& features, vector<unsigned char>& featureLabels, vector<Point>& labelPoints, int filterSize, int patchSize) {
	vector<Mat> data;
	if (Utils::checkExistInHDF(hdf5_fileName, parent_name, dataset_name, filterSize, patchSize)) {
		Utils::readDataFromHDF(hdf5_fileName, parent_name, dataset_name, data, filterSize, patchSize);
		Mat feature = data[0];
		Mat pts = data[1]; //labelPoint
		for (int i = 0; i < feature.rows; ++i) {
			features.push_back(feature.row(i));
			featureLabels.push_back((unsigned char)(pts.at<int>(i, 0)));
			Point p;
			p.y = pts.at<int>(i, 1); //row
			p.x = pts.at<int>(i, 2); //col
			labelPoints.push_back(p);
		}
	}
}

// shuffle the data, and record the original index of the shuffled data
vector<int> Utils::shuffleDataSet(vector<Mat>& data, vector<unsigned char>& data_label) {
	int size = data.size();
	vector<int> ind(size);
	std::random_device random_device;
	std::mt19937 engine{ random_device() };
	std::uniform_int_distribution<int> rnd(0, size - 1);
	for (int i = 0; i < size; ++i) {
		Mat temp = data[i];
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

float Utils::calculatePredictionAccuracy(const vector<unsigned char>& classResult, const vector<unsigned char>& testLabels)
{
	float accuracy = 0.0;
	if (classResult.size() != testLabels.size()) {
		cerr << "Predicted and actual label vectors differ in length. Somethig doesn't seem right." << endl;
		exit(-1);
	}
	else {
		int dim = classResult.size();
		float hit, miss;
		hit = 0;
		miss = 0;
		for (int i = 0; i < dim; ++i) {
			if (classResult[i] == testLabels[i]) {
				hit++;
			}
			else {
				miss++;
			}
		}
		accuracy = float(hit / dim);
	}
	return accuracy;
}

Mat Utils::getConfusionMatrix(const map<unsigned char, string>& className, vector<unsigned char>& classResult, vector<unsigned char>& testLabels) {
	map<pair<unsigned char, signed char>, int> testCount;

	for (int i = 0; i < testLabels.size(); ++i) {
		for (int j = 0; j < classResult.size(); ++j) {
			pair temp = make_pair(testLabels[i], classResult[j]);
			testCount[temp]++;
		}
	}

	int numOfClass = className.size();
	vector<unsigned char> classList(numOfClass);
	for (auto it = className.begin(); it != className.end(); it++) {
		classList.push_back(it->first);
	}

	Mat count = Mat(className.size(), className.size(), CV_8UC1);
	for (int i = 0; i < numOfClass; ++i) {
		for (int j = 0; j < numOfClass; ++j) {
			pair temp = make_pair(classList[i], classList[j]);
			count.at<unsigned char>(i, j) = testCount[temp];
		}
	}
	return count;
}

void Utils::deleteDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, int filterSize, int patchSize) {
	string parent = parent_name;

	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }
	for (int i = 0; i < dataset_name.size(); ++i) {
		if (patchSize != 0) {
			deleteDataFromHDF(filename, parent, dataset_name[i] + "_patchSize_" + to_string(patchSize));
		}
		else {
			deleteDataFromHDF(filename, parent, dataset_name[i]);
		}
	}
}


void Utils::deleteDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		cout << parent_name << " is not existed." << endl;
	}else {
		if (!h5io->hlexists(datasetName)) {
			cout << datasetName << " is not existed." << endl;
		}else {
	        int result = h5io->dsdelete(datasetName);
			if (!result) {
				cout << "delete dataset " << datasetName << " success." << endl;
			}
			else {
				cout << "Failed to delete " << datasetName << endl;
			}
		}
	}
}



void Utils::writeDataToHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, const vector<Mat>& data, int filterSize , int patchSize) {
	string parent = parent_name;
	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }

	if (data.size() == dataset_name.size()) {
		for (int i = 0; i < data.size(); ++i) {
			if(patchSize !=0){
			writeDataToHDF(filename, parent, dataset_name[i]+ "_patchSize_"+ to_string(patchSize), data[i]);
			}
			else {
				writeDataToHDF(filename, parent, dataset_name[i] , data[i]);
			}
		}
	}
	else {
		cout << "the size of dataset_name doesn't match that of data" << endl;
	}
}

void Utils::writeDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& src) {
	if(!src.empty()){
		Mat data = src.clone();
		if (data.channels() > 1) {
			for (size_t i = 0; i < data.total() * data.channels(); ++i)
				((int*)data.data)[i] = (int)i;
		}

		Ptr<hdf::HDF5> h5io = hdf::open(filename);

		// first we need to create the parent group
		if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

		// create the dataset if it not exists
		string datasetName = parent_name + dataset_name;
		if (!h5io->hlexists(datasetName)) {
			h5io->dscreate(data.rows, data.cols, data.type(), datasetName);
			h5io->dswrite(data, datasetName);

			// check if the data are correctly write to hdf file
			Mat expected = Mat(Size(data.size()), data.type());
			h5io->dsread(expected, datasetName);
			float diff = norm(data - expected);
			CV_Assert(abs(diff) < 1e-10);

			if (h5io->hlexists(datasetName))
			{
				//cout << "write " << datasetName << " to " << filename << " success." << endl;
			}
			else {
				cout << "Failed to write " << datasetName << " to " << filename << endl;
			}
		}
		else {
			cout << datasetName << " is already existed." << endl;
		}
		h5io->close();
	}
}

void Utils::readDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, vector<Mat>& data, int filterSize, int patchSize) {
	string parent = parent_name;
	if (!data.empty()) { data.clear(); }

	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }
		for (int i = 0; i < dataset_name.size(); ++i) {
			Mat temp;
			if (patchSize != 0) {
				readDataFromHDF(filename, parent, dataset_name[i] + "_patchSize_" + to_string(patchSize), temp);
			}
			else {
				readDataFromHDF(filename, parent, dataset_name[i], temp);
			}
			if (!temp.empty()) { data.push_back(temp); }
		}
}

void Utils::readDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name, Mat& data) {

	Ptr<hdf::HDF5> h5io = hdf::open(filename);

	string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		//cout << parent_name << " is not existed" << endl;
		data = Mat();
	}
	else if (!h5io->hlexists(datasetName) ) { 
		//cout << datasetName << " is not existed" << endl;  
		data = Mat(); 
	} else {
		vector<int> data_size = h5io->dsgetsize(datasetName);

		data = Mat(data_size[0],data_size[1],h5io->dsgettype(datasetName));

	    h5io->dsread(data, datasetName);
		//cout << "get " <<  datasetName  << " success" << endl;
	}

	h5io->close();
}


void Utils::writeAttrToHDF(const String& filename,const String& attribute_name,const int &attribute_value) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		cout << " already existed" << endl;
	}
	h5io->close();
}

void Utils::writeAttrToHDF(const String& filename, const String& attribute_name, const string &attribute_value) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		cout << " already existed" << endl;
	}
	h5io->close();
}

void Utils::readAttrFromHDF(const String& filename, const String& attribute_name,  string &attribute_value) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		cout << attribute_name<<" is not existed" << endl;
	}
	else {
		h5io->atread(&attribute_value, attribute_name);
	}
	h5io->close();
}

void Utils::readAttrFromHDF(const String& filename, const String& attribute_name, int& attribute_value) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		cout << attribute_name << " is not existed" << endl;
	}
	else {
		h5io->atread(&attribute_value, attribute_name);
	}
	h5io->close();
}

Mat Utils::readTiff(string filepath) {
	const char* file = filepath.c_str();
	GeoTiff* tiff = new GeoTiff(file);
	cout << "this tiff file has: cols " << tiff->NCOLS << " , rows " << tiff->NROWS << " , channels " << tiff->NLEVELS << endl;
	Mat data = tiff->GetMat().clone();
	delete tiff;
	return data;
}

bool Utils::checkExistInHDF(const String& filename, const String& parent_name, const string& dataset_name) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	bool flag = true;
	
	if (!h5io->hlexists(parent_name)) {
		flag = false;
		//cout << parent_name << " is not existed" << endl;
	}else if (!h5io->hlexists(parent_name + dataset_name)) {
		flag = false;
		//cout << parent_name + dataset_name << " is not existed" << endl;
	}
	h5io->close();
	return flag;
}


bool Utils::checkExistInHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name, int filterSize, int patchSize ) {
	bool flag = true;
	string parent = parent_name;
	vector<string> dataset = dataset_name;
	if (filterSize != 0) {
		parent = parent_name + "_filterSize_" + to_string(filterSize);
	}

	for (auto& n : dataset) {
		if(patchSize !=0){
		   n =  n + "_patchSize_" + to_string(patchSize);
		}
		bool temp = checkExistInHDF(filename, parent, n);
		flag = flag && temp;
	}
	return	flag;
}


bool Utils::insertDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& data) {
	bool flag = true;
	if(!data.empty()){
		Ptr<hdf::HDF5> h5io = hdf::open(filename);

		if (checkExistInHDF(filename, parent_name, dataset_name)) {
			string dataset = parent_name + dataset_name;
			vector<int> data_size = h5io->dsgetsize(dataset);
			// expand the dataset at row direction
			int offset[2] = { data_size[0],0 };

			if ((h5io->dsgettype(dataset) == data.type()) && (data_size[1] == data.cols)) {
				h5io->dsinsert(data, dataset, offset);

				//check if insert success
				//cout << endl;
				//cout << "insert " << data.rows << " rows to " << dataset << " success " << endl;
				//cout << dataset << " rows in total: " << data.rows + offset[0] << endl;
			}

			else {
				flag = false;
				cout << endl;
				cout << " the new data has different size and type with the existed data" << endl;
				cout << dataset << " insert failed" << endl;
			}
		}
		else {
			// first we need to create the parent group
			if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

			string dataset = parent_name + dataset_name;
			int chunks[2] = { 1, data.cols };
			// create Unlimited x data.cols, data.type() space, dataset can be expanded unlimted on the row direction
			h5io->dscreate(hdf::HDF5::H5_UNLIMITED, data.cols, data.type(), dataset, hdf::HDF5::H5_NONE, chunks);
			// the first time to write data, offset at row,col direction is 0
			int offset[2] = { 0, 0 };
			h5io->dsinsert(data, dataset, offset);
			cout << endl;
			//cout << "insert " << data.rows << " rows to" << dataset << " success " << endl;
			//cout << dataset << " rows in total: " << data.rows + offset[0] << endl;
		}
	}
	return flag;
}

bool Utils::insertDataToHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name, const vector<Mat>& data, int filterSize, int patchSize) {
	bool flag = true;
	string parent = parent_name;
	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }

	if (data.size() == dataset_name.size()) {
		for (int i = 0; i < data.size(); ++i) {
			bool temp;
			if (patchSize != 0) {
				temp =insertDataToHDF(filename, parent, dataset_name[i] + "_patchSize_" + to_string(patchSize), data[i]);
			}
			else {
				temp =insertDataToHDF(filename, parent, dataset_name[i], data[i]);
			}
			flag = flag && temp;
		}
	}
	else {
		flag = false;
		cout << "the size of dataset_name doesn't match that of data"<<endl;
	}
	return flag;
}


vector<Point> Utils::generateSamplePoints(const Mat& labelMap, const int& patchSize, const int& stride) {

	vector<Point> samplePoints;
	for (int row = 0; row < labelMap.rows - patchSize; row += stride) {
		for (int col = 0; col < labelMap.cols - patchSize; col += stride) {
			Rect cell = Rect(col, row, patchSize, patchSize);

			int halfsize = patchSize / 2;
			
			//record the central points of each patch
			samplePoints.push_back(Point(col + halfsize, row + halfsize));
		}
	}
	return samplePoints;
}

void Utils::getRandomSamplePoint(const Mat& labelMap, vector<Point>& samplePoints, const unsigned char &sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass) {
	 
	vector<Point> temp = generateSamplePoints(labelMap, sampleSize, stride);
	map<unsigned char, vector<Point>> count;
	for (auto& p : temp) {
		unsigned char label = labelMap.at<unsigned char>(p.y, p.x);
		if (label == sampleLabel) {
			count[sampleLabel].push_back(p);
		}
	}
	vector<Point> pts = count[sampleLabel];

	if (numOfSamplePointPerClass > 0) {
		std::random_device random_device;
		std::mt19937 engine{ random_device() };
		std::uniform_int_distribution<int> pt(0, pts.size() - 1);
		size_t num = 0;
		size_t iter = 0;

		while (num < numOfSamplePointPerClass) {
			Point p = pts[pt(engine)];

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
		 cout << "load all the sample points" << endl;
		 copy(pts.begin(), pts.end(), back_inserter(samplePoints));
	 }
}


void Utils::applyML(const vector<Mat>& data, const vector<unsigned char>& data_labels,int trainPercent, const string & classifier_type, vector<unsigned char>& results) {

	cout << "start to classify data with classifier :" << classifier_type << endl;

	// classify result
	 results = vector<unsigned char>(data_labels.size());

	//copy the original data
	vector<Mat> temp(data.begin(), data.end());
	vector<unsigned char> temp_labels(data_labels.begin(), data_labels.end());

	vector<Mat> train;
	vector<unsigned char> train_labels;
	vector<Mat> test;
	vector<unsigned char> test_labels;

	int total_folds = 100 / (100 - trainPercent);
	float accuracy = 0.0;
	for (int fold = 1; fold < total_folds + 1; ++fold) {
		float acc = 0.0;
		vector<int> test_ind = Utils::DivideTrainTestData(temp, temp_labels, trainPercent, train, train_labels, test, test_labels, fold);
		vector<unsigned char> test_result;
		Mat traindata, traindata_label, testdata;
		vconcat(train, traindata);
		vconcat(train_labels, traindata_label);
		traindata_label.convertTo(traindata_label, CV_32FC1);
		traindata.convertTo(traindata, CV_32FC1);
		cv::Ptr<cv::ml::TrainData> cv_data = cv::ml::TrainData::create(traindata, 0, traindata_label);

		if (classifier_type == "KNN") {
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
		else if (classifier_type == "RF") {
			cv::Ptr<cv::ml::RTrees>  randomForest(cv::ml::RTrees::create());
			auto criterRamdomF = cv::TermCriteria();
			criterRamdomF.type = cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS;
			criterRamdomF.epsilon = 1e-8;
			criterRamdomF.maxCount = 5000;
			randomForest->setMaxCategories(2);
			randomForest->setMaxDepth(3000);
			randomForest->setMinSampleCount(1);
			randomForest->setTruncatePrunedTree(false);
			randomForest->setUse1SERule(false);
			randomForest->setUseSurrogates(false);
			randomForest->setPriors(cv::Mat());
			randomForest->setTermCriteria(criterRamdomF);
			randomForest->setCVFolds(1);

			randomForest->train(cv_data);
			for (auto const& x_test : test) {
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
		acc =  count / test.size();
		accuracy = accuracy + acc;
		 
		train.clear();
		train_labels.clear();
		test.clear();
		test_labels.clear();
		test_result.clear();
	}

	accuracy = accuracy / total_folds;
	cout << "cross validation accuracy: " << accuracy << endl;
}