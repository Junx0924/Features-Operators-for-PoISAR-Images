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
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
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
Vec3b Utils::getLabelColor(unsigned char ground_truth, unsigned char class_result)
{
	 Vec3b labelColor;

	// Color is BGR not RGB!
	 Vec3b red = Vec3b(49, 60, 224); //city
	 Vec3b red_wrong = Vec3b(59, 70, 224); //city

	 Vec3b blue = Vec3b(164, 85, 50); //street
	 Vec3b blue_wrong = Vec3b(164, 95, 60); //street

	 Vec3b yellow = Vec3b(0, 190, 246); //field
	 Vec3b yellow_wrong = Vec3b(20, 190, 246); //field

	 Vec3b dark_green = Vec3b(66, 121, 79); //forest
	 Vec3b dark_green_wrong = Vec3b(76, 121, 89); //forest

	 Vec3b light_green = Vec3b(0, 189, 181); // grassland
	 Vec3b light_green_wrong = Vec3b(10, 189, 191); // grassland

	 Vec3b black = Vec3b(0, 0, 0);

	vector<Vec3b> right_color = { red,  yellow, dark_green, light_green,blue, black };
	vector<Vec3d> wrong_color = { red_wrong,  yellow_wrong, dark_green_wrong, light_green_wrong,blue_wrong, black };
	
	
	if (ground_truth == class_result) {
		labelColor = right_color[int(ground_truth)];
	}
	else {
		labelColor = wrong_color[int(ground_truth)];
	}
	
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
		for (size_t i = 0; i < indOfClass.size(); i++) {
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

void Utils::generateColorMap(const String& hdf5_fileName, const string& feature_name, const string & knn_result, int filterSize, int patchSize) {
	vector<unsigned char> labels;
	Mat pts;
	Mat labelMap;

	string dataset = "/knn";
	string parent = feature_name;
	if (patchSize != 0) { dataset = dataset + "_patchSize_" + to_string(patchSize); }
	if (filterSize != 0) { parent = feature_name + "_filterSize_" + to_string(filterSize); }

	Utils::readDataFromHDF(hdf5_fileName, feature_name, dataset,pts);
	Utils::readDataFromHDF(hdf5_fileName, "/masks", "/labelMap", labelMap);
	if(!pts.empty()){
		Mat colormap = Mat::zeros(Size(labelMap.size()), CV_8UC3);
		for (int i = 0; i < pts.rows; i++) {
			int row = pts.at<int>(i, 1);
			int col = pts.at<int>(i, 2);
			unsigned char label = unsigned char(pts.at<int>(i, 0));
			unsigned char ground_truth = labelMap.at<unsigned char>(row, col);
			colormap.at<Vec3b>(row, col) = getLabelColor(ground_truth, label);
			if (patchSize > 0) {
				for (int r = row - patchSize / 2; r < row + patchSize / 2; r++) {
					for (int c = col - patchSize / 2; c < col + patchSize / 2; c++)
						colormap.at<Vec3b>(r, c) = getLabelColor(ground_truth, label);
				}
			}
		}
		cv::imwrite("colormap.png", colormap);
	}
	else {
		cout << "can't find " << parent + dataset << " in " << hdf5_fileName << endl;
	}
	//cv::imshow("colormap of " + feature_name, colormap);
	//cv::waitKey(0);

}

void Utils::classifyFeaturesKNN(const String& hdf5_fileName,  const string& feature_name, int k, int trainPercent,int filterSize, int patchSize) {
	KNN* knn = new KNN();
	
	vector<string> feature_type = { "/texture","/color" ,"/CTelememts" ,"/polStatistic","/decomp" ,"/MP" };
	vector<string> dataset_name = { "/feature" ,"/patchLabel" };

	for (size_t i = 0; i < feature_type.size(); i++) {
		if( feature_name == feature_type[i]){
			vector<Mat> features;
			vector<Point> labelPoints;
			vector<unsigned char> labels;
			vector<unsigned char> class_results;
			cout << "get " << feature_type[i] << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << endl;

			Utils::getFeaturesFromHDF(hdf5_fileName, feature_type[i], dataset_name, features, labels, labelPoints, filterSize, patchSize);
			knn->applyKNN(features, labels, k, 80, class_results);
			Utils::saveClassResultToHDF(hdf5_fileName, feature_type[i], "/knn", class_results, labelPoints,filterSize,patchSize);

			features.clear();
			labelPoints.clear();
			labels.clear();
			class_results.clear();
		}
	}
	delete knn;
}

// save the classify result to hdf5
void Utils::saveClassResultToHDF(const String& hdf5_fileName, const String& parent_name, const string& dataset_name, const vector<unsigned char>& class_result, const vector<Point>& points,int filterSize,int patchSize) {
	Mat pts = Mat(points.size(), 3, CV_32SC1);
	for (size_t i = 0; i < points.size(); i++) {
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
		for (int i = 0; i < feature.rows; i++) {
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
	for (int i = 0; i < size; i++) {
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

double Utils::calculatePredictionAccuracy(const vector<unsigned char>& classResult, const vector<unsigned char>& testLabels)
{
	double accuracy = 0.0;
	if (classResult.size() != testLabels.size()) {
		cerr << "Predicted and actual label vectors differ in length. Somethig doesn't seem right." << endl;
		exit(-1);
	}
	else {
		int dim = classResult.size();
		double hit, miss;
		hit = 0;
		miss = 0;
		for (int i = 0; i < dim; i++) {
			if (classResult[i] == testLabels[i]) {
				hit++;
			}
			else {
				miss++;
			}
		}
		accuracy = double(hit / dim);
	}
	return accuracy;
}

Mat Utils::getConfusionMatrix(const map<unsigned char, string>& className, vector<unsigned char>& classResult, vector<unsigned char>& testLabels) {
	map<pair<unsigned char, signed char>, int> testCount;

	for (int i = 0; i < testLabels.size(); i++) {
		for (int j = 0; j < classResult.size(); j++) {
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
	for (int i = 0; i < numOfClass; i++) {
		for (int j = 0; j < numOfClass; j++) {
			pair temp = make_pair(classList[i], classList[j]);
			count.at<unsigned char>(i, j) = testCount[temp];
		}
	}
	return count;
}

void Utils::deleteDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, int filterSize, int patchSize) {
	string parent = parent_name;

	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }
	for (int i = 0; i < dataset_name.size(); i++) {
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
		for (int i = 0; i < data.size(); i++) {
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
			for (size_t i = 0; i < data.total() * data.channels(); i++)
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
			double diff = norm(data - expected);
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
		for (int i = 0; i < dataset_name.size(); i++) {
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
				//cout << "insert " << data.rows << " rows to" << dataset << " success " << endl;
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
		for (int i = 0; i < data.size(); i++) {
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

	cout << "start to generate sample points with patchSize " << patchSize << ", stride " << stride << "..." << endl;

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

void Utils::getRandomSamplePoint(const Mat& labelMap, vector<Point>& samplePoints, vector<unsigned char> & sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass) {
	 
	vector<Point> temp = generateSamplePoints(labelMap, sampleSize, stride);
	if (numOfSamplePointPerClass > 0) {
		map<unsigned char, vector<Point>> count;
		for (auto& p : temp) {
			unsigned char label = labelMap.at<unsigned char>(p.y, p.x);
			count[label].push_back(p);
		}
	
		for (auto const& c : count)
		{
			unsigned char label = c.first;
			if(label !=unsigned char(0)){
				vector<Point> rows = c.second;
				std::random_device random_device;
				std::mt19937 engine{ random_device() };
				std::uniform_int_distribution<int> pt(0, rows.size() - 1);
				size_t num = 0;
				size_t iter = 0;

				while (num < numOfSamplePointPerClass) {
					Point p = rows[pt(engine)];

					// get samples in homogeneous areas 
					// this is only for checking the accuracy of features
					unsigned char label = labelMap.at<unsigned char>(p.y, p.x);
					unsigned char sample_upcorner = labelMap.at<unsigned char>(p.y - sampleSize / 2, p.x - sampleSize / 2);
					unsigned char sample_downcorner = labelMap.at<unsigned char>(p.y + sampleSize / 2, p.x + sampleSize / 2);
					unsigned char sample_leftcorner = labelMap.at<unsigned char>(p.y + sampleSize / 2, p.x - sampleSize / 2);
					unsigned char sample_rightcorner = labelMap.at<unsigned char>(p.y - sampleSize / 2, p.x + sampleSize / 2);
					if ((label == sample_upcorner) && (label == sample_downcorner) &&
						(label == sample_leftcorner) && (label == sample_rightcorner) ) {
						samplePoints.push_back(p);
						sampleLabel.push_back(label);
						num++;
					}
					iter++;
					if (iter > rows.size()) { break; }
				}
		    }
		}
	}
	 else {
		 cout << "load all the sample points" << endl;
		 copy(temp.begin(), temp.end(), back_inserter(samplePoints));
		 for (auto& p : samplePoints) {
			 sampleLabel.push_back(labelMap.at<unsigned char>(p.y, p.x));
		 }
	 }
}

/*===================================================================
 * Function: getSafeSamplePoints
 * Author: Jun Xiang
 *
 * Summary:
 *   Extract sample points from mask area or any img
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
						count = new_ind.size();
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
	} 
}