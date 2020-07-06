#include "featureProcess.hpp"

/************************************************************
Dividing the data samples into training and test samples
eg: make sure each class is divided 80% as train, 20% as test
int fold: the cross validation fold number, an integer between {1, 100 / (100 - percentOfTrain)}
Modified by: Jun 15.06.2020
return: the index of test data in the original data
*************************************************************/
std::vector<int> featureProcess::DivideTrainTestData(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_label, int percentOfTrain,
	std::vector<cv::Mat>& train_img, std::vector<unsigned char>& train_label, std::vector<cv::Mat>& test_img, std::vector<unsigned char>& test_label, int fold) {

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
		copy(it->second.begin(), it->second.end(), back_inserter(indOfClass));

		std::vector<int> train_temp, test_temp;
		int train_temp_size = 0;
		int test_temp_size = 0;
		for (size_t i = 0; i < indOfClass.size(); ++i) {
			if (train_temp_size < test_size) {
				test_temp.push_back(indOfClass[i + (fold - 1) * test_size]);
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


// shuffle the data, and record the original index of the shuffled data
std::vector<int> featureProcess::shuffleDataSet(std::vector<cv::Mat>& data, std::vector<unsigned char>& data_label) {
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

float featureProcess::calculatePredictionAccuracy(const std::string& feature_name, const std::vector<unsigned char>& classResult, const std::vector<unsigned char>& testLabels)
{
	std::string overall_accuracy = "oa_" + feature_name.substr(1) + ".txt";
	std::ofstream fout(overall_accuracy);
	if (!feature_name.empty()) {
		fout << feature_name.substr(1) << std::endl;
	}
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
			a = correct + a;
			std::cout << "accuracy for class " << std::to_string(label) << ": " << class_accuracy << std::endl;

			if (!feature_name.empty()) {
				fout << "accuracy for class " << std::to_string(label) << ": " << class_accuracy << std::endl;
			}
		}
		accuracy = a / testLabels.size();
		std::cout << "overall accuracy: " << accuracy << std::endl;

		if (!feature_name.empty()) {
			fout << "oa: " << accuracy << std::endl;
		}
	}
	return  accuracy;
}

cv::Mat featureProcess::getConfusionMatrix(const std::map<unsigned char, std::string>& className, std::vector<unsigned char>& classResult, std::vector<unsigned char>& testLabels) {
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


void featureProcess::applyML(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_labels, int trainPercent, const std::string& classifier_type, std::vector<unsigned char>& results) {

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
		std::vector<int> test_ind = featureProcess::DivideTrainTestData(temp, temp_labels, trainPercent, train, train_labels, test, test_labels, fold);
		std::vector<unsigned char> test_result;
		cv::Mat traindata, traindata_label, testdata;
		vconcat(train, traindata);
		vconcat(test, testdata);
		vconcat(train_labels, traindata_label);
		traindata_label.convertTo(traindata_label, CV_32SC1);
		traindata.convertTo(traindata, CV_32FC1);
		testdata.convertTo(testdata, CV_32FC1);

		if (classifier_type == "opencvKNN") {
			cv::Ptr<cv::ml::TrainData> cv_data = cv::ml::TrainData::create(traindata, 0, traindata_label);
			cv::Ptr<cv::ml::KNearest>  knn(cv::ml::KNearest::create());
			knn->setDefaultK(20);
			knn->setIsClassifier(true);
			knn->train(cv_data);
			for (auto& x_test : test) {
				x_test.convertTo(x_test, CV_32FC1);
				auto knn_result = knn->predict(x_test);
				test_result.push_back(unsigned char(knn_result));
			}
		}
		else if (classifier_type == "KNN") {
			KNN* knn = new KNN();
			knn->KNNTest(train, train_labels, test, test_labels, 20, test_result);
			delete knn;
		}
		else if (classifier_type == "opencvFLANN") {
			int K = 20;
			cv::flann::Index flann_index(
				traindata,
				cv::flann::KDTreeIndexParams(4),
				cvflann::FLANN_DIST_EUCLIDEAN
			);
			cv::Mat indices(testdata.rows, K, CV_32S);
			cv::Mat dists(testdata.rows, K, CV_32F);
			flann_index.knnSearch(testdata, indices, dists, K, cv::flann::SearchParams(200));
			KNN* knn = new KNN();
			for (int i = 0; i < testdata.rows; i++) {
				std::vector<std::pair<float, unsigned char>> dist_vec(K);
				for (int j = 0; j < K; j++) {
					unsigned char temp = train_labels[indices.at<int>(i, j)];
					float distance = dists.at<float>(i, j);
					dist_vec[j] = std::make_pair(distance, temp);
				}
				// voting 
				test_result.push_back(knn->Classify(dist_vec, K));
				dist_vec.clear();
			}
			delete knn;
		}
		else if (classifier_type == "opencvRF") {
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
			for (auto& x_test : test) {
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


//input: Mat& feature, one sample per row
//new_dims: default 2
cv::Mat featureProcess::featureDimReduction(const cv::Mat& features, int new_dims) {
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
	TSNE::run(X, N, D, Y, new_dims, perplexity, 0.5, -1, false, max_iter, 250, 250);


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