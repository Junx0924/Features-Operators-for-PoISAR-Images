#include "FeatureProcess.h"

/*===================================================================
 * Function: caculFeatures
 *
 * Summary:
 *   shuffle the samples and split them into batches with porper class distribution
 *	 calulate features and save to hdf5 file
 *
 * Arguments:
 *	 std::vector<cv::Mat> data : complex mat with values [HH, VV, HV]
 *   cv::Mat LabelMap 
 *   std::map<unsigned char, std::string>classNames : record the class name of each label
 *	 int numOfSamplePoint - the number of samples for one type of class, default 0 means to load all the possible sample points
 *   int stride : default 1
 *   unsigned char classlabel  -  choose which class to load, default 255 means to load all the classes
 *
 * Output:
 *   
=====================================================================
*/
void FeatureProcess::caculFeatures(const std::vector<cv::Mat>& data, const cv::Mat& LabelMap, const std::map<unsigned char, std::string>& classNames, int numOfSamplePoint, int stride , unsigned char classlabel) {

	writeLabelMapToHDF(this->hdf5_file, data, LabelMap, classNames);

	std::vector<cv::Point> samplePoints;
	std::vector<unsigned char> sampleLabel;

	LoadSamplePoints(samplePoints,sampleLabel, LabelMap, classNames, this->patch_Size, numOfSamplePoint, stride, classlabel);

	std::map<std::string, int> feature_type = { {"texture",1},{"color",2},{"ctelements",3},{"polstatistic",4},{"decomp",5},{"mp",6} };
	std::vector<std::string> dataset_name = { "/feature" ,"/groundtruth" };
	std::string parent = "/" + this->featureName + "_filterSize_" + std::to_string(this->filter_Size) + "_patchSize_" + std::to_string(this->patch_Size);
	std::cout << "start to calculate " << this->featureName << " with filterSize " << this->filter_Size << " , patchSize " << this->patch_Size << std::endl;

	//shuffle all the samplePoints based on its label
	//make sure each class has right portion in each batch
	std::vector<std::vector<int>> subInd;
	DataProcess::splitVec(sampleLabel, subInd, this->batch_Size);


	for (auto i = 0; i < subInd.size(); i++) {
		std::vector<int> ind = subInd[i];
		size_t N = ind.size();
		cv::Mat feature, pts;

		for (size_t j = 0; j < N; ++j) {
			cv::Point p = samplePoints[ind[j]];
			int patchLabel = sampleLabel[ind[j]];

			cv::Mat temp = cv::Mat(1, 3, CV_32SC1);
			temp.at<int>(0, 0) = patchLabel;
			temp.at<int>(0, 1) = p.y; //row
			temp.at<int>(0, 2) = p.x; //col
			pts.push_back(temp);
			cv::Mat hh, vv, hv;
			getSample(data, p, this->patch_Size, this->filter_Size, hh, vv, hv);
			cv::Mat temp_feature;
			switch (feature_type[this->featureName]) {
			case 1:
				temp_feature = caculTexture(hh, vv, hv);
				break;
			case 2:
				temp_feature = caculColor(hh, vv, hv);
				break;
			case 3:
				temp_feature = caculCTelements(hh, vv, hv);
				break;
			case 4:
				temp_feature = caculPolStatistic(hh, vv, hv);
				break;
			case 5:
				temp_feature = caculDecomp(hh, vv, hv);
				break;
			case 6:
				temp_feature = caculMP(hh, vv, hv);
				break;
			default:
				std::cout << "feature name not existed:" << this->featureName << std::endl;
				exit(-1);
				break;
			}
			feature.push_back(temp_feature);
		}
		hdf5::insertData(this->hdf5_file, parent, dataset_name[0], feature);
		hdf5::insertData(this->hdf5_file, parent, dataset_name[1], pts);
		std::cout << "calculate " << this->featureName << " progress: " << float(i + 1) / float(subInd.size()) * 100.0 << "% \n" << std::endl;
		feature.release();
		pts.release();
	}
}

/*===================================================================
 * Function: generateColorMap
 *
 * Summary:
 *   Generate the colormap of classified results, calculate the overall accuracy for each class
 *
 * Arguments:
 *   std::string & classifier_type - choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
 * Returns:
 *   void
=====================================================================
*/
void FeatureProcess::generateColorMap(const std::string& classifier_type) {
	std::string parent = "/" + this->featureName + "_filterSize_" + std::to_string(this->filter_Size) + "_patchSize_" + std::to_string(patch_Size);

	int totalrows = hdf5::getRowSize(this->hdf5_file, parent, "/" + classifier_type);

	std::vector<unsigned char> labels;
	cv::Mat labelMap;
	hdf5::readData(this->hdf5_file, "/masks", "/labelMap", labelMap);
	if (totalrows > 0) {
		std::vector<unsigned char> class_results;
		std::vector<unsigned char> ground_truth_labels;
		cv::Mat colorResultMap = cv::Mat::zeros(cv::Size(labelMap.size()), CV_8UC3);
		cv::Mat groundTruthMap = cv::Mat::zeros(cv::Size(labelMap.size()), CV_8UC3);

		int offset_row = 0;
		int partSize;
		if (this->batch_Size > totalrows) { this->batch_Size = totalrows; }
		int partsCount = totalrows / this->batch_Size;
		for (int i = 0; i < partsCount; ++i) {
			partSize = totalrows / (partsCount - i);
			totalrows -= partSize;
			if (totalrows < 0) { break; }
			cv::Mat pts;
			hdf5::readData(this->hdf5_file, parent, "/" + classifier_type, pts, offset_row, partSize);

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

		std::cout << std::endl;
		std::cout << "classifier: " << classifier_type << std::endl;
		std::cout << "generate " << this->featureName + "_classresult.png" << std::endl;

		cv::imwrite(this->featureName + "_colormap.png", colorResultMap);
		cv::imwrite("groundTruthMap.png", groundTruthMap);

		std::map<unsigned char, std::string> className = this->getClassName(this->hdf5_file);
		DataProcess::calculatePredictionAccuracy(this->featureName, class_results, ground_truth_labels, className);
	}
	else {
		std::cout << "can't find " << parent + "/" + classifier_type << " in " << this->hdf5_file << std::endl;
	}
}

/*===================================================================
 * Function: generateFeatureMap
 *
 * Summary:
 *   Generate the visulization of feature map for each single feature in feature group
 *
 * Returns:
 *   void
=====================================================================
*/
void FeatureProcess::generateFeatureMap() {
	std::string parent = "/" + this->featureName + "_filterSize_" + std::to_string(filter_Size) + "_patchSize_" + std::to_string(patch_Size);
	std::vector<std::string> dataset_name = { "/labelMap", "/groundtruth", "/feature" };

	//create folder for feature map images
	std::string outputpath = "featureMap_" + this->featureName;
	std::filesystem::create_directories(outputpath);

	std::vector<int> cols;
	std::vector<std::string> png_name;
	if (this->featureName == "polstatistic") {
		std::cout << "generate the feature map for each polarimetric parameter" << std::endl;
		// the median value for each polarimetric parameter
		cols = { 0,5,10,15,20,25,30,35,40,45 };
		png_name = { "Intensity_of_HH_channel","Intensity_of_HV_channel","Intensity_of_VV_channel","Phase_difference_HH-VV",
			"Co-polarize_ratio","Cross-polarized_ratio","HV_VV_ratio","Copolarization_ratio","Depolarization_ratio","Amplitude_of_HH-VV_correlation" };
	}
	else if (this->featureName == "ctelements") {
		std::cout << "generate the feature map for each upper conner elements from C /T matrix" << std::endl;
		cols = { 4,13,22,31,40,49,58,67,76,85,94,103 };
		png_name = { "T00","T01","T02","T11","T12","T22" ,"C00","C01","C02", "C11", "C12", "C22" };
	}
	else if (this->featureName == "decomp") {
		std::cout << "generate the feature map for target decomposition component" << std::endl;
		cols = { 4,13,22,31,40,49,58,67,76,85,94,103 };
		png_name = { "cloude_entropy", "cloude_anisotropy", "cloude_alpha", "freeman_surface", "freeman_double-bounce" , "freeman_volume", "krogager_sphere", "krogager_diplane", "krogager_helix", "pauli_alpha" ,"pauli_beta", "pauli_gamma" };
	}
	else if (this->featureName == "color") {
		std::cout << "generate the feature map for MPEG-7 DCD and CSD" << std::endl;
		for (int i = 0; i < 32; i++) { cols.push_back(i); png_name.push_back("csd bin " + std::to_string(i + 1)); }
		cols.push_back(32); png_name.push_back("dominant_color_value1");
		cols.push_back(33); png_name.push_back("dominant_color_value2");
		cols.push_back(34); png_name.push_back("dominant_color_value3");
		cols.push_back(35); png_name.push_back("dominant_color_weight");
	}
	else if (this->featureName == "texture") {
		std::cout << "generate the feature map for texture on HH channel" << std::endl;
		for (int i = 0; i < 8; i++) { cols.push_back(i); png_name.push_back("GLCM engergy bin " + std::to_string(i + 1)); }
		for (int i = 8; i < 16; i++) { cols.push_back(i); png_name.push_back("GLCM Contrast bin " + std::to_string(i-8 + 1)); }
		for (int i = 16; i < 24; i++) { cols.push_back(i); png_name.push_back("GLCM Homogenity bin " + std::to_string(i - 16 + 1)); }
		for (int i = 24; i < 32; i++) { cols.push_back(i); png_name.push_back("GLCM Entropy bin " + std::to_string(i - 24 + 1)); }
		for (int i = 32; i < 64; i++) { cols.push_back(i); png_name.push_back("LBP bin " + std::to_string(i - 32 + 1)); }
	}
	else if (this->featureName == "mp") {
		std::cout << "generate the feature map for mp features on HH channel" << std::endl;
		png_name = { "opening","opening_by_construction","closing","closing_by_reconstruction" };
		cv::Mat hh;
		hdf5::readData(hdf5_file, "/masks", "/hh_intensity", hh);
		std::vector<cv::Mat> temp = morph::CaculateMP(hh, 3);

		for (int i = 0; i < temp.size(); i++) {
			temp[i].convertTo(temp[i], CV_8UC1);
			cv::equalizeHist(temp[i], temp[i]);
			cv::applyColorMap(temp[i], temp[i], cv::COLORMAP_JET);
			std::string outputpng = outputpath + "\\" + png_name[i] + ".png";
			std::cout << "generate " << outputpng << std::endl;
			cv::imwrite(outputpng, temp[i]);
		}
	}

	if (this->featureName != "mp") {
		std::vector<cv::Mat> featureMap(png_name.size());

		cv::Mat labelMap;
		hdf5::readData(hdf5_file, "/masks", dataset_name[0], labelMap);
		for (auto& f : featureMap) { f = cv::Mat(cv::Size(labelMap.size()), CV_32FC1); }

		int totalrows = hdf5::getRowSize(hdf5_file, parent, "/feature");
		if (totalrows > 0) {
			int offset_row = 0;
			int partSize;
			if (batch_Size > totalrows) { batch_Size = totalrows; }
			int partsCount = totalrows / batch_Size;
			for (int i = 0; i < partsCount; ++i) {
				partSize = totalrows / (partsCount - i);
				totalrows -= partSize;
				if (totalrows < 0) { break; }
				cv::Mat pts, feature;
				hdf5::readData(hdf5_file, parent, dataset_name[1], pts, offset_row, partSize);
				hdf5::readData(hdf5_file, parent, dataset_name[2], feature, offset_row, partSize);
				feature.convertTo(feature, CV_32FC1);

				for (int j = 0; j < feature.rows; j++) {
					int row = pts.at<int>(j, 1);
					int col = pts.at<int>(j, 2);
					for (int k = 0; k < cols.size(); k++) {
						featureMap[k].at<float>(row, col) = feature.at<float>(j, cols[k]);
					}
				}
				offset_row = offset_row + partSize;
			}

			int i = 0;

			for (auto& f : featureMap) {
				//get min and max, stretch it to 0-255
				f.convertTo(f, CV_8UC1);
				cv::equalizeHist(f, f);
				cv::applyColorMap(f, f, cv::COLORMAP_JET);
				std::string outputpng = outputpath + "\\" + png_name[i] + ".png";
				std::cout << "generate " << outputpng << std::endl;
				cv::imwrite(outputpng, f);
				i = i + 1;
			}
		}
		else {
			std::cout << "can't find " << parent + "/feature" << " in " << hdf5_file << std::endl;
		}
	}
}


/*===================================================================
 * Function: featureDimReduction
 *
 * Summary:
 *   reduced the feature dimension by T-SNE
 *	 dump one batch to txt file for plotting
 *	 check the KNN accuracy on reduced feature data
 *
 * Arguments:
 *	int batchID : default 0
 * Returns:
 *   void
=====================================================================
*/
void FeatureProcess::featureDimReduction(int batchID) {

	std::vector<std::string> dataset_name = { "/feature" ,"/groundtruth","/dimReduced_feature" };
	std::string parent = "/" + this->featureName + "_filterSize_" + std::to_string(filter_Size) + "_patchSize_" + std::to_string(patch_Size);
	cv::Mat  feature, groundtruths;
	hdf5::readData(hdf5_file, parent, dataset_name[0], feature, batchID, batch_Size);
	hdf5::readData(hdf5_file, parent, dataset_name[1], groundtruths, batchID, batch_Size);
	std::cout << "get " << feature.rows << " rows for " << this->featureName << " feature" << std::endl;

	cv::Mat reduced_feature = DataProcess::featureDimReduction(feature, 2);

	std::string dim_reduce = "dimReduced_" + this->featureName + ".txt";
	std::cout << "save dimension reduced feature to " << dim_reduce << std::endl;
	std::ofstream fout(dim_reduce);

	std::vector<cv::Mat> newfeatures(reduced_feature.rows);
	std::vector<unsigned char> labels(reduced_feature.rows);
	for (int i = 0; i < reduced_feature.rows; i++) {
		//dump this batch to txt file for ploting
		fout << groundtruths.at<int>(i, 0) << "," << reduced_feature.at<float>(i, 0) << "," << reduced_feature.at<float>(i, 1) << std::endl;

		cv::Mat temp(1, 2, CV_32FC1);
		temp.at<float>(0, 0) = reduced_feature.at<float>(i, 0);
		temp.at<float>(0, 1) = reduced_feature.at<float>(i, 1);
		newfeatures[i] = temp;
		labels[i] = groundtruths.at<int>(i, 0);
	}
	std::vector<unsigned char> results;
	//check the KNN accuracy of dim reduced features
	DataProcess::applyML(newfeatures, labels, 80, "opencvKNN", results,10);
	std::map<unsigned char, std::string> className = this->getClassName(hdf5_file);
	DataProcess::calculatePredictionAccuracy("", results, labels, className);
}


/*===================================================================
 * Function: classifyFeaturesML
 *
 * Summary:
 *   Read the features from hdf5 file, classify them and write the classifiy results into hdf5 file
 *
 * Arguments:
 *   std::string & classifier_type - choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
 *	 int trainPercent
 *   int K : default 10
 * Returns:
 *   void
=====================================================================
*/
void FeatureProcess::classifyFeaturesML(const std::string classifier_type, int trainPercent, int K) {
	std::string parent = "/" + this->featureName + "_filterSize_" + std::to_string(this->filter_Size) + "_patchSize_" + std::to_string(this->patch_Size);
	std::vector<std::string> dataset_name = { "/feature" ,"/groundtruth" };

	std::map<unsigned char, std::string> className = this->getClassName(this->hdf5_file);

	if (hdf5::checkExist(this->hdf5_file, parent, "/" + classifier_type)) {
		hdf5::deleteData(this->hdf5_file, parent, "/" + classifier_type);
	}

	int fullSize = hdf5::getRowSize(this->hdf5_file, parent, dataset_name[0]);
	std::cout << "get " << fullSize << " rows for " << this->featureName << " feature from hdf5 file with filterSize " << this->filter_Size << " , patchSize " << this->patch_Size << std::endl;

	int offset_row = 0;
	int partSize;
	if (this->batch_Size > fullSize) { this->batch_Size = fullSize; }
	int partsCount = fullSize / this->batch_Size;
	if (fullSize != 0) {
		for (int i = 0; i < partsCount; ++i) {
			partSize = fullSize / (partsCount - i);
			fullSize -= partSize;
			if (fullSize < 0) { break; }
			std::vector<cv::Mat> features;
			std::vector<cv::Point> labelPoints;
			std::vector<unsigned char> labels;
			this->getFeaturesFromHDF(this->hdf5_file, parent, features, labels, labelPoints, offset_row, partSize);
			std::cout << "get " << features.size() << " rows for " << this->featureName << " feature from hdf5 file with filterSize " << this->filter_Size << " , patchSize " << this->patch_Size << std::endl;

			std::vector<unsigned char> class_results;
			DataProcess::applyML(features, labels, trainPercent, classifier_type, class_results,K);
			DataProcess::calculatePredictionAccuracy("", class_results, labels, className);
			saveClassResultToHDF(this->hdf5_file, parent, classifier_type, class_results, labelPoints);
			offset_row = offset_row + partSize;
			std::cout << "classifiy " << this->featureName << " progress: " << float(i + 1) / float(partsCount) * 100.0 << "% \n" << std::endl;
		}
	}
	else {
		std::cout << this->featureName << " with filterSize " << this->filter_Size << " , patchSize " << this->patch_Size << " is not existed in hdf5 file " << std::endl;
	}
}


/*===================================================================
 * Function: LoadSamplePoints
 *
 * Summary:
 *   load sample points
 *
 * Arguments:
 *   const int& patchSize 
 *	 const int& numOfSamplePoint - the number of samples for one type of class, 0 means load all the possible sample points
 *	 int stride
 *   const unsigned char& classlabel - choose which class to load, 255 means to load all the classes
 * output:
 *
=====================================================================
*/
void FeatureProcess::LoadSamplePoints(std::vector<cv::Point>& samplePoints, std::vector<unsigned char>& sampleLabel, const cv::Mat& LabelMap, const std::map<unsigned char, std::string>& classNames, const int& patchSize, const int& numOfSamplePoint, int stride, const unsigned char& classlabel) {
	
	if (!samplePoints.empty()) {
		samplePoints.clear();
	}

	std::cout << "start to generate sample points with patchSize " << patchSize << ", stride " << stride << "..." << std::endl;

	for (const auto& classname : classNames) {
		unsigned char label = unsigned char(0);
		std::string name = classname.second;
		std::vector<cv::Point> pts;
		if (classlabel == unsigned char(255)) {
			label = classname.first;
		}
		else if (classlabel == classname.first) {
			label = classlabel;
		}
		else {
			continue;
		}

		DataProcess::getRandomSamplePoint(LabelMap, pts, label, patchSize, stride, numOfSamplePoint);
		std::cout << "Get " << pts.size() << " sample points for class " << name << std::endl;
		for (size_t i = 0; i < pts.size(); i++) {
			sampleLabel.push_back(label);
		}
		copy(pts.begin(), pts.end(), back_inserter(samplePoints));
		pts.clear();
	}
}

/*===================================================================
 * Function: getSample
 *
 * Summary:
 *   get sample mat centered at sample point
 *   apply refined Lee filter to the sample mat if filtersize is not 0
 *
 * Arguments:
 *  const std::vector<cv::Mat>& data : complex mat with values [HH, VV, HV]
 *  const Point& p - sample point
 *	int patchSize
 *	int filtersize - choose from {0, 5, 7, 9, 11}
 * output:
 * Mat& hh, Mat& vv, Mat& hv
=====================================================================
*/
void FeatureProcess::getSample(const std::vector<cv::Mat>& data, const cv::Point& p, int patchSize, int filtersize, cv::Mat& hh, cv::Mat& vv, cv::Mat& hv) {
	int size = patchSize;
	int start_x = int(p.x) - patchSize / 2;
	int start_y = int(p.y) - patchSize / 2;
	cv::Rect roi = cv::Rect(start_x, start_y, size, size);

	//boundary check
	//check if the sample corners are on the border
	int x_min = p.x - int(patchSize / 2); // (x,y) -> (col,row)
	int x_max = p.x + int(patchSize / 2);
	int y_min = p.y - int(patchSize / 2);
	int y_max = p.y + int(patchSize / 2);
	if (x_max < data[0].cols && y_max < data[0].rows && y_min >= 0 && x_min >= 0) {
		if (data.size() == 3) {
			if (filtersize == 5 || filtersize == 7 || filtersize == 9 || filtersize == 11) {
				hh = data[0](roi).clone();
				vv = data[1](roi).clone();
				hv = data[2](roi).clone();

				RefinedLee* filter = new RefinedLee(filtersize, 1);
				filter->filterFullPol(hh, vv, hv);
				delete filter;
			}
			else {
				hh = data[0](roi);
				vv = data[1](roi);
				hv = data[2](roi);
			}
		}
		else if (data.size() == 2) {
			vv = data[0](roi);
			hv = data[1](roi);
		}
	}
	else {
		std::cout << "out of boundary, get sample at point (" << p.x << "," << p.y << "with patchSize " << patchSize << " failed " << std::endl;
		hh = cv::Mat();
		vv = cv::Mat();
		hv = cv::Mat();
	}
}

// calculate texture features on each channel
cv::Mat FeatureProcess::caculTexture(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv) {
	std::vector<cv::Mat> temp(3);
	// intensity of HH channel
	temp[0] = polsar::logTransform(polsar::getComplexAmpl(hh));

	// intensity of VV channel
	temp[1] = polsar::logTransform(polsar::getComplexAmpl(vv));

	// intensity of HV channel
	temp[2] = polsar::logTransform(polsar::getComplexAmpl(hv));

	std::vector<cv::Mat> output;
	for (const auto& t : temp) {
		cv::Mat result;
		hconcat(cvFeatures::GetGLCM(t, 8, GrayLevel::GRAY_8, 32), cvFeatures::GetLBP(t, 1, 8, 32), result);
		output.push_back(result);
	}

	cv::Mat result;
	vconcat(output, result);
	return result.reshape(1, 1);
}

// calculate color features on Pauli Color Coding
cv::Mat FeatureProcess::caculColor(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv) {
	cv::Mat colorImg = polsar::GetPauliColorImg(hh, vv, hv);
	cv::Mat result;
	cv::hconcat(cvFeatures::GetMPEG7CSD(colorImg, 32), cvFeatures::GetMPEG7DCD(colorImg, 1), result);
	return result;
}

// calculate morphological profile on each channel with diameter (1,3,5)
cv::Mat FeatureProcess::caculMP(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv) {
	cv::Mat hh_log = polsar::logTransform(polsar::getComplexAmpl(hh));
	cv::Mat vv_log = polsar::logTransform(polsar::getComplexAmpl(vv));
	cv::Mat hv_log = polsar::logTransform(polsar::getComplexAmpl(hv));
	cv::Mat result;
	result.push_back(cvFeatures::GetMP(hh_log, { 1,3,5 }));
	result.push_back(cvFeatures::GetMP(vv_log, { 1,3,5 }));
	result.push_back(cvFeatures::GetMP(hv_log, { 1,3,5 }));

	return result.reshape(1, 1);
}

// calculate target decomposition
cv::Mat FeatureProcess::caculDecomp(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv) {
	cv::Mat result;

	std::vector<cv::Mat> pauli;
	std::vector<cv::Mat> circ;
	std::vector<cv::Mat> lexi;
	polsar::getPauliBasis(hh, vv, hv, pauli);
	polsar::getCircBasis(hh, vv, hv, circ);
	polsar::getLexiBasis(hh, vv, hv, lexi);
	std::vector<cv::Mat> covariance;
	std::vector<cv::Mat> coherency;
	polsar::GetCoherencyT(pauli, coherency);
	polsar::GetCovarianceC(lexi, covariance);

	std::vector<cv::Mat> decomposition;
	polsar::GetCloudePottierDecomp(coherency, decomposition); //3  
	polsar::GetFreemanDurdenDecomp(covariance, decomposition); //3  
	polsar::GetKrogagerDecomp(circ, decomposition); // 3  
	polsar::GetPauliDecomp(pauli, decomposition); // 3  
	vconcat(decomposition, result);
	return result.reshape(1, 1);
}


// get polsar features on elements of covariance matrix C and coherency matrix T
cv::Mat FeatureProcess::caculCTelements(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv) {
	cv::Mat result;
	std::vector<cv::Mat> temp;
	polsar::GetCTelements(hh, vv, hv, temp);

	for (auto& d : temp) {
		result.push_back(d);
	}

	return result.reshape(1, 1);

}


// get polsar features on statistic of polsar parameters
cv::Mat FeatureProcess::caculPolStatistic(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv) {

	std::vector<cv::Mat> temp;
	// intensity of HH channel
	cv::Mat hh_log = polsar::logTransform(polsar::getComplexAmpl(hh));

	// intensity of VV channel
	cv::Mat vv_log = polsar::logTransform(polsar::getComplexAmpl(vv));

	// intensity of HV channel
	cv::Mat hv_log = polsar::logTransform(polsar::getComplexAmpl(hv));

	// phase difference HH-VV
	cv::Mat phaseDiff = polsar::getPhaseDiff(hh, vv);

	//statistic of Co-polarize ratio VV-HH
	cv::Mat coPolarize = vv_log - hh_log;

	// Cross-polarized ratio HV-HH
	cv::Mat crossPolarize = hv_log - hh_log;

	// polarized ratio HV-VV
	cv::Mat otherPolarize = hv_log - vv_log;

	//Copolarization ratio
	cv::Mat copolarizationRatio = polsar::getCoPolarizationRatio(hh, vv, 3);

	//deCopolarization ratio
	cv::Mat deCopolarizationRatio = polsar::getDePolarizationRatio(hh, vv, hv, 3);

	// amplitude of HH-VV correlation
	cv::Mat amplitudeCorrelation = polsar::logTransform(polsar::calcuCoherenceOfPol(hh, vv, 3)) - hh_log - vv_log;

	temp.push_back(hh_log);
	temp.push_back(vv_log);
	temp.push_back(hv_log);
	temp.push_back(phaseDiff);
	temp.push_back(coPolarize);
	temp.push_back(crossPolarize);
	temp.push_back(otherPolarize);
	temp.push_back(copolarizationRatio);
	temp.push_back(deCopolarizationRatio);
	temp.push_back(amplitudeCorrelation);

	std::vector<cv::Mat> statistic;
	for (const auto& t : temp) {
		statistic.push_back(cvFeatures::GetStatistic(t));
	}

	cv::Mat output;
	cv::hconcat(statistic, output);
	return output.reshape(1, 1);
}


void FeatureProcess::writeLabelMapToHDF(const std::string& hdf5_fileName, const std::vector<cv::Mat>& data, const cv::Mat& labelmap, const std::map<unsigned char, std::string>& classnames) {

	std::string parent_name = "/masks";

	if (!hdf5::checkExist(hdf5_fileName, parent_name, "/labelMap")) {
		// save labelMap to hdf5
		hdf5::writeData(hdf5_fileName, parent_name, "/labelMap", labelmap);
		std::cout << "write labelMap to hdf5 success " << std::endl;

		// save the class name to hdf5
		cv::Mat classlabels;
		for (auto& name : classnames) {
			classlabels.push_back(name.first);
			hdf5::writeAttr(hdf5_fileName, std::to_string(name.first), name.second);
		}
		hdf5::writeAttr(hdf5_fileName, "classlabels", classlabels.reshape(1, 1));
	}

	// save the intensity of HH,VV,HV to hdf5
	if (!hdf5::checkExist(hdf5_fileName, parent_name, "/hh_intensity")) {
		std::cout << "write the intensity of HH, VV, HV to hdf5 success " << std::endl;
		cv::Mat hh = polsar::logTransform(data[0]);
		cv::Mat vv = polsar::logTransform(data[1]);
		cv::Mat hv = polsar::logTransform(data[2]);
		hdf5::writeData(hdf5_fileName, parent_name, "/hh_intensity", hh);
		hdf5::writeData(hdf5_fileName, parent_name, "/vv_intensity", vv);
		hdf5::writeData(hdf5_fileName, parent_name, "/hv_intensity", hv);
	}
}



void FeatureProcess::getSampleInfo(const std::string& hdf5_fileName, const cv::Mat& pts, int patchSize) {
	std::cout << "it has " << pts.rows << " samples with patchSize " << patchSize << std::endl;
	std::map<int, int> count;
	for (int row = 0; row < pts.rows; row++) {
		int label = pts.at<int>(row, 0);
		count[label]++;
	}
	std::cout << "class name (unknown, 0) means this patch cross class boarder or unclassified" << std::endl;
	for (auto const& c : count)
	{
		int label = c.first;
		int sampleNum = c.second;
		std::string class_name;
		hdf5::readAttr(hdf5_fileName, "label_" + std::to_string(label), class_name);
		std::cout << class_name << " : " << std::to_string(label) << " : number of samples: " << sampleNum << std::endl;
	}
}


std::map<unsigned char, std::string> FeatureProcess::getClassName(const std::string& filename) {
	// get the class names from hdf5
	cv::Mat classlabels;
	hdf5::readAttr(filename, "classlabels", classlabels);
	std::map<unsigned char, std::string> className;
	for (int i = 0; i < classlabels.cols; i++) {
		unsigned char label = classlabels.at<unsigned char>(0, i);
		std::string class_name;
		hdf5::readAttr(filename, std::to_string(label), class_name);
		if (!class_name.empty()) { className[label] = class_name; }
	}
	return className;
}

/*
************************************************************************
input: the label
return: the color
************************************************************************
*/
cv::Vec3b FeatureProcess::getLabelColor(unsigned char class_result)
{
	cv::Vec3b labelColor;

	// Color is BGR not RGB!
	cv::Vec3b black = cv::Vec3b(0, 0, 0);// unclassified, class 0

	cv::Vec3b red = cv::Vec3b(49, 60, 224); //city, class 1

	cv::Vec3b yellow = cv::Vec3b(0, 190, 246); //field, class 2

	cv::Vec3b dark_green = cv::Vec3b(66, 121, 79); //forest, class 3

	cv::Vec3b light_green = cv::Vec3b(0, 189, 181); // grassland, class 4

	cv::Vec3b blue = cv::Vec3b(164, 85, 50); //street, class 5

	std::vector<cv::Vec3b> right_color = { black, red,  yellow, dark_green, light_green,blue, };

	labelColor = right_color[int(class_result)];

	return labelColor;
}


/*===================================================================
 * Function: saveClassResultToHDF
 *
 * Summary:
 *   write back the classified results to hdf5 ( class result from classifier,sample points)
 *
 * Arguments:
 *   std::string& hdf5_fileName - hdf5 filename
 *   std::string& parent_name
 *   std::string & classResult_name - choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
 *	 std::vector<unsigned char>& class_result
 *	 std::vector<cv::Point> & points
 * output:
 *
=====================================================================
*/
void FeatureProcess::saveClassResultToHDF(const std::string& hdf5_fileName, const std::string& parent_name, const std::string& classResult_name, const std::vector<unsigned char>& class_result, const std::vector<cv::Point>& points) {
	cv::Mat pts = cv::Mat(points.size(), 3, CV_32SC1);
	for (size_t i = 0; i < points.size(); ++i) {
		pts.at<int>(i, 0) = (int)(class_result[i]);
		pts.at<int>(i, 1) = points[i].y; //row
		pts.at<int>(i, 2) = points[i].x; //col
	}
	hdf5::insertData(hdf5_fileName, parent_name, "/" + classResult_name, pts);
}


/*===================================================================
 * Function: getFeaturesFromHDF
 *
 * Summary:
 *   get features data and its groundtruth from hdf5
 *
 * Arguments:
 *   std::string& hdf5_fileName - hdf5 filename
 *   std::string& parent_name
 *	 int batchSize
 *	 int offset_row - the start row
 *   int counts_rows - the number of samples for output
 * output:
 *	 std::vector<cv::Mat>& features
 *   std::vector<unsigned char>& featureLabels
 *	 std::vector<cv::Point> & labelPoints
=====================================================================
*/
void FeatureProcess::getFeaturesFromHDF(const std::string& hdf5_fileName, const std::string& parent_name, std::vector<cv::Mat>& features, std::vector<unsigned char>& featureLabels, std::vector<cv::Point>& labelPoints, int offset_row, int counts_rows) {
	std::vector<std::string> dataset_name = { "/feature" ,"/groundtruth" };
	cv::Mat feature, pts;
	if (hdf5::checkExist(hdf5_fileName, parent_name, dataset_name[0]) &&
		hdf5::checkExist(hdf5_fileName, parent_name, dataset_name[1])) {
		hdf5::readData(hdf5_fileName, parent_name, dataset_name[0], feature, offset_row, counts_rows);
		hdf5::readData(hdf5_fileName, parent_name, dataset_name[1], pts, offset_row, counts_rows);

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