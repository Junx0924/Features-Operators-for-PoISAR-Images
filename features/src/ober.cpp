#include "ober.hpp" 
#include "cvFeatures.hpp"
#include "sarFeatures.hpp"
#include <opencv2/opencv.hpp>

#ifdef VC
#include <filesystem>
#endif // VC

#ifdef GCC
#include <dirent.h>
#endif

#include <complex>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void ober::LoadSamplePoints(const int &size, const int &num) {

	samplePointNum = num;
	sampleSize = size;

	samplePoints->reserve(static_cast<int>(labels.size()) * samplePointNum); // pre-allocate memory
	samplePointClassLabel->reserve(static_cast<int>(labels.size()) * samplePointNum); // pre-allocate memory

	// get the sample points in each mask area
	for (int i = 0; i < masks.size(); i++) {
		vector<Point> pts;
		Utils::getSafeSamplePoints(masks[i], samplePointNum, sampleSize, pts);
		cout << "Get " << pts.size() << " sample points for class " << classNames[labels[i]] << endl;
		// store the sample points
		samplePoints->insert(samplePoints->end(), pts.begin(), pts.end());
		// store the sample points label
		for (int j = 0; j < pts.size(); j++) { samplePointClassLabel->push_back(labels[i]); }
	}
}

// get patches of 3 channel (HH+VV,HV,HH-VV) intensity(dB)
void ober::GetPauliColorPatches(vector<Mat>& patches, vector<unsigned char>& classValue) {
	
	if (!samplePoints->empty()) {
			for (const auto& p : *samplePoints) {
				Mat hh, vv, hv;
				getSample(p, hh, vv, hv);

				patches.push_back(polsar::GetPauliColorImg(hh, vv, hv));
		}
			classValue = *samplePointClassLabel;
	}
}


// get patches of 3 channel (HH,VV,HV) intensity(dB)
void ober::GetPatches(vector<Mat>& patches, vector<unsigned char>& classValue) {
		 
		if(!samplePoints->empty()){
				for (const auto& p : *samplePoints) {
					Mat hh, vv, hv;
					getSample(p, hh, vv, hv);

					patches.push_back(polsar::GetFalseColorImg(hh, vv, hv));
				}
				classValue = *samplePointClassLabel;
		}
}


// get texture features(LBP and GLCM) on HH,VV,VH, default feature mat size 3*64
void ober::GetTextureFeature(vector<Mat>& features, vector<unsigned char>& classValue) {

		std::map< unsigned char, int> count;
		cout << "start to draw texture features ... " <<endl;

		for (int i = 0; i < samplePoints->size();i++) {
			Point p = samplePoints->at(i);

			cout << classNames[samplePointClassLabel->at(i)] << " :draw texture feature at Point (" << p.x << ", " << p.y << ")" << endl;

			Mat hh, vv, hv;
			getSample(p, hh, vv, hv);

			vector<Mat> temp;
			// intensity of HH channel
			  hh = polsar::logTransform(polsar::getComplexAmpl(hh));
			// intensity of VV channel
			  vv = polsar::logTransform(polsar::getComplexAmpl(vv));
			// intensity of HV channel
			  hv = polsar::logTransform(polsar::getComplexAmpl(hv));

			temp.push_back(hh);
			temp.push_back(vv);
			temp.push_back(hv);

			Mat result;
			for (const auto& t : temp) {
				Mat temp;
				cv::hconcat(cvFeatures::GetGLCM(t, 8, GrayLevel::GRAY_8, 32), cvFeatures::GetLBP(t, 1, 8, 32), temp);
				result.push_back(temp);
			}
			features.push_back(result);
			classValue.push_back(samplePointClassLabel->at(i));

			count[samplePointClassLabel->at(i)]++;
		}

		for (auto it = count.begin(); it != count.end(); ++it)
		{
			std::cout << "get "<< it->second  <<" texture features for class " <<  classNames[it->first] << std::endl;
		}
}



// get color features(MPEG-7 DCD,CSD) on Pauli Color image, default feature mat size 1*44
void ober::GetColorFeature(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw color features ... " << endl;

	for (int i = 0; i < samplePoints->size(); i++) {
		Point p = samplePoints->at(i);

		cout << classNames[samplePointClassLabel->at(i)] << " :draw color feature at Point (" << p.y << ", " << p.x << ")" << endl;
		
		Mat hh, vv, hv;
		getSample(p, hh, vv, hv);

		Mat colorImg = polsar::GetPauliColorImg(hh, vv, hv);

		Mat result;
		cv::hconcat(cvFeatures::GetMPEG7CSD(colorImg, 32), cvFeatures::GetMPEG7DCD(colorImg, 3), result);
		features.push_back(result);
		classValue.push_back(samplePointClassLabel->at(i));

		count[samplePointClassLabel->at(i)]++;
	}

	for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " color features for class " << classNames[it->first] << std::endl;
	}
}


// get MP features on HH,VV,VH, default feature mat size (sampleSize*9,sampleSize)
void ober::GetMPFeature(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw MP features ... " << endl;

	for (int i = 0; i < samplePoints->size(); i++) {
		Point p = samplePoints->at(i);

		cout << classNames[samplePointClassLabel->at(i)] << " :draw MP feature at Point (" << p.y << ", " << p.x << ")" << endl;

		Mat hh, vv, hv;
		getSample(p, hh, vv, hv);

		vector<Mat> temp;
		// intensity of HH channel
		hh = polsar::logTransform(polsar::getComplexAmpl(hh));
		// intensity of VV channel
		vv = polsar::logTransform(polsar::getComplexAmpl(vv));
		// intensity of HV channel
		hv = polsar::logTransform(polsar::getComplexAmpl(hv));

		temp.push_back(hh);
		temp.push_back(vv);
		temp.push_back(hv);

		Mat result;
		for (const auto& t : temp) {
			result.push_back(cvFeatures::GetMP(t, { 1,2,3 }));
		}
		features.push_back(result);
		classValue.push_back(samplePointClassLabel->at(i));

		count[samplePointClassLabel->at(i)]++;
	}

	for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " mp features for class " << classNames[it->first] << std::endl;
	}
}


// get polsar features on target decompostion 
void ober::GetDecompFeatures(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw target decomposition features of polsar parameters ... " << endl;

	for (int i = 0; i < samplePoints->size(); i++) {
		Point p = samplePoints->at(i);

		cout << classNames[samplePointClassLabel->at(i)] << " :draw target decomposition feature at Point (" << p.y << ", " << p.x << ")" << endl;

		Mat hh, vv, hv;
		getSample(p, hh, vv, hv);

		//caculate the decomposition at sample point
		Mat result;
		vector<Mat> decomposition;
		getTargetDecomposition(hh, vv, hv, decomposition);

		for (auto& d : decomposition) {
			result.push_back(d);
		}

		features.push_back(result);
		classValue.push_back(samplePointClassLabel->at(i));

		count[samplePointClassLabel->at(i)]++;
	}

	for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " target decomposition features for class " << classNames[it->first] << std::endl;
	}
}

// get polsar features on elements of covariance matrix C and coherency matrix T
void ober::GetCTFeatures(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw matrix C and T elements ... " << endl;

	for (int i = 0; i < samplePoints->size(); i++) {
		Point p = samplePoints->at(i);

		cout << classNames[samplePointClassLabel->at(i)] << " :draw matrix C and T elements at Point (" << p.y << ", " << p.x << ")" << endl;

		Mat hh, vv, hv;
		getSample(p, hh, vv, hv);

		//caculate the decomposition at sample point
		Mat result;
		vector<Mat> decomposition;
		getCTelements(hh, vv, hv, decomposition);

		for (auto& d : decomposition) {
			result.push_back(d);
		}

		features.push_back(result);
		classValue.push_back(samplePointClassLabel->at(i));

		count[samplePointClassLabel->at(i)]++;
	}

	for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " CT elements features for class " << classNames[it->first] << std::endl;
	}
}

// get polsar features on statistic of polsar parameters
void ober::GetPolsarStatistic(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw statistic features of polsar parameters ... " << endl;

	for (int i = 0; i < samplePoints->size(); i++) {
		Point p = samplePoints->at(i);

		cout << classNames[samplePointClassLabel->at(i)] << " :draw statistic polarimetric feature at Point (" << p.y << ", " << p.x << ")" << endl;

		Mat hh, vv, hv;
		getSample(p, hh, vv, hv);

		Mat result1;
		vector<Mat> statistic;
		getStatisticFeature(hh, vv, hv, statistic);
		cv::hconcat(statistic, result1);

		 
		features.push_back(result1);
		classValue.push_back(samplePointClassLabel->at(i));

		count[samplePointClassLabel->at(i)]++;
	}

	for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " statistic polarimetric features for class " << classNames[it->first] << std::endl;
	}
}



// get statistical (min,max,mean,median,std) on polsar parameters
// vector<mat>& result - vector length : 7, mat size: 1*5
void ober::getStatisticFeature(const Mat& hh, const Mat& vv, const Mat hv, vector<Mat>& result) {

	vector<Mat> temp;
	// intensity of HH channel
	Mat hh_log = polsar::logTransform(polsar::getComplexAmpl(hh));
	// intensity of VV channel
	Mat vv_log = polsar::logTransform(polsar::getComplexAmpl(vv));
	// intensity of HV channel
	Mat hv_log = polsar::logTransform(polsar::getComplexAmpl(hv));
	// phase difference HH-VV
	Mat phaseDiff = polsar::getPhaseDiff(hh, vv);
	//statistic of Co-polarize ratio VV-HH
	Mat coPolarize = vv_log - hh_log;
	// Cross-polarized ratio HV-HH
	Mat crossPolarize = hv_log - hh_log;
	// polarized ratio HV-VV
	Mat otherPolarize = hv_log - vv_log;


	temp.push_back(hh_log);
	temp.push_back(vv_log);
	temp.push_back(hv_log);
	temp.push_back(phaseDiff);
	temp.push_back(coPolarize);
	temp.push_back(crossPolarize);
	temp.push_back(otherPolarize);

	for (const auto& t : temp) {
		result.push_back(cvFeatures::GetStatistic(t));
	}
}

// get upper triangle matrix elements of C, T
// vector<mat>& result - vector length: 12, mat size: (hh.size())
void ober::getCTelements(const Mat& hh, const Mat& vv, const Mat hv, vector<Mat> & result) {
	vector<Mat> pauli;
	vector<Mat> circ;
	vector<Mat> lexi;
	polsar::getPauliBasis(hh, vv, hv, pauli);
	polsar::getCircBasis(hh, vv, hv, circ);
	polsar::getLexiBasis(hh, vv, hv, lexi);
	vector<Mat> covariance;
	vector<Mat> coherency;
	polsar::GetCoherencyT(pauli, coherency);
	polsar::GetCovarianceC(lexi, covariance);

	// upper triangle matrix elements of covariance matrix C and coherency matrix T
	std::array<int, 6> ind = { 0,1,2,4,5,8 };
	for (auto& i : ind) {
		result.push_back(polsar::logTransform(covariance[i]));  // 6
		result.push_back(polsar::logTransform(coherency[i]));   // 6
	}
}

// get target decomposition features
// vector<mat>& result - vector length: , mat size: (hh.size())
void ober::getTargetDecomposition(const Mat& hh, const Mat& vv, const Mat hv, vector<Mat>& result) {
	vector<Mat> pauli;
	vector<Mat> circ;
	vector<Mat> lexi;
	polsar::getPauliBasis(hh, vv, hv, pauli);
	polsar::getCircBasis(hh, vv, hv, circ);
	polsar::getLexiBasis(hh, vv, hv, lexi);
	vector<Mat> covariance;
	vector<Mat> coherency;
	polsar::GetCoherencyT(pauli, coherency);
	polsar::GetCovarianceC(lexi, covariance);

	//polsar::GetCloudePottierDecomp(coherency, result); //3  
	polsar::GetFreemanDurdenDecomp(coherency, result); //3  
	//polsar::GetKrogagerDecomp(circ, result); // 3  
	//polsar::GetPauliDecomp(pauli, result); // 3  
	//polsar::GetHuynenDecomp(covariance, result); // 9  
	//polsar::GetYamaguchi4Decomp(coherency, covariance, result); //4  
}

// set despeckling filter size, choose from ( 5, 7, 9, 11)
void ober::SetFilterSize(int filter_size) { filterSize = filter_size; }

// get data at sample point
void ober::getSample(const Point& sample_point, Mat& hh, Mat& vv, Mat& hv) {
	int start_x = int(sample_point.x) - sampleSize / 2;
	int start_y = int(sample_point.y) - sampleSize / 2;
	Rect roi = Rect(start_x, start_y, sampleSize, sampleSize);
	if (filterSize ==5 || filterSize ==7 || filterSize ==9 || filterSize ==11) {
		hh = data[0](roi).clone();
		vv = data[1](roi).clone();
		hv = data[2](roi).clone();
		RefinedLee* filter = new RefinedLee(filterSize, 1);
		filter->filterFullPol(hh, vv, hv);
		delete filter;
	}
	else {
		hh = data[0](roi);
		vv = data[1](roi);
		hv = data[2](roi);
	}
}



#ifdef VC
void ober::loadData(string RATfolderPath) {
    vector<string> fnames;
    fnames.reserve(5);

    fs::recursive_directory_iterator iter(RATfolderPath);
    fs::recursive_directory_iterator end;
	while (iter != end) {
		string tmp = iter->path().string();
		

		fnames.push_back(tmp);
		error_code ec;
		iter.increment(ec);
		if (ec) {
			std::cerr << "Error while accessing:" << iter->path().string() << "::" << ec.message() << "\n";
		}
	}

    switch (fnames.size()) {
	case 1: {        // one rat file with scattering vector or matrix
		loadRAT(fnames[0], this->data);
		break;
	}
	case 2: {        // dual-pol, one file per channel
		vector<Mat> ch1, ch2;
		loadRAT(fnames[0], ch1);
		loadRAT(fnames[1], ch2);
		this->data.push_back(ch1[0]);
		this->data.push_back(ch2[0]);
		break;
	}
	case 3: {        // full-pol, averaged cross-pol, one file per channel
		vector<Mat> hh, vv, xx;
		loadRAT(fnames[0], hh);
		loadRAT(fnames[1], vv);
		loadRAT(fnames[2], xx);
		this->data.push_back(hh[0]);
		this->data.push_back(vv[0]);
		this->data.push_back(xx[0]);
		break;
	}
	case 4: {        // full-pol, individual cross-pol, one file per channel
		vector<Mat> hh, vv, hv, vh;
		loadRAT(fnames[0], hh);
		loadRAT(fnames[1], vv);
		loadRAT(fnames[2], hv);
		loadRAT(fnames[2], vh);
		this->data.push_back(hh[0]);
		this->data.push_back(vv[0]);
		this->data.push_back(0.5 * (hv[0] + vh[0]));
		break;
	}
    }
}


#endif

/**************************************************************
Function to load Oberpfaffenhofen PolSAR data file (RAT format)
***************************************************************/
Size ober::loadRAT2(string fname, vector<Mat>& data, bool metaOnly) {

	bool verbose = false;

	// header info
	int magiclong;
	float version;
	int ndim;
	int nchannel;
	int dim[8];
	int var;
	int sub[2];
	int type;
	int reserved[9];

	// open file
	fstream file(fname.c_str(), ios::in | ios::binary);
	if (!file)
		cerr << "ERROR: Cannot open file: " << fname << endl;

	// read header
	file.read((char*)(&magiclong), sizeof(magiclong));
	file.read((char*)(&version), sizeof(version));
	file.read((char*)(&ndim), sizeof(ndim));
	file.read((char*)(&nchannel), sizeof(nchannel));
	file.read((char*)(&dim), sizeof(dim));
	file.read((char*)(&var), sizeof(var));
	file.read((char*)(&sub), sizeof(sub));
	file.read((char*)(&type), sizeof(type));
	file.read((char*)(&reserved), sizeof(reserved));

	if (verbose) {
		cout << "Number of image dimensions:\t" << ndim << endl;
		cout << "Image dimensions:\t";
		for (int i = 0; i < ndim - 1; i++)
			cout << dim[i] << " x ";
		cout << dim[ndim - 1] << endl;
		cout << "Data type:\t" << var << endl;
		cout << "Type:\t" << type << endl;
	}

	if (metaOnly) {
		file.close();
		return Size(dim[ndim - 2], dim[ndim - 1]);
	}

	vector<unsigned> tileSize(2);
	unsigned tile;
	vector<unsigned> tileStart(2);
	this->getTileInfo(Size(dim[ndim - 2], dim[ndim - 1]), this->border, tile, tileSize, tileStart);

	if (verbose) {
		cout << "Tile:\t\t" << tile << endl;
		cout << "Tile size (cols x rows):\t" << tileSize[0] << "x" << tileSize[1] << endl;
		cout << "Tile start (col x row):\t" << tileStart[0] << "x" << tileStart[1] << endl;
	}

	file.seekg(0);
	file.seekg(1000);
	int nChannels = 0, dsize = 0;
	switch (var) {
	case 1:
		nChannels = 1;
		dsize = 1;
		break;
	case 2:
		nChannels = 1;
		dsize = 4;
		break;
	case 3:
		nChannels = 1;
		dsize = 4;
		break;
	case 4:
		nChannels = 1;
		dsize = 4;
		break;
	case 5:
		nChannels = 1;
		dsize = 8;
		break;
	case 12:
		nChannels = 1;
		dsize = 4;
		break;
	case 13:
		nChannels = 1;
		dsize = 4;
		break;
	case 14:
		nChannels = 1;
		dsize = 8;
		break;
	case 15:
		nChannels = 1;
		dsize = 8;
		break;
	case 6:
		nChannels = 2;
		dsize = 4;
		break;
	case 9:
		nChannels = 2;
		dsize = 8;
		break;
	default: cerr << "ERROR: arraytyp not recognized (wrong format?)" << endl;
	}

	char* buf = new char(dsize);
	char* swap = new char(dsize);
	int i, j, x, y;
	Mat img, real, imag;
	switch (ndim) {
	case 2:
		data.resize(1);
		if (nChannels == 1)
			data[0] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
		else
			data[0] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
		for (y = 0; y < dim[1]; y++) {
			for (x = 0; x < dim[0]; x++) {
				double realVal, imagVal;
				file.read((char*)(&buf), dsize);
				for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
				switch (var) {
				case 1:
					dsize = 1;
					realVal = *((char*)buf);
					break;	// byte
				case 2:
					dsize = 4;
					realVal = *((int*)buf);
					break;	// int
				case 3:
					dsize = 4;
					realVal = *((long*)buf);
					break;	// long
				case 4:
					dsize = 4;
					realVal = *((float*)buf);
					break;	// float
				case 5:
					dsize = 8;
					realVal = *((double*)buf);
					break;	// double
				case 6:
					dsize = 4;					// complex
					realVal = *((float*)buf);
					file.read((char*)(&buf), dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((float*)buf);
					break;
				case 9:
					dsize = 8;					// dcomplex
					realVal = *((double*)buf);
					file.read((char*)(&buf), dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((double*)buf);
					break;
				case 12:
					dsize = 4;
					realVal = *((unsigned int*)buf);
					break;	// uint
				case 13:
					dsize = 4;
					realVal = *((unsigned long*)buf);
					break;	// ulong
				case 14:
					dsize = 4;
					realVal = *((double*)buf);
					break;	// l64
				case 15:
					dsize = 4;
					realVal = *((double*)buf);
					break;	// ul64
				}
				if ((dim[1] - y - 1 < tileStart[1]) || (dim[1] - y - 1 >= tileStart[1] + tileSize[1])) continue;
				if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
				if (nChannels != 2)
					data[0].at<float>(dim[1] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
				else
					data[0].at<Vec2f>(dim[1] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
			}
		}
		break;
	case 3:
		data.resize(dim[0]);
		for (i = 0; i < dim[0]; i++) {
			if (nChannels == 1)
				data[i] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
			else
				data[i] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
		}
		for (y = 0; y < dim[2]; y++) {
			for (x = 0; x < dim[1]; x++) {
				for (i = 0; i < dim[0]; i++) {
					double realVal, imagVal;
					file.read((char*)(&buf), dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					switch (var) {
					case 1:
						dsize = 1;
						realVal = *((char*)buf);
						break;	// byte
					case 2:
						dsize = 4;
						realVal = *((int*)buf);
						break;	// int
					case 3:
						dsize = 4;
						realVal = *((long*)buf);
						break;	// long
					case 4:
						dsize = 4;
						realVal = *((float*)buf);
						break;	// float
					case 5:
						dsize = 8;
						realVal = *((double*)buf);
						break;	// double
					case 6:
						dsize = 4;					// complex
						realVal = *((float*)buf);
						file.read((char*)(&buf), dsize);
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						imagVal = *((float*)buf);
						break;
					case 9: dsize = 8;					// dcomplex
						realVal = *((double*)buf);
						file.read((char*)(&buf), dsize);
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						imagVal = *((double*)buf);
						break;
					case 12:
						dsize = 4;
						realVal = *((unsigned int*)buf);
						break;	// uint
					case 13:
						dsize = 4;
						realVal = *((unsigned long*)buf);
						break;	// ulong
					case 14:
						dsize = 4;
						realVal = *((double*)buf);
						break;	// l64
					case 15:
						dsize = 4;
						realVal = *((double*)buf);
						break;	// ul64
					}
					if ((dim[2] - y - 1 < tileStart[1]) || (dim[2] - y - 1 >= tileStart[1] + tileSize[1])) continue;
					if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
					if (nChannels != 2)
						data.at(i).at<float>(dim[2] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
					else
						data.at(i).at<Vec2f>(dim[2] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
				}
			}
		}
		break;
	case 4:
		data.resize(dim[0] * dim[1]);
		for (i = 0; i < dim[0]; i++) {
			for (j = 0; j < dim[1]; j++) {
				if (nChannels == 1)
					data[i * dim[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
				else
					data[i * dim[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
			}
		}
		for (y = 0; y < dim[3]; y++) {
			for (x = 0; x < dim[2]; x++) {
				for (j = 0; j < dim[0]; j++) {
					for (i = 0; i < dim[1]; i++) {
						double realVal, imagVal;
						file.read((char*)(&buf), dsize);
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						switch (var) {
						case 1:
							dsize = 1;
							realVal = *((char*)buf);
							break;	// byte
						case 2:
							dsize = 4;
							realVal = *((int*)buf);
							break;	// int
						case 3:
							dsize = 4;
							realVal = *((long*)buf);
							break;	// long
						case 4:
							dsize = 4;
							realVal = *((float*)buf);
							break;	// float
						case 5:
							dsize = 8;
							realVal = *((double*)buf);
							break;	// double
						case 6: dsize = 4;					// complex
							realVal = *((float*)buf);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((float*)buf);
							break;
						case 9: dsize = 8;					// dcomplex
							realVal = *((double*)buf);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((double*)buf);
							break;
						case 12:
							dsize = 4;
							realVal = *((unsigned int*)buf);
							break;	// uint
						case 13:
							dsize = 4;
							realVal = *((unsigned long*)buf);
							break;	// ulong
						case 14:
							dsize = 4;
							realVal = *((double*)buf);
							break;	// l64
						case 15:
							dsize = 4;
							realVal = *((double*)buf);
							break;	// ul64
						}
						if ((dim[3] - y - 1 < tileStart[1]) || (dim[3] - y - 1 >= tileStart[1] + tileSize[1])) continue;
						if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
						if (nChannels != 2)
							data.at(j * dim[1] + i).at<float>(dim[3] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
						else
							data.at(j * dim[1] + i).at<Vec2f>(dim[3] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
					}
				}
			}
		}
		break;
	}
	return Size(dim[ndim - 2], dim[ndim - 1]);
}

/**************************************************************
Function to load Oberpfaffenhofen PolSAR data file (RAT format
***************************************************************/

Size ober::loadRAT(string fname, vector<Mat>& data, bool metaOnly) {

	bool verbose = true;

	// header info
	unsigned int dim;
	vector<unsigned int> imgSize;
	unsigned int var;
	unsigned int type;
	unsigned int dummy;
	char info[80];

	//check if it is rat file
	size_t pos = 0;
	if (fname.find("rat", pos) == std::string::npos) { 
		cout << " this is not rat file" << endl;
		exit(-1); }

	// open file
	fstream file(fname.c_str(), ios::in | ios::binary);
	if (!file) {
		cout << "ERROR: Cannot open file: " << fname << endl;
		exit(-1);
	}

	// read header
	file.read((char*)(&dim), sizeof(dim));
	dim = (dim >> 24) | ((dim << 8) & 0x00FF0000) | ((dim >> 8) & 0x0000FF00) | (dim << 24);

	if (dim > 1000) {
		return loadRAT2(fname, data, metaOnly);
	}

	imgSize.resize(dim);
	for (int i = 0; i < dim; i++) {
		file.read((char*)(&imgSize[i]), sizeof(imgSize[i]));
		imgSize[i] = (imgSize[i] >> 24) | ((imgSize[i] << 8) & 0x00FF0000) | ((imgSize[i] >> 8) & 0x0000FF00) | (imgSize[i] << 24);
	}
	file.read((char*)(&var), sizeof(var));
	var = (var >> 24) | ((var << 8) & 0x00FF0000) | ((var >> 8) & 0x0000FF00) | (var << 24);
	file.read((char*)(&type), sizeof(type));
	type = (type >> 24) | ((type << 8) & 0x00FF0000) | ((type >> 8) & 0x0000FF00) | (type << 24);
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read(info, sizeof(info));

	if (verbose) {
		cout << "Number of image dimensions:\t" << dim << endl;
		cout << "Image dimensions:\t";
		for (int i = 0; i < dim - 1; i++) cout << imgSize[i] << " x ";
		cout << imgSize[dim - 1] << endl;
		cout << "Data type:\t" << var << endl;
		cout << "Type:\t" << type << endl;
		cout << "Info:\t" << info << endl;
	}

	if (metaOnly) {
		file.close();
		return Size(imgSize[dim - 2], imgSize[dim - 1]);
	}

	vector<unsigned> tileSize(2);
	unsigned tile;
	vector<unsigned> tileStart(2);
	this->getTileInfo(Size(imgSize[dim - 2], imgSize[dim - 1]), this->border, tile, tileSize, tileStart);

	if (verbose) {
		cout << "Tile:\t\t" << tile << endl;
		cout << "Tile size (cols x rows):\t" << tileSize[0] << "x" << tileSize[1] << endl;
		cout << "Tile start (col x row):\t" << tileStart[0] << "x" << tileStart[1] << endl;
	}
	int nChannels = 0, dsize = 0;
	switch (var) {
	case 1:
		nChannels = 1;
		dsize = 1;
		break;
	case 2:
		nChannels = 1;
		dsize = 4;
		break;
	case 3:
		nChannels = 1;
		dsize = 4;
		break;
	case 4:
		nChannels = 1;
		dsize = 4;
		break;
	case 5:
		nChannels = 1;
		dsize = 8;
		break;
	case 12:
		nChannels = 1;
		dsize = 4;
		break;
	case 13:
		nChannels = 1;
		dsize = 4;
		break;
	case 14:
		nChannels = 1;
		dsize = 8;
		break;
	case 15:
		nChannels = 1;
		dsize = 8;
		break;
	case 6:										//comes here - Oberpfaffenhofen
		nChannels = 2;
		dsize = 4;
		break;
	case 9:
		nChannels = 2;
		dsize = 8;
		break;
	default: cerr << "ERROR: arraytyp not recognized (wrong format?)" << endl;
		exit(-1);
	}
	char* buf = new char(dsize);
	char* swap = new char(dsize);
	int i, j, x, y;
	Mat img;
	switch (dim) {
	case 2:         // scalar SAR image (e.g. only magnitude)
		data.resize(1);
		if (nChannels == 1)
			data[0] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
		else
			data[0] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
		for (y = 0; y < imgSize[1]; y++) {
			for (x = 0; x < imgSize[0]; x++) {
				//file.read((char*)(&buf), dsize);
				file.read(buf, dsize);
				double realVal, imagVal;
				// swap number
				for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
				switch (var) {
				case 1:
					realVal = *((char*)swap);
					break;	// byte
				case 2:
					realVal = *((int*)swap);
					break;	// int
				case 3:
					realVal = *((long*)swap);
					break;	// long
				case 4:
					realVal = *((float*)swap);
					break;	// float
				case 5:
					realVal = *((double*)swap);
					break;	// double
				case 6:
					realVal = *((float*)swap);
					//file.read((char*)(&buf), dsize);
					file.read(buf, dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((float*)swap);
					break;
				case 9:
					realVal = *((double*)swap);
					file.read((char*)(&buf), dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((double*)swap);
					break;
				case 12:
					realVal = *((unsigned int*)swap);
					break;	// uint
				case 13:
					realVal = *((unsigned long*)swap);
					break;	// ulong
				case 14:
					realVal = *((double*)swap);
					break;	// l64
				case 15:
					realVal = *((double*)swap);
					break;	// ul64
				}
				if ((imgSize[1] - y - 1 < tileStart[1]) || (imgSize[1] - y - 1 >= tileStart[1] + tileSize[1])) continue;
				if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
				if (nChannels != 2)
					data[0].at<float>(imgSize[1] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
				else
					data[0].at<Vec2f>(imgSize[1] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
			}
		}
		break;
	case 3:         // 3D SAR image (e.g. scattering vector)				//comes here - oberpfaffenhofen
		data.resize(imgSize[0]);
		for (i = 0; i < imgSize[0]; i++) {
			if (nChannels == 1)
				data[i] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
			else
				data[i] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
		}
		for (y = 0; y < imgSize[2]; y++)
		{
			for (x = 0; x < imgSize[1]; x++)
			{
				for (i = 0; i < imgSize[0]; i++)
				{
					//file.read((char*)(&buf), dsize);
					file.read(buf, dsize);
					double realVal, imagVal;
					for (int d = 0; d < dsize; d++)
						swap[d] = buf[dsize - d - 1];
					switch (var) {
					case 1:
						realVal = *((char*)swap);
						break;	// byte
					case 2:
						realVal = *((int*)swap);
						break;	// int
					case 3:
						realVal = *((long*)swap);
						break;	// long
					case 4:
						realVal = *((float*)swap);
						break;	// float
					case 5:
						realVal = *((double*)swap);
						break;	// double
					case 6:
						realVal = *((float*)swap);			// complex
						//file.read((char*)(&buf), dsize);
						file.read(buf, dsize);							//comes here..oberpffafenhofen
						for (int d = 0; d < dsize; d++)
							swap[d] = buf[dsize - d - 1];
						imagVal = *((float*)swap);
						break;
					case 9:
						realVal = *((double*)swap);					// dcomplex
						file.read((char*)(&buf), dsize);
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						imagVal = *((double*)swap);
						break;
					case 12:
						realVal = *((unsigned int*)swap);
						break;	// uint
					case 13:
						realVal = *((unsigned long*)swap);
						break;	// ulong
					case 14:
						realVal = *((double*)swap);
						break;	// l64
					case 15:
						realVal = *((double*)swap);
						break;	// ul64
					}
					if ((imgSize[2] - y - 1 < tileStart[1]) || (imgSize[2] - y - 1 >= tileStart[1] + tileSize[1])) continue;
					if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
					if (nChannels != 2)
						data.at(i).at<float>(imgSize[2] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
					else
						data.at(i).at<Vec2f>(imgSize[2] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
				}
			}
		}
		break;
	case 4:             // 4D SAR image (e.g. scattering matrix)
		data.resize(imgSize[0] * imgSize[1]);
		for (i = 0; i < imgSize[0]; i++) {
			for (j = 0; j < imgSize[1]; j++) {
				if (nChannels == 1)
					data[i * imgSize[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
				else
					data[i * imgSize[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
			}
		}
		for (y = 0; y < imgSize[3]; y++) {
			for (x = 0; x < imgSize[2]; x++) {
				for (j = 0; j < imgSize[0]; j++) {
					for (i = 0; i < imgSize[1]; i++) {
						file.read((char*)(&buf), dsize);
						double realVal, imagVal;
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						switch (var) {
						case 1:
							dsize = 1; realVal = *((char*)swap);
							break;	// byte
						case 2:
							dsize = 4; realVal = *((int*)swap);
							break;	// int
						case 3:
							dsize = 4; realVal = *((long*)swap);
							break;	// long
						case 4:
							dsize = 4; realVal = *((float*)swap);
							break;	// float
						case 5:
							dsize = 8; realVal = *((double*)swap);
							break;	// double
						case 6:
							dsize = 4;					// complex
							realVal = *((float*)swap);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((float*)swap);
							break;
						case 9:
							dsize = 8;					// dcomplex
							realVal = *((double*)swap);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((double*)swap);
							break;
						case 12:
							dsize = 4; realVal = *((unsigned int*)swap);
							break;	// uint
						case 13:
							dsize = 4; realVal = *((unsigned long*)swap);
							break;	// ulong
						case 14:
							dsize = 4; realVal = *((double*)swap);
							break;	// l64
						case 15:
							dsize = 4; realVal = *((double*)swap);
							break;	// ul64
						}
						if ((imgSize[3] - y - 1 < tileStart[1]) || (imgSize[3] - y - 1 >= tileStart[1] + tileSize[1])) continue;
						if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
						if (nChannels != 2)
							data.at(j * imgSize[1] + i).at<float>(imgSize[3] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
						else
							data.at(j * imgSize[1] + i).at<Vec2f>(imgSize[3] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
					}
				}
			}
		}
		break;
	}
	return Size(imgSize[dim - 2], imgSize[dim - 1]);
}

/**************************************************************
Function to read labels from label directory
Author: Anupama Rajkumar
Date: 23.05.2020
***************************************************************/
#ifdef VC

void ober::ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages) {
	fs::recursive_directory_iterator iter(labelPath);
	fs::recursive_directory_iterator end;
	while (iter != end) {
		string tp = iter->path().string();
		size_t pos = 0;
		//get the filename from path without extension
		string base_filename = tp.substr(tp.find_last_of("/\\") + 1);
		size_t position = base_filename.find(".");
		string fileName = (string::npos == position) ? base_filename : base_filename.substr(0, position);
		labelNames.push_back(fileName);

		Mat img = imread(tp);

		if (!img.data) {
			cout << "ERROR: Cannot find labelled image" << endl;
			cout << "Press enter to exit..." << endl;
			cin.get();
			exit(0);
		}
		// mask file should be 1 channel
		if (img.channels() > 1) {
			cvtColor(img, img, COLOR_BGR2GRAY);
		}
		else
		{
			img.convertTo(img, CV_8UC1);
		}
		labelImages.push_back(img);

		error_code ec;
		iter.increment(ec);
		if (ec) {
			std::cerr << "Error while accessing:" << iter->path().string() << "::" << ec.message() << "\n";
		}
	}
}
#endif // VC

#ifdef GCC

void ober::ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages) {
	struct dirent* entry;
	DIR* dir = opendir(labelPath.c_str());

	if (dir == NULL)
		return;

	std::size_t current = 0;
	int i = 0;
	while ((entry = readdir(dir)) != NULL)
	{
		if (strlen(entry->d_name) < 10)
			continue; // Ignore current folder (.) and parent folder (..)

		labelNames.push_back(entry->d_name);
		string filename = labelPath + labelNames[i];
		Mat_<float> img = imread(filename, IMREAD_GRAYSCALE);
		if (!img.data)
		{
			cout << "ERROR: file " << filename << " not found" << endl;
			cout << "Press enter to exit" << endl;
			cin.get();
			exit(-3);
		}
		// convert to floating point precision
		img.convertTo(img, CV_32FC1);
		labelImages.push_back(img);
		cout << "Loaded image " << labelNames[i] << endl;
		i++;
	}
	closedir(dir);
}

#endif //GCC

void ober::getTileInfo(cv::Size size, unsigned border, unsigned& tile, vector<unsigned>& tileSize, vector<unsigned>& tileStart) {

	cout << "Size:" << size.width << "x" << size.height << endl;				//1390 x 6640; number of channels:3
	tileSize[0] = size.width;
	tileSize[1] = size.height;

	tileStart[0] = 0;
	tileStart[1] = 0;

}