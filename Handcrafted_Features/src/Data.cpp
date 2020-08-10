#ifdef VC
#include <filesystem>
#endif // VC

#ifdef GCC
#include <dirent.h>
#endif
#include "Data.hpp" 


using namespace std;
using namespace cv;
namespace fs = std::filesystem;
 



/*************************************************************************
Generating a label map
Author : Anupama Rajkumar
Date : 27.05.2020
Description : Idea is to create a single label map from a list of various
label classes.This map serves as points of reference when trying to classify
patches
**************************************************************************
*/
cv::Mat Data::generateLabelMap(const std::vector<cv::Mat>& masks) {
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


#ifdef VC
void Data::loadData(string RATfolderPath) {
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
Size Data::loadRAT2(string fname, vector<Mat>& data, bool metaOnly) {

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

Size Data::loadRAT(string fname, vector<Mat>& data, bool metaOnly) {

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
		exit(-1);
	}

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
	for (auto i = 0; i < dim; i++) {
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
		for (auto i = 0; i < dim - 1; i++) cout << imgSize[i] << " x ";
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

void Data::ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages) {
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

void Data::ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages) {
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

void Data::getTileInfo(cv::Size size, unsigned border, unsigned& tile, vector<unsigned>& tileSize, vector<unsigned>& tileStart) {

	cout << "Size:" << size.width << "x" << size.height << endl;				//1390 x 6640; number of channels:3
	tileSize[0] = size.width;
	tileSize[1] = size.height;

	tileStart[0] = 0;
	tileStart[1] = 0;

}