#pragma once
#ifndef  SARFEATURES_HPP_
#define  SARFEATURES_HPP_
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

namespace polsar {
	//Generate false color image
	// R:HH, G:HV, B:VV
	Mat GetFalseColorImg(const Mat& hh, const Mat& vv, const Mat& hv, bool normed = false);
	//R: HH+VV, G:HV, B: HH-VV
	Mat GetPauliColorImg(const Mat& hh, const Mat& vv, const Mat& hv);

	// process complex scattering values
	Mat getComplexAmpl(const Mat& in);
	Mat getComplexAngle(const Mat& in);
	Mat logTransform(const Mat& in);

	// convert CV_32FC2 to Complexf
	Mat_<Complexf> getComplexMat(const Mat& src);
	// get conjugation of complex matrix
	Mat getConj(const Mat& src);
	// get multiplication of two complex matrix
	Mat getMul(const Mat& src1, const Mat& src2);

	void getLexiBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& lexi);
	void getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli);
	void getCircBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& circ);

	// get coherency or covariance matrix, default winSize 3
	void vec2mat(const vector<Mat>& basis, vector<Mat>& mat, int winSize);

	Mat getPhaseDiff(const Mat& hh, const Mat& vv);

	// get coherency matrix T from Pauli basis
	void GetCoherencyT(const vector<Mat>& pauli, vector<Mat>& coherency, int winSize = 3);

	// get covariance matrix C from lexicographic basis
	void GetCovarianceC(const vector<Mat>& lexi, vector<Mat>& covariance, int winSize = 3);


	// coherent decomposition 
	// get krogager decomposition (sphere,diplane,helix) from circular basis
	void GetKrogagerDecomp(const vector<Mat>& circ, vector<Mat>& decomposition);

	// coherent decomposition 
	// get pauli decompostition (|alpha|,|beta|,|gamma|) from Pauli basis
	void GetPauliDecomp(const vector<Mat>& pauli, vector<Mat>& decomposition);

	// model based decomposition
	// get Ps, Pd, Pv from covariance matrix
	void GetFreemanDurdenDecomp(const vector<Mat>& covariance, vector<Mat>& decomposition);
	void freemanDurdenDecomp(const Mat_<Complexf>& covariance, vector<float>& result);

	// model based decomposition
	// get Ps, Pd, Pv, Pc from covariance matrix
	void GetYamaguchi4Decomp(const vector<Mat>& coherency, const vector<Mat>& covariance, vector<Mat>& decomposition);
	void yamaguchi4Decomp(const Mat_<Complexf>& coherency, const Mat_<Complexf>& covariance, vector<float>& result);

	// eigenvector based decomposition
	// get H, a, A from  coherency matrix
	void GetCloudePottierDecomp(const vector<Mat>& coherency, vector<Mat>& decomposition);
	void cloudePottierDecomp(Mat_<Complexf>& coherency, vector<float>& result);

	// dichotomy of the kennaugh matrix
	// get A0, B0, B, C, D, E, F, G, H from coherency matrix
	void GetHuynenDecomp(const vector<Mat> & covariance, vector<Mat> & decomposition);
	void huynenDecomp(const Mat_<Complexf>& covariance, vector<float>& result);
	
}

#endif