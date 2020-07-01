#include "sarFeatures.hpp"
#include "cvFeatures.hpp"
#include <complex>
#include <Eigen/Eigenvalues>
#include <algorithm>    
 


using namespace cv;
using namespace std;

static constexpr float m_Epsilon = 1e-20f;
const float PI_F = 3.14159265358979f;
const float CONST_180_PI = 180.0f / PI_F;

Mat polsar::getComplexAmpl(const Mat& in) {

	Mat out, phase;
	vector<Mat> channels;
	split(in, channels);

	//pow(channels[0], 2, channels[0]);
	//pow(channels[1], 2, channels[1]);
    //out = channels[0] + channels[1];
	//pow(out, 0.5, out);
	cv::cartToPolar(channels[0], channels[1], out, phase);
	
	return out;
}

Mat polsar::logTransform(const Mat& in) {

	Mat out = in.clone();
	if (in.channels() == 2) {
		out = getComplexAmpl(in);
	}
	out = out + 1;
	log(out, out);

	return out;
}

Mat polsar::getComplexAngle(const Mat& in) {

	vector<Mat> channels;
	split(in, channels);
	Mat amp, out;
	cartToPolar(channels[0], channels[1], amp, out);

	return out;

}

void polsar::getLexiBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& lexi) {
	lexi.push_back(hh);
	lexi.push_back(sqrt(2.0) * hv);
	lexi.push_back(vv);
}


void polsar::getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli) {
	pauli.push_back((hh + vv) / sqrt(2.0));
	pauli.push_back((hh - vv) / sqrt(2.0));
	pauli.push_back(hv * sqrt(2.0));
}

/**
* Create Span image.
*
* @param sourceBands         the input bands
* @param sourceTileRectangle The source tile rectangle.
* @param span                The span image.
*/
// The pixel value of the span image is given by the trace of the covariance or coherence matrix for the pixel.
void polsar::createSpanImage(const Mat& m00, const Mat& m11, const Mat& m22, Mat& span) {

	span = Mat(Size(m00.size()), m00.type());

	span = (m00 + m11 + m22) / 4.0;
}

void polsar::getCircBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& circ) {

	// i*s_hv = i*(a+ib) = ai-b = -b+ia
	Mat a, b;
	cv::extractChannel(hv, a, 0);
	cv::extractChannel(hv, b, 1);
	vector<Mat> channels;
	channels.push_back(-1.0 * b);
	channels.push_back(a);
	Mat i_hv;
	merge(channels, i_hv);

	// s_rr = (s_hh - s_vv + i*2*s_hv)/2
	Mat srr = (hh - vv + 2.0 * i_hv) / 2.0;

	// s_ll = (s_vv - s_hh + i*2*s_hv)/2
	Mat sll = (vv - hh + 2.0 * i_hv) / 2.0;

	// s_lr = i*(s_hh + s_vv)/2
	Mat temp = hh + vv;
	cv::extractChannel(temp, a, 0);
	cv::extractChannel(temp, b, 1);
	channels.clear();
	channels.push_back(-1.0 * b);
	channels.push_back(a);
	Mat slr;
	merge(channels, slr);
	slr = 0.5 * slr;

	circ.push_back(sll);
	circ.push_back(srr);
	circ.push_back(slr);

}

// return the log form
Mat polsar::getCoPolarizationRatio(const Mat& hh, const Mat& vv, int winSize) {
	Mat output1, output2, output;
	output1 = calcuCoherenceOfPol(vv, vv, winSize);
	output2 = calcuCoherenceOfPol(hh, hh, winSize);

	output =logTransform(output1) - logTransform(output2);
	return output;
}

// return the log form
Mat polsar::getDePolarizationRatio(const Mat& hh, const Mat& vv, const Mat& hv, int winSize) {

	Mat output;
	Mat output1 = calcuCoherenceOfPol(hv, hv, winSize);
	Mat	output2 = calcuCoherenceOfPol(hh, hh, winSize);
	Mat output3 = calcuCoherenceOfPol(vv, vv, winSize);  

	output = logTransform(output1) - logTransform(output2 + output3);
	return output;
}

// get <band1*conju(band2)>
// output: 2 channel matrix
Mat polsar::calcuCoherenceOfPol(const Mat& band1, const Mat& band2, int winSize) {
	Mat output1, output;
	cv::mulSpectrums(band1, band2, output1, 0, true); // band1* conju(band2)
	cv::blur(output1, output, Size(winSize, winSize));
	return output;
}

// get the relative phases
Mat polsar::getPhaseDiff(const Mat& hh, const Mat& vv) {
	//Mat temp = getMul(hh, getConj(vv));
	//return getComplexAngle(temp);
	Mat hh_real, hh_imag, vv_real, vv_imag;
	extractChannel( hh, hh_real, 0);
	extractChannel( hh, hh_imag, 1);
	extractChannel( vv, vv_real, 0);
	extractChannel(vv, vv_imag, 1);
	Mat output= Mat(Size(hh.size()), CV_32FC1);
	for (int i = 0; i < hh.rows; i++) {
		for (int j = 0; j < hh.cols; j++) {
			output.at<float>(i, j) = atan(hh_imag.at<float>(i, j) / hh_real.at<float>(i, j)) - atan(vv_imag.at<float>(i, j) / vv_real.at<float>(i, j));
		}
	}
	return output;
}

void polsar::vec2mat(const vector<Mat>& basis, vector<Mat>& mat, int winSize) {
	Mat m00, m01, m02, m11, m12, m22;

	mulSpectrums(basis.at(0), basis.at(0), m00, 0, true); //|k_0 | ^ 2
	mulSpectrums(basis.at(0), basis.at(1), m01, 0, true); //k_0*conj(k_1)
	mulSpectrums(basis.at(0), basis.at(2), m02, 0, true); //k_0*conj(k_2)
	mulSpectrums(basis.at(1), basis.at(1), m11, 0, true); //k_1|^2
	mulSpectrums(basis.at(1), basis.at(2), m12, 0, true); //k_1*conj(k_2)
	mulSpectrums(basis.at(2), basis.at(2), m22, 0, true); //|k_2|^2 

	//m00 = getMul(basis.at(0), getConj(basis.at(0))); // k_0*conj(k_1)
	//m01 = getMul(basis.at(0), getConj(basis.at(1))); // k_0*conj(k_1)
	//m02 = getMul(basis.at(0), getConj(basis.at(2))); // k_0*conj(k_2)
	//m11 = getMul(basis.at(1), getConj(basis.at(1))); // k_1*conj(k_1)
	//m12 = getMul(basis.at(1), getConj(basis.at(2))); // k_1*conj(k_2)
	//m22 = getMul(basis.at(2), getConj(basis.at(2))); // k_2*conj(k_2)


	cv::blur(m00, m00, Size(winSize, winSize));
	cv::blur(m01, m01, Size(winSize, winSize));
	cv::blur(m02, m02, Size(winSize, winSize));
	cv::blur(m11, m11, Size(winSize, winSize));
	cv::blur(m12, m12, Size(winSize, winSize));
	cv::blur(m22, m22, Size(winSize, winSize));

	// the real part is the squared amplitude
	Mat m00_dat, m11_dat, m22_dat;
	extractChannel(m00, m00_dat, 0);
	extractChannel(m11, m11_dat, 0);
	extractChannel(m22, m22_dat, 0);

	mat.push_back(m00_dat);
	mat.push_back(m01);
	mat.push_back(m02);
	mat.push_back(m11_dat);
	mat.push_back(m12);
	mat.push_back(m22_dat);
}

// get the whole C or T matrix from up corner elements
void polsar::getCompleteMatrix(const vector<Mat>& mat, vector<Mat>& complete_mat) {
	Mat m00 = mat[0];
	Mat m01 = mat[1];
	Mat m02 = mat[2];
	Mat m11 = mat[3];
	Mat m12 = mat[4];
	Mat m22 = mat[5];
	Mat m10 = getConj(m01);
	Mat m20 = getConj(m02);
	Mat m21 = getConj(m12);
	complete_mat.push_back(m00);
	complete_mat.push_back(m01);
	complete_mat.push_back(m02);
	complete_mat.push_back(m10);
	complete_mat.push_back(m11);
	complete_mat.push_back(m12);
	complete_mat.push_back(m20);
	complete_mat.push_back(m21);
	complete_mat.push_back(m22);
}

Mat polsar::GetColorImg(const Mat& R, const Mat& G, const Mat& B, bool normed) {
	vector<Mat> Channels;
	Mat output;
	Channels.push_back(B);
	Channels.push_back(G);
	Channels.push_back(R);
	merge(Channels, output);
	if (normed) {
		normalize(output, output, 0, 255, NORM_MINMAX);
		output.convertTo(output, CV_8UC3);
	}
	return output;
}

Mat polsar::GetFalseColorImg(const Mat& hh, const Mat& vv, const Mat& hv, const Mat& vh, bool normed)
{
	Mat R, G, B;
	if (!hh.empty() && !vv.empty() && !hv.empty()) {
		R = logTransform(getComplexAmpl(hh));
		G = logTransform(getComplexAmpl(hv));
		B = logTransform(getComplexAmpl(vv));
	}
	else if (!vv.empty() && !vh.empty()) {
		R = logTransform(getComplexAmpl(vv));
		G = logTransform(getComplexAmpl(vh));
		B = Mat::zeros(vv.rows, vv.cols, R.type()); //VV/VH
		B = R / G;
	}
	//else if (!hh.empty() && !hv.empty()) {
	//	R = logTransform(getComplexAmpl(hh));
	//	G = logTransform(getComplexAmpl(hv));
	//	B = Mat::zeros(hh.rows, hh.cols, R.type()); //HH/HV
	//	B = R / G;
	//}
	else {
		cout << "input pol data is empty" << endl;
		return Mat();
	}
	return GetColorImg(R, G, B,normed);
}



//R: |HH+VV|, G:|HV|, B: |HH-VV|
Mat polsar::GetPauliColorImg(const Mat& hh, const Mat& vv, const Mat& hv) {

	vector<Mat> pauli;
	getPauliBasis(hh, vv, hv, pauli);
	Mat R = logTransform(getComplexAmpl(pauli[0]));
	Mat G = logTransform(getComplexAmpl(pauli[2]));
	Mat B = logTransform(getComplexAmpl(pauli[1]));

	//Mat R = logTransform(getComplexAmpl(hh+vv));
	//Mat G = logTransform(getComplexAmpl(hv));
	//Mat B = logTransform(getComplexAmpl(hh-vv));

	//cut everything over 2.5x the mean value
	float R_mean = cv::mean(R)[0];
	float G_mean = cv::mean(G)[0];
	float B_mean = cv::mean(B)[0];
	threshold(R, R, 2.5 * R_mean, 2.5 * R_mean, THRESH_TRUNC);
	threshold(G, G, 2.5 * G_mean, 2.5 * G_mean, THRESH_TRUNC);
	threshold(B, B, 2.5 * B_mean, 2.5 * B_mean, THRESH_TRUNC);
	return GetColorImg(R, G, B, true);
}

// get the complex conjugation of a 2 channel matrix
Mat polsar::getConj(const Mat& src) {
	Mat temp = Mat(Size(src.size()), CV_32FC2);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			temp.at<Vec2f>(i, j)[0] = src.at<Vec2f>(i, j)[0];
			temp.at<Vec2f>(i, j)[1] = src.at<Vec2f>(i, j)[1] * -1.0;
		}
	return temp;
}

// get the complex muliplication of two 2 channel matrix
Mat polsar::getMul(const Mat& src1, const Mat& src2) {
	Mat temp = Mat(Size(src1.size()), CV_32FC2);
	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++) {
			// (a+bi)(c+di) = ac-bd + (ad + bc)i
			float a = src1.at<Vec2f>(i, j)[0];
			float b = src1.at<Vec2f>(i, j)[1];
			float c = src2.at<Vec2f>(i, j)[0];
			float d = src2.at<Vec2f>(i, j)[1];
			temp.at<Vec2f>(i, j)[0] = a * c - b * d;
			temp.at<Vec2f>(i, j)[1] = a * d + b * c;
		}
	return temp;
}



// convert CV_32FC2 to Complexf
Mat_<Complexf> polsar::getComplexMat(const Mat& src) {
	Mat dst = Mat_<Complexf>(Size(src.size()));
	if (src.channels() == 2) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst.at<Complexf>(i, j).re = src.at<Vec2f>(i, j)[0];
				dst.at<Complexf>(i, j).im = src.at<Vec2f>(i, j)[1];
			}
		}
	}
	else if (src.channels() == 1) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst.at<Complexf>(i, j).re = src.at<float>(i, j);
				dst.at<Complexf>(i, j).im = 0.0f;
			}
		}
	}
	return dst;
}

void polsar::GetKrogagerDecomp(const vector<Mat>& circ, vector<Mat>& decomposition) {
	// sphere, diplane, helix-decomposition
	Mat k_s, k_d, k_h;
	Mat s_ll_amp, s_rr_amp, s_lr_amp;
	s_ll_amp = getComplexAmpl(circ.at(0));
	s_rr_amp = getComplexAmpl(circ.at(1));
	s_lr_amp = getComplexAmpl(circ.at(2));

	// k_s = |s_lr|
	k_s = logTransform( s_lr_amp);

	// k_d = min( |s_ll|, |s_rr| )
	min(s_ll_amp, s_rr_amp, k_d);
	k_d = logTransform(k_d);

	// k_h = | |s_ll| - |s_rr| |
	k_h = abs(s_ll_amp - s_rr_amp);
	k_h = logTransform(k_h);

	decomposition.push_back(k_s);
	decomposition.push_back(k_d);
	decomposition.push_back(k_h);
}

// reference
// https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/blob/develop/Modules/Filtering/Polarimetry/include/otbReciprocalPauliDecompImageFilter.h
void polsar::GetPauliDecomp(const vector<Mat>& pauli, vector<Mat> & decomposition) {

	decomposition.push_back(logTransform(getComplexAmpl(pauli.at(0))));
	decomposition.push_back(logTransform(getComplexAmpl(pauli.at(1))));
	decomposition.push_back(logTransform(getComplexAmpl(pauli.at(2))));
}

// reference:
// https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/blob/develop/Modules/Filtering/Polarimetry/include/otbReciprocalHuynenDecompImageFilter.h
void polsar::huynenDecomp(const Mat_<Complexf>& covariance, vector<float>& result) {
	float A0, B0, B, C, D, E, F, G, H;
	A0 = covariance.at<Complexf>( 0, 0).re / 2.0f;
	B0 = (covariance.at<Complexf>(1, 1)+covariance.at<Complexf>(2, 2)).re/ 2.0f;
	B = covariance.at<Complexf>(1, 1).re - B0;
	C = covariance.at<Complexf>(0, 1).re;
	D = -1.0f*covariance.at<Complexf>(0, 1).im;
	E = covariance.at<Complexf>(1, 2).re;
	F = covariance.at<Complexf>(1, 2).im;
	G = covariance.at<Complexf>(0, 2).im;
	H = covariance.at<Complexf>(0, 2).re;
	
	result.push_back(A0);
	result.push_back(B0);
	result.push_back(B);
	result.push_back(C);
	result.push_back(D);
	result.push_back(E);
	result.push_back(F);
	result.push_back(G);
	result.push_back(H);
}


void polsar::GetHuynenDecomp(const vector<Mat>& upcorner_coherence, vector<Mat>& decomposition) {
	Mat A0, B0, B, C, D, E, F, G, H;
	extractChannel(upcorner_coherence[0], A0, 0);
	A0 = A0 / 2.0;
    
	extractChannel(upcorner_coherence[3]+ upcorner_coherence[5], B0, 0);
	B0 =  B0 / 2.0;

	extractChannel(upcorner_coherence[3], B, 0);
	B = B - B0;

	extractChannel(upcorner_coherence[1], C, 0);

	extractChannel(upcorner_coherence[1], D, 1);
	D = -1.0 * D;

	extractChannel(upcorner_coherence[4], E, 0);

	extractChannel(upcorner_coherence[4], F, 1);

	extractChannel(upcorner_coherence[2], G, 1);

	extractChannel(upcorner_coherence[2], H, 0);

	decomposition.push_back(A0);
	decomposition.push_back(B0);
	decomposition.push_back(B);
	decomposition.push_back(C);
	decomposition.push_back(D);
	decomposition.push_back(E);
	decomposition.push_back(F);
	decomposition.push_back(G);
	decomposition.push_back(H);
}

// reference
// https://github.com/senbox-org/s1tbx/blob/master/rstb/rstb-op-polarimetric-tools/src/main/java/org/csa/rstb/polarimetric/gpf/decompositions/FreemanDurden.java
void polsar::freemanDurdenDecomp(const Mat_<Complexf>& covariance, vector<float>& result) {

	float fd, fv, fs, c11, c13Re, c13Im, c33, alphaRe, alphaIm, betaRe, betaIm;
	// compute fv from m22 and subtract fv from m11, m13, m33
	fv = 4.0f * covariance.at<Complexf>(1, 1).re;
	c11 = covariance.at<Complexf>(0, 0).re - fv * 3.0f / 8.0f;
	c13Re = covariance.at<Complexf>(0, 2).re - fv / 8.0f;
	c13Im = covariance.at<Complexf>(0, 2).im;
	c33 = covariance.at<Complexf>(2, 2).re - fv * 3.0f / 8.0f;
	float a1 = c11 * c33;

	if (c11 <= m_Epsilon || c33 <= m_Epsilon) {
		fs = 0.0f;
		fd = 0.0f;
		alphaRe = 0.0f;
		alphaIm = 0.0f;
		betaRe = 0.0f;
		betaIm = 0.0f;
	}
	else {

		float a2 = c13Re * c13Re + c13Im * c13Im;
		if (a1 < a2) {
			float c13 = std::sqrt(a2);
			c13Re = std::sqrt(a1) * c13Re / c13;
			c13Im = std::sqrt(a1) * c13Im / c13;
		}
		// get sign of Re(C13), if it is minus, set beta = 1; else set alpha = -1
		if (c13Re < 0.0) {

			betaRe = 1.0;
			betaIm = 0.0;
			fs = std::abs((a1 - c13Re * c13Re - c13Im * c13Im) / (c11 + c33 - 2 * c13Re));
			fd = std::abs(c33 - fs);
			alphaRe = (c13Re - fs) / fd;
			alphaIm = c13Im / fd;

		}
		else {

			alphaRe = -1.0;
			alphaIm = 0.0;
			fd = std::abs((a1 - c13Re * c13Re - c13Im * c13Im) / (c11 + c33 + 2 * c13Re));
			fs = std::abs(c33 - fd);
			betaRe = (c13Re + fd) / fs;
			betaIm = c13Im / fs;
		}
	}

	// compute Ps, Pd and Pv
	float ps = fs * (1 + betaRe * betaRe + betaIm * betaIm);
	float pd = fd * (1 + alphaRe * alphaRe + alphaIm * alphaIm);
	float pv = fv;
	result.push_back(ps);
	result.push_back(pd);
	result.push_back(pv);
}


// reference:
// https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/blob/develop/Modules/Filtering/Polarimetry/include/otbReciprocalHAlphaImageFilter.h
void polsar::cloudePottierDecomp(Mat_<Complexf>& coherence, vector<float>& result) {

	Eigen::Map<Eigen::Matrix<std::complex<float>, 3, 3, Eigen::RowMajor>> eigen_mat(coherence.ptr<std::complex<float>>(), coherence.rows, coherence.cols);
	Eigen::ComplexEigenSolver<Eigen::MatrixXcf> ces;
	ces.compute(eigen_mat);

	vector<int> V;// indicating positions
	vector<float> realEigenValues;
	// extract the first component of each eigen vector sorted by eigen value decrease order
	vector<complex<float>> cos_alpha;

	int N = ces.eigenvalues().rows();
	for (int i = 0; i < N; i++) { V.push_back(i); }
	// sort eigen values in decreasing order, and record the original index in V
	sort(V.begin(), V.end(), [&](int i, int j) {return ces.eigenvalues()[i].real() > ces.eigenvalues()[j].real(); });

	for(auto & i: V){
		realEigenValues.push_back(ces.eigenvalues()[i].real());
		cos_alpha.push_back(ces.eigenvectors()(0.0f, i));
	}
	//check the size of eigen values
	if (N == 2) {
		realEigenValues.push_back(0.0f);
		cos_alpha.push_back(complex<float>(0.0f, 0.0f));
	}
	if (N == 1) {
		realEigenValues.push_back(0.0);
		realEigenValues.push_back(0.0);
		cos_alpha.push_back(complex<float>(0.0f, 0.0f));
		cos_alpha.push_back(complex<float>(0.0f, 0.0f));
	}

	// Entropy estimation
	float totalEigenValues = 0.0f;
	float p[3];
	float plog[3];
	float entropy;
	float alpha;
	float anisotropy;
	for (unsigned int k = 0; k < 3; ++k)
	{
		realEigenValues[k] = std::max(realEigenValues[k], 0.0f);
		totalEigenValues += realEigenValues[k];
	}


	for (unsigned int k = 0; k < 3; ++k)
	{
		p[k] = realEigenValues[k] / totalEigenValues;

		if (p[k] < m_Epsilon) // n=log(n)-->0 when n-->0
			plog[k] = 0.0f;
		else
			plog[k] = -p[k] * log(p[k]) / log(3.0f);
	}

	entropy = 0.0f;
	for (unsigned int k = 0; k < 3; ++k)
		entropy += plog[k];

	// Anisotropy estimation
	anisotropy = (realEigenValues[1] - realEigenValues[2]) / (realEigenValues[1] + realEigenValues[2] + m_Epsilon);

	// alpha estimation
	float val0, val1, val2;
	float a0, a1, a2;

	val0 = std::abs(cos_alpha[0]);
	a0 = acos(std::abs(val0)) * CONST_180_PI;

	val1 = std::abs(cos_alpha[1]);
	a1 = acos(std::abs(val1)) * CONST_180_PI;

	val2 = std::abs(cos_alpha[2]);
	a2 = acos(std::abs(val2)) * CONST_180_PI;

	alpha = p[0] * a0 + p[1] * a1 + p[2] * a2;

	result.push_back(entropy);
	result.push_back(anisotropy);
	result.push_back(alpha);
}

//reference
//https://github.com/senbox-org/s1tbx/blob/master/rstb/rstb-op-polarimetric-tools/src/main/java/org/csa/rstb/polarimetric/gpf/decompositions/Yamaguchi.java
void polsar::yamaguchi4Decomp(const Mat_<Complexf>& coherence, const Mat_<Complexf>& covariance, vector<float>& result) {
	float ratio, d, cR, cI, c0, s, pd, pv, ps, pc, span, k1, k2, k3;

	span = coherence.at<Complexf>(0, 0).re + coherence.at<Complexf>(1, 1).re + coherence.at<Complexf>(2, 2).re;
	pc = 2.0f * std::abs(coherence.at<Complexf>(1,2).im);
	ratio = 10.0f * std::log10(covariance.at<Complexf>(2,2).re / covariance.at<Complexf>(0,0).re);

	if (ratio <= -2.0f) {
		k1 = 1.0f / 6.0f;
		k2 = 7.0f / 30.0f;
		k3 = 4.0f / 15.0f;
	}
	else if (ratio > 2.0f) {
		k1 = -1.0f / 6.0f;
		k2 = 7.0f / 30.0f;
		k3 = 4.0f / 15.0f;
	}
	else { // -2 < ratio <= 2
		k1 = 0.0f;
		k2 = 1.0f / 4.0f;
		k3 = 1.0f / 4.0f;
	}

	pv = (coherence.at<Complexf>(2,2).re - 0.5f * pc) / k3;

	if (pv <= m_Epsilon) { // Freeman-Durden 3 component decomposition
		pc = 0.0f;
		freemanDurdenDecomp(covariance, result);
		result.push_back(pc);
	}
	else { // Yamaguchi 4 component decomposition

		s = coherence.at<Complexf>(0,0).re - 0.5f * pv;
		d = coherence.at<Complexf>(1,1).re - k2 * pv - 0.5f * pc;
		cR = coherence.at<Complexf>(0, 1).re - k1 * pv;
		cI = coherence.at<Complexf>(0, 1).im;

		if (pv + pc < span) {

			c0 = covariance.at<Complexf>(0,2).re - 0.5f * covariance.at<Complexf>(1, 1).re + 0.5f * pc;
			if (c0 < m_Epsilon) {
				ps = s - (cR * cR + cI * cI) / d;
				pd = d + (cR * cR + cI * cI) / d;
			}
			else {
				ps = s + (cR * cR + cI * cI) / s;
				pd = d - (cR * cR + cI * cI) / s;
			}

			if (ps > m_Epsilon && pd < m_Epsilon) {
				pd = 0.0f;
				ps = span - pv - pc;
			}
			else if (ps < m_Epsilon && pd > m_Epsilon) {
				ps = 0.0f;
				pd = span - pv - pc;
			}
			else if (ps < m_Epsilon && pd < m_Epsilon) {
				ps = 0.0f;
				pd = 0.0f;
				pv = span - pc;
			}

		}
		else {
			ps = 0.0f;
			pd = 0.0f;
			pv = span - pc;
		}
		result.push_back(ps);
		result.push_back(pd);
		result.push_back(pv);
		result.push_back(pc);
	}
}

void polsar::GetFreemanDurdenDecomp(const vector<Mat>& upcorner_covariance, vector<Mat>& decomposition) {
	int rows = upcorner_covariance[0].rows;
	int cols = upcorner_covariance[0].cols;

	// record the result
	Mat Ps = Mat::zeros(rows, cols, CV_32FC1);
	Mat Pd = Mat::zeros(rows, cols, CV_32FC1);
	Mat Pv = Mat::zeros(rows, cols, CV_32FC1);

	Mat_<Complexf>  m = Mat_<Complexf>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			m = restoreMatrix(upcorner_covariance, i, j);
			vector<float> result;
			freemanDurdenDecomp(m, result);
			Ps.at<float>(i, j) = result[0];
			Pd.at<float>(i, j) = result[1];
			Pv.at<float>(i, j) = result[2];
		}
	}
	decomposition.push_back(Ps);
	decomposition.push_back(Pd);
	decomposition.push_back(Pv);
}

void polsar::GetCloudePottierDecomp(const vector<Mat>& upcorner_coherence, vector<Mat>& decomposition) {
	// restore the original coherecy matrix from the diagonal and the upper elements 
	int rows = upcorner_coherence[0].rows;
	int cols = upcorner_coherence[0].cols;

	// record the result
	Mat H = Mat::zeros(rows,cols, CV_32FC1);
	Mat A = Mat::zeros(rows, cols,  CV_32FC1);
	Mat Alpha = Mat::zeros(rows, cols,  CV_32FC1);

	Mat_<Complexf> t = Mat_<Complexf>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j <cols; j++) {
			t = restoreMatrix(upcorner_coherence, i, j);
			vector<float> result;
			cloudePottierDecomp(t, result);
			H.at<float>(i, j) = result[0];
			Alpha.at<float>(i, j) = result[1];
			A.at<float>(i, j) = result[2];
		}
	}
	decomposition.push_back(H);
	decomposition.push_back(Alpha);
	decomposition.push_back(A);
}

void polsar::GetYamaguchi4Decomp(const vector<Mat>& upcorner_coherence, const vector<Mat> & upcorner_covariance, vector<Mat>& decomposition) {
	 
	int rows = upcorner_coherence[0].rows;
	int cols = upcorner_coherence[0].cols;
	// record the result
	Mat Ps = Mat::zeros(rows,cols, CV_32FC1);
	Mat Pd= Mat::zeros(rows, cols, CV_32FC1);
	Mat Pv = Mat::zeros(rows, cols, CV_32FC1);
	Mat Pc = Mat::zeros(rows, cols, CV_32FC1);

	// restore the original coherecy matrix from the diagonal and the upper elements 
	Mat_<Complexf> t = Mat_<Complexf>(3, 3);
	Mat_<Complexf> c = Mat_<Complexf>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			t = restoreMatrix(upcorner_coherence, i, j);
			c = restoreMatrix(upcorner_covariance, i, j);
			vector<float> result;
			yamaguchi4Decomp(t, c, result);
			Ps.at<float>(i, j) = result[0];
			Pd.at<float>(i, j) = result[1];
			Pv.at<float>(i, j) = result[2];
			Pc.at<float>(i, j) = result[3];
		}
	}
	decomposition.push_back(Ps);
	decomposition.push_back(Pd);
	decomposition.push_back(Pv);
	decomposition.push_back(Pc);
}


void polsar::GetCoherencyT(const vector<Mat>& pauli, vector<Mat>& upcorner_coherence, int winSize) { vec2mat(pauli, upcorner_coherence, winSize); }

void polsar::GetCovarianceC(const vector<Mat>& lexi, vector<Mat>& upcorner_covariance, int winSize) { vec2mat(lexi, upcorner_covariance, winSize); }


// get the log upper triangle elements of matrix elements of C, T
void polsar::GetCTelements(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& result) {

	vector<Mat> pauli;
	vector<Mat> circ;
	vector<Mat> lexi;
	polsar::getPauliBasis(hh, vv, hv, pauli);
	polsar::getCircBasis(hh, vv, hv, circ);
	polsar::getLexiBasis(hh, vv, hv, lexi);
	vector<Mat> covariance;
	vector<Mat> coherence;
	polsar::GetCoherencyT(pauli, coherence);
	polsar::GetCovarianceC(lexi, covariance);

	// upper triangle matrix elements of covariance matrix C and coherence matrix T
	copy(coherence.begin(), coherence.end(), std::back_inserter(result));
	copy(covariance.begin(), covariance.end(), std::back_inserter(result));

	for (auto& e : result) {
		Mat temp;
		if (e.channels() == 1) { 
			temp = polsar::logTransform(e);
		}
		else if (e.channels() ==2 ) {
			temp = polsar::logTransform(polsar::getComplexAmpl(e));
		}
		e = temp;
	}
}

//restore 3*3 covariance or coherence mat from upcorner elements
Mat polsar::restoreMatrix(const vector<Mat>& mat, int row, int col) {
	Mat_<Complexf> m(3, 3);
	m.at<Complexf>(0, 0) = Complex(mat[0].at<float>(row, col), 0.0f);
	m.at<Complexf>(0, 1) = Complex(mat[1].at<Vec2f>(row, col)[0], mat[1].at<Vec2f>(row, col)[1]);
	m.at<Complexf>(0, 2) = Complex(mat[2].at<Vec2f>(row, col)[0], mat[2].at<Vec2f>(row, col)[1]);
	m.at<Complexf>(1, 0) = m.at<Complexf>(0, 1).conj();
	m.at<Complexf>(1, 1) = Complex(mat[3].at<float>(row, col), 0.0f);
	m.at<Complexf>(1, 2) = Complex(mat[4].at<Vec2f>(row, col)[0], mat[4].at<Vec2f>(row, col)[1]);
	m.at<Complexf>(2, 0) = m.at<Complexf>(0, 2).conj();
	m.at<Complexf>(2, 1) = m.at<Complexf>(1, 2).conj();
	m.at<Complexf>(2, 2) = Complex(mat[5].at<float>(row, col), 0.0f);
	return m;
}

