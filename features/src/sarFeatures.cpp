#include "sarFeatures.hpp"
#include <complex>
#include "Eigenvalues"
#include <algorithm>    // std::sort
 


using namespace cv;
using namespace std;

static constexpr float m_Epsilon = 1e-6f;
const float PI_F = 3.14159265358979f;
const float CONST_180_PI = 180.0 / PI_F;

Mat polsar::getComplexAmpl(const Mat& in) {

	vector<Mat> channels;

	split(in, channels);
	pow(channels[0], 2, channels[0]);
	pow(channels[1], 2, channels[1]);
	Mat out = channels[0] + channels[1];
	pow(out, 0.5, out);

	return out;

}

// should be 10*log(|A|)
Mat polsar::logTransform(const Mat& in) {

	Mat out;
	if (in.channels() == 2) {
		out = getComplexAmpl(in);
	}
	else
		out = in;

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
	lexi.push_back(sqrt(2) * hv);
	lexi.push_back(vv);
}


void polsar::getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli) {
	pauli.push_back( (hh + vv) / sqrt(2));
	pauli.push_back( (hh - vv) / sqrt(2));
	pauli.push_back( hv * sqrt(2));
}


void polsar::getCircBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& circ) {

	// i*s_hv = i*(a+ib) = ai-b = -b+ia
	Mat a, b;
	cv::extractChannel(hv, a, 0);
	cv::extractChannel(hv, b, 1);
	vector<Mat> channels;
	channels.push_back(-1 * b);
	channels.push_back(a);
	Mat i_hv;
	merge(channels, i_hv);

	// s_rr = (s_hh - s_vv + i*2*s_hv)/2
	Mat srr = (hh - vv + 2 * i_hv) / 2.0;

	// s_ll = (s_vv - s_hh + i*2*s_hv)/2
	Mat sll = (vv - hh + 2 * i_hv) / 2.0;

	// s_lr = i*(s_hh + s_vv)/2
	Mat temp = hh + vv;
	cv::extractChannel(temp, a, 0);
	cv::extractChannel(temp, b, 1);
	channels.clear();
	channels.push_back(-1 * b);
	channels.push_back(a);
	Mat slr;
	merge(channels, slr);
	slr = 0.5 * slr;

	circ.push_back(sll);
	circ.push_back(srr);
	circ.push_back(slr);

}


void polsar::vec2mat(const vector<Mat>& basis, vector<Mat>& mat, int winSize) {
	Mat m00, m01, m02, m10, m11, m12, m20, m21, m22;
	pow(getComplexAmpl(basis.at(0)), 2, m00); // |k_1|^2
	m01 = getMul(basis.at(0), getConj(basis.at(1))); // k_1*conj(k_2)
	m02 = getMul(basis.at(0), getConj(basis.at(2))); // k_1*conj(k_3)
	m10 = getMul(basis.at(1), getConj(basis.at(0))); // k_2*conj(k_1)
	pow(getComplexAmpl(basis.at(1)), 2, m11); // |k_2|^2 
	m12 = getMul(basis.at(1), getConj(basis.at(2))); // k_2*conj(k_3)
	m20 = getMul(basis.at(2), getConj(basis.at(0))); // k_3*conj(k_1)
	m21 = getMul(basis.at(2), getConj(basis.at(1))); // k_3*conj(k_2)
	pow(getComplexAmpl(basis.at(2)), 2, m22); // |k_3|^2 

	cv::blur(m00, m00, Size(winSize, winSize));
	cv::blur(m01, m01, Size(winSize, winSize));
	cv::blur(m02, m02, Size(winSize, winSize));
	cv::blur(m10, m10, Size(winSize, winSize));
	cv::blur(m11, m11, Size(winSize, winSize));
	cv::blur(m12, m12, Size(winSize, winSize));
	cv::blur(m20, m20, Size(winSize, winSize));
	cv::blur(m21, m21, Size(winSize, winSize));
	cv::blur(m22, m22, Size(winSize, winSize));

	mat.push_back(m00);
	mat.push_back(m01);
	mat.push_back(m02);
	mat.push_back(m10);
	mat.push_back(m11);
	mat.push_back(m12);
	mat.push_back(m20);
	mat.push_back(m21);
	mat.push_back(m22);
}



// R:|HH|, G:|HV|, B:|VV|
Mat polsar::GetFalseColorImg(const Mat& hh, const Mat& vv, const Mat& hv, bool normed)
{
	vector<Mat>  Channels;
	Mat output;
	Mat R = logTransform(getComplexAmpl(hh));
	Mat G = logTransform(getComplexAmpl(hv));
	Mat B = logTransform(getComplexAmpl(vv));
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



//R: |HH+VV|, G:|HV|, B: |HH-VV|
Mat polsar::GetPauliColorImg(const Mat& hh, const Mat& vv, const Mat& hv) {

	Mat R = hh + vv;
	Mat G = hv;
	Mat B = hh - vv;
	return GetFalseColorImg(R, G, B, true);
}

// get the complex conjugation of a 2 channel matrix
Mat polsar::getConj(const Mat& src) {
	Mat temp = Mat(Size(src.size()), CV_32FC2);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			std::complex a(src.at<Vec2f>(i, j)[0], src.at<Vec2f>(i, j)[1]);
			std::complex c = std::conj(a);
			temp.at<Vec2f>(i, j)[0] = c.real();
			temp.at<Vec2f>(i, j)[1] = c.imag();
		}
	return temp;
}

// get the complex muliplication of two 2 channel matrix
Mat polsar::getMul(const Mat& src1, const Mat& src2) {
	Mat temp = Mat(Size(src1.size()), CV_32FC2);
	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++) {
			std::complex a(src1.at<Vec2f>(i, j)[0], src1.at<Vec2f>(i, j)[1]);
			std::complex b(src2.at<Vec2f>(i, j)[0], src2.at<Vec2f>(i, j)[1]);
			std::complex c = a * b;
			temp.at<Vec2f>(i, j)[0] = c.real();
			temp.at<Vec2f>(i, j)[1] = c.imag();
		}
	return temp;
}

// get the phase diff of two CV_32FC2 matrix
Mat polsar::getPhaseDiff(const Mat& hh, const Mat& vv) {
	Mat temp = getMul(hh, getConj(vv));
	return getComplexAngle(temp);
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
	A0 = covariance.at<Complexf>( 0, 0).re / 2.0;
	B0 = (covariance.at<Complexf>(1, 1)+covariance.at<Complexf>(2, 2)).re/ 2.0;
	B = covariance.at<Complexf>(1, 1).re - B0;
	C = covariance.at<Complexf>(0, 1).re;
	D = -1.0*covariance.at<Complexf>(0, 1).im;
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


void polsar::GetHuynenDecomp(const vector<Mat>& covariance, vector<Mat>& decomposition) {
	Mat A0, B0, B, C, D, E, F, G, H;
	extractChannel(covariance[0], A0, 0);
	A0 = A0 / 2.0;
    
	extractChannel(covariance[4]+ covariance[8], B0, 0);
	B0 =  B0 / 2.0;

	extractChannel(covariance[4], B, 0);
	B = B - B0;

	extractChannel(covariance[1], C, 0);

	extractChannel(covariance[1], D, 1);
	D = -1.0 * D;

	extractChannel(covariance[5], E, 0);

	extractChannel(covariance[5], F, 1);

	extractChannel(covariance[2], G, 1);

	extractChannel(covariance[2], H, 0);

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
	c11 = covariance.at<Complexf>(0, 0).re - fv * 3.0 / 8.0;
	c13Re = covariance.at<Complexf>(0, 2).re - fv / 8.0;
	c13Im = covariance.at<Complexf>(0, 2).im;
	c33 = covariance.at<Complexf>(2, 2).re - fv * 3.0 / 8.0;
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

		double a2 = c13Re * c13Re + c13Im * c13Im;
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
void polsar::cloudePottierDecomp(Mat_<Complexf>& coherency, vector<float>& result) {

	Eigen::Map<Eigen::Matrix<std::complex<float>, 3, 3, Eigen::RowMajor>> eigen_mat(coherency.ptr<std::complex<float>>(), coherency.rows, coherency.cols);
	Eigen::ComplexEigenSolver<Eigen::MatrixXcf> ces;
	ces.compute(eigen_mat);

	// sort eigen values in decreasing order, and record the original index
	vector<int> V = { 0,1,2 }; // indicating positions
	sort(V.begin(), V.end(), [&](int i, int j) {return ces.eigenvalues()[i].real() > ces.eigenvalues()[j].real(); });

	vector<float> realEigenValues(3);
	realEigenValues[0] = ces.eigenvalues()[V[0]].real();
	realEigenValues[1] = ces.eigenvalues()[V[1]].real();
	realEigenValues[2] = ces.eigenvalues()[V[2]].real();

	// extract the first component of each eigen vector sorted by eigen value decrease order
	vector<complex<float>> cos_alpha(3);
	cos_alpha[0] = ces.eigenvectors()(0, V[0]);
	cos_alpha[1] = ces.eigenvectors()(0, V[1]);
	cos_alpha[2] = ces.eigenvectors()(0, V[2]);

	// Entropy estimation
	float totalEigenValues = 0.0;
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
			plog[k] = 0.0;
		else
			plog[k] = -p[k] * log(p[k]) / log(3.0);
	}

	entropy = 0.0;
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
void polsar::yamaguchi4Decomp(const Mat_<Complexf>& coherency, const Mat_<Complexf>& covariance, vector<float>& result) {
	float ratio, d, cR, cI, c0, s, pd, pv, ps, pc, span, k1, k2, k3;

	span = coherency.at<Complexf>(0, 0).re + coherency.at<Complexf>(1, 1).re + coherency.at<Complexf>(2, 2).re;
	pc = 2.0 * std::abs(coherency.at<Complexf>(1,2).im);
	ratio = 10.0 * std::log10(covariance.at<Complexf>(2,2).re / covariance.at<Complexf>(0,0).re);

	if (ratio <= -2.0) {
		k1 = 1.0 / 6.0;
		k2 = 7.0 / 30.0;
		k3 = 4.0 / 15.0;
	}
	else if (ratio > 2.0) {
		k1 = -1.0 / 6.0;
		k2 = 7.0 / 30.0;
		k3 = 4.0 / 15.0;
	}
	else { // -2 < ratio <= 2
		k1 = 0.0;
		k2 = 1.0 / 4.0;
		k3 = 1.0 / 4.0;
	}

	pv = (coherency.at<Complexf>(2,2).re - 0.5 * pc) / k3;

	if (pv <= m_Epsilon) { // Freeman-Durden 3 component decomposition
		pc = 0.0f;
		freemanDurdenDecomp(covariance, result);
		result.push_back(pc);
	}
	else { // Yamaguchi 4 component decomposition

		s = coherency.at<Complexf>(0,0).re - 0.5f * pv;
		d = coherency.at<Complexf>(1,1).re - k2 * pv - 0.5f * pc;
		cR = coherency.at<Complexf>(0, 1).re - k1 * pv;
		cI = coherency.at<Complexf>(0, 1).im;

		if (pv + pc < span) {

			c0 = covariance.at<Complexf>(0,2).re - 0.5 * covariance.at<Complexf>(1, 1).re + 0.5 * pc;
			if (c0 < m_Epsilon) {
				ps = s - (cR * cR + cI * cI) / d;
				pd = d + (cR * cR + cI * cI) / d;
			}
			else {
				ps = s + (cR * cR + cI * cI) / s;
				pd = d - (cR * cR + cI * cI) / s;
			}

			if (ps > m_Epsilon && pd < m_Epsilon) {
				pd = 0.0;
				ps = span - pv - pc;
			}
			else if (ps < m_Epsilon && pd > m_Epsilon) {
				ps = 0.0;
				pd = span - pv - pc;
			}
			else if (ps < m_Epsilon && pd < m_Epsilon) {
				ps = 0.0;
				pd = 0.0;
				pv = span - pc;
			}

		}
		else {
			ps = 0.0;
			pd = 0.0;
			pv = span - pc;
		}
		result.push_back(ps);
		result.push_back(pd);
		result.push_back(pv);
		result.push_back(pc);
	}
}

void polsar::GetFreemanDurdenDecomp(const vector<Mat>& covariance, vector<Mat>& decomposition) {
	// restore the original coherecy matrix from the diagonal and the upper elements 
	Mat C00, C02, C11, C22;
	C00 = getComplexMat(covariance[0]);
	C02 = getComplexMat(covariance[2]);
	C11 = getComplexMat(covariance[4]);
	C22 = getComplexMat(covariance[8]);

	// record the result
	Mat Ps = Mat::zeros(Size(C00.size()), CV_32FC1);
	Mat Pd = Mat::zeros(Size(C00.size()), CV_32FC1);
	Mat Pv = Mat::zeros(Size(C00.size()), CV_32FC1);

	Mat_<Complexf>  m = Mat_<Complexf>(3, 3);
	for (int i = 0; i < C00.rows; i++) {
		for (int j = 0; j < C00.cols; j++) {
			m.at<Complexf>(0, 0) = C00.at<Complexf>(i, j);
			m.at<Complexf>(0, 2) = C11.at<Complexf>(i, j);
			m.at<Complexf>(1, 1) = C11.at<Complexf>(i, j);
			m.at<Complexf>(2, 2) = C22.at<Complexf>(i, j);
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

void polsar::GetCloudePottierDecomp(const vector<Mat>& coherency, vector<Mat>& decomposition) {
	// restore the original coherecy matrix from the diagonal and the upper elements 
	Mat T00, T01, T02, T10, T11, T12, T20, T21, T22;
	T00 = getComplexMat(coherency[0]);
	T01 = getComplexMat(coherency[1]);
	T02 = getComplexMat(coherency[2]);
	T10 = getComplexMat(coherency[3]);
	T11 = getComplexMat(coherency[4]);
	T12 = getComplexMat(coherency[5]);
	T20 = getComplexMat(coherency[6]);
	T21 = getComplexMat(coherency[7]);
	T22 = getComplexMat(coherency[8]);

	// record the result
	Mat H = Mat::zeros(Size(T00.size()), CV_32FC1);
	Mat A = Mat::zeros(Size(T00.size()), CV_32FC1);
	Mat Alpha = Mat::zeros(Size(T00.size()), CV_32FC1);

	Mat_<Complexf> t = Mat_<Complexf>(3, 3);
	for (int i = 0; i < T00.rows; i++) {
		for (int j = 0; j < T00.cols; j++) {
			t.at<Complexf>(0, 0) = T00.at<Complexf>(i, j);
			t.at<Complexf>(0, 1) = T01.at<Complexf>(i, j);
			t.at<Complexf>(0, 2) = T11.at<Complexf>(i, j);
			t.at<Complexf>(1, 0) = T10.at<Complexf>(i, j);
			t.at<Complexf>(1, 1) = T11.at<Complexf>(i, j);
			t.at<Complexf>(1, 2) = T12.at<Complexf>(i, j);
			t.at<Complexf>(2, 0) = T20.at<Complexf>(i, j);
			t.at<Complexf>(2, 1) = T21.at<Complexf>(i, j);
			t.at<Complexf>(2, 2) = T22.at<Complexf>(i, j);

			vector<float> result;
			cloudePottierDecomp(t, result);
			H.at<float>(i, j) = result[0];
			A.at<float>(i, j) = result[1];
			Alpha.at<float>(i, j) = result[2];
		}
	}
	decomposition.push_back(H);
	decomposition.push_back(A);
	decomposition.push_back(Alpha);

}

void polsar::GetYamaguchi4Decomp(const vector<Mat>& coherency, const vector<Mat> & covariance, vector<Mat>& decomposition) {
	// restore the original coherecy matrix from the diagonal and the upper elements 
	Mat T00, T01, T02, T10, T11, T12, T20, T21, T22;
	T00 = getComplexMat(coherency[0]);
	T01 = getComplexMat(coherency[1]);
	T02 = getComplexMat(coherency[2]);
	T10 = getComplexMat(coherency[3]);
	T11 = getComplexMat(coherency[4]);
	T12 = getComplexMat(coherency[5]);
	T20 = getComplexMat(coherency[6]);
	T21 = getComplexMat(coherency[7]);
	T22 = getComplexMat(coherency[8]);

	// restore the original coherecy matrix from the diagonal and the upper elements 
	Mat C00, C01, C02, C10, C11, C12, C20, C21, C22;
	C00 = getComplexMat(covariance[0]);
	C01 = getComplexMat(covariance[1]);
	C02 = getComplexMat(covariance[2]);
	C10 = getComplexMat(covariance[3]);
	C11 = getComplexMat(covariance[4]);
	C12 = getComplexMat(covariance[5]);
	C20 = getComplexMat(covariance[6]);
	C21 = getComplexMat(covariance[7]);
	C22 = getComplexMat(covariance[8]);

	// record the result
	Mat Ps = Mat::zeros(Size(T00.size()), CV_32FC1);
	Mat Pd= Mat::zeros(Size(T00.size()), CV_32FC1);
	Mat Pv = Mat::zeros(Size(T00.size()), CV_32FC1);
	Mat Pc = Mat::zeros(Size(T00.size()), CV_32FC1);

	Mat_<Complexf> t = Mat_<Complexf>(3, 3);
	Mat_<Complexf> c = Mat_<Complexf>(3, 3);
	for (int i = 0; i < T00.rows; i++) {
		for (int j = 0; j < T00.cols; j++) {
			t.at<Complexf>(0, 0) = T00.at<Complexf>(i, j);
			t.at<Complexf>(0, 1) = T01.at<Complexf>(i, j);
			t.at<Complexf>(0, 2) = T11.at<Complexf>(i, j);
			t.at<Complexf>(1, 0) = T10.at<Complexf>(i, j);
			t.at<Complexf>(1, 1) = T11.at<Complexf>(i, j);
			t.at<Complexf>(1, 2) = T12.at<Complexf>(i, j);
			t.at<Complexf>(2, 0) = T20.at<Complexf>(i, j);
			t.at<Complexf>(2, 1) = T21.at<Complexf>(i, j);
			t.at<Complexf>(2, 2) = T22.at<Complexf>(i, j);

			c.at<Complexf>(0, 0) = C00.at<Complexf>(i, j);
			c.at<Complexf>(0, 1) = C01.at<Complexf>(i, j);
			c.at<Complexf>(0, 2) = C11.at<Complexf>(i, j);
			c.at<Complexf>(1, 0) = C10.at<Complexf>(i, j);
			c.at<Complexf>(1, 1) = C11.at<Complexf>(i, j);
			c.at<Complexf>(1, 2) = C12.at<Complexf>(i, j);
			c.at<Complexf>(2, 0) = C20.at<Complexf>(i, j);
			c.at<Complexf>(2, 1) = C21.at<Complexf>(i, j);
			c.at<Complexf>(2, 2) = C22.at<Complexf>(i, j);

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


void polsar::GetCoherencyT(const vector<Mat>& pauli, vector<Mat>& coherency, int winSize) { vec2mat(pauli, coherency, winSize); }

void polsar::GetCovarianceC(const vector<Mat>& lexi, vector<Mat>& covariance, int winSize) { vec2mat(lexi, covariance, winSize); }