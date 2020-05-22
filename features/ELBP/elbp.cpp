#include "opencv2/core/core.hpp"
#include "elbp.hpp"
#include <string>
#include <iostream>

//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
using namespace cv;
using namespace std;

template <typename _Tp>  void elbp::elbp_(const Mat& src, Mat &dst, int radius, int neighbors) 
{
    // allocate memory for result, set to zero
    dst = Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);

    for (int n = 0; n < neighbors; n++) {
        // sample points
        float x = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        float y = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        // iterate through your data
        for (int i = radius; i < src.rows - radius; i++) {
            for (int j = radius; j < src.cols - radius; j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1 * src.at<_Tp>(i + fy, j + fx) + w2 * src.at<_Tp>(i + fy, j + cx) + w3 * src.at<_Tp>(i + cy, j + fx) + w4 * src.at<_Tp>(i + cy, j + cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) || (std::abs(t - src.at<_Tp>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

void elbp::elbp(const Mat& src, Mat & dst, int radius, int neighbors)
{
    std::string type = elbp::type2str(src.type());
    if (type == "CV_8SC1") { elbp::elbp_<char>(src, dst, radius, neighbors); }
    else if (type == "CV_8UC1") { elbp::elbp_<unsigned char>(src, dst, radius, neighbors); }
    else if (type == "CV_16SC1") { elbp::elbp_<short>(src, dst, radius, neighbors); }
    else if (type == "CV_16UC1") { elbp::elbp_<unsigned short>(src, dst, radius, neighbors); }
    else if (type == "CV_32SC1") { elbp::elbp_<int>(src, dst, radius, neighbors); }
    else if (type == "CV_32FC1") { elbp::elbp_<float>(src, dst, radius, neighbors); }
    else   { elbp::elbp_<double>(src, dst, radius, neighbors); }
    
}

Mat elbp::elbp(const Mat& src, int radius, int neighbors, bool normed) {
    Mat dst;
    if (src.channels() > 1)
    {
        Mat tmp;
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
        elbp::elbp(tmp, dst, radius, neighbors);
    }
    else {
        elbp::elbp(src, dst, radius, neighbors);
    }
    
    if (normed) {
        cv::normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    }
    return dst;
}


Mat elbp::histc(const Mat& src, int minVal, int maxVal , bool normed )
{
    Mat result;
    // Establish the number of bins.
    int histSize = maxVal - minVal + 1;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
    const float* histRange = {  range };
    // calc histogram
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
    // normalize
    if (normed) {
        result /= (int)src.total();
    }
    return result.reshape(1, 1);
}


Mat elbp::spatial_histogram(const Mat &src, int numPatterns,
    int grid_x, int grid_y, bool normed )
{
    // calculate LBP patch size
    int width = src.cols / grid_x;
    int height = src.rows / grid_y;
    // allocate memory for the spatial histogram
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // return matrix with zeros if no data was given
    if (src.empty())
        return result.reshape(1, 1);
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    
    for (int i = 0; i < grid_y; i++) {
        for (int j = 0; j < grid_x; j++) {
            Mat src_cell = Mat(src, Range(i * height, (i + 1) * height), Range(j * width, (j + 1) * width));
            Mat cell_hist = elbp::histc(src_cell, 0, (numPatterns - 1), normed);
            // copy to the result matrix
            Mat hist = result.row(resultRowIdx);
            cell_hist.reshape(1, 1).convertTo(hist, CV_32FC1);
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return result.reshape(1, 1);
}


 
std::string elbp::type2str(int type) {
    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch (depth) {
    case CV_8U:  r = "CV_8U"; break;
    case CV_8S:  r = "CV_8S"; break;
    case CV_16U: r = "CV_16U"; break;
    case CV_16S: r = "CV_16S"; break;
    case CV_32S: r = "CV_32S"; break;
    case CV_32F: r = "CV_32F"; break;
    case CV_64F: r = "CV_64F"; break;
    default:     r = "User"; break;
    }
    r += "C";
    r += (chans + '0');
    return r;
}

