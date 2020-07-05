#pragma once
#ifndef NCA_HPP
#define NCA_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <vector>

// dimension reduction using Neighbourhood Components Analysis
//reference: https://kevinzakka.github.io/2020/02/10/nca/
void nearest_neighbors(const std::vector<cv::Mat>& input, const std::vector<unsigned char>& label) {
    unsigned int correct = 0;

    for (unsigned int i = 0; i < input.size(); ++i) {
        double min_norm = std::numeric_limits<double>::infinity();
        unsigned char min_norm_label;
        for (unsigned int j = 0; j < input.size(); ++j) {
            if (i == j) continue;
            cv::Mat a = input[i] - input[j];
            double norm = std::sqrt(a.dot(a));
            if (norm < min_norm) {
                min_norm = norm;
                min_norm_label = label[j];
            }
        }

        if (label[i] == min_norm_label) {
            ++correct;
        }
    }

    std::cout << "Got " << correct << " correct out of " << input.size() << std::endl;
}

// vector<Mat> input: mat size(1,cols)
cv::Mat scaling_matrix(const std::vector<cv::Mat>& input,int dimA) {
    cv::Mat A;
    if (input.empty()) return A;

    int size = input[0].total();

    std::vector< std::pair<float, float> > minmax(size, std::make_pair(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()));
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < size; ++j) {
            cv::Mat temp;
            input[i].convertTo(temp, CV_32FC1);
            float val = temp.at<float>(0,j);
            if (val < minmax[j].first) {
                minmax[j] = std::make_pair(val, minmax[j].second);
            }
            if (val > minmax[j].second) {
                minmax[j] = std::make_pair(minmax[j].first, val);
            }
        }
    }

    A = cv::Mat::zeros(size, dimA, CV_32FC1);
    for (unsigned int i = 0; i < size; ++i) {
        for(int j =0; j< dimA;j++){
            A.at<float>(i, j) = 1.0 / (minmax[i].second - minmax[i].first);
        }
    }

    return A;
}

std::vector<cv::Mat> scale(const cv::Mat& ScaleA, const std::vector<cv::Mat>& input) {
    std::vector<cv::Mat> scaled_input;
    for (const auto & i:input) {
        cv::Mat temp;
        i.convertTo(temp, CV_32FC1);
        scaled_input.push_back(temp * ScaleA);
    }

    return scaled_input;
}

cv::Mat neighborhood_components_analysis(const std::vector<cv::Mat>& input, const std::vector<unsigned char>& label, const cv::Mat& init, unsigned int iterations, float learning_rate) {
    cv::Mat A = init;// A.rows == input[0].cols
    for (unsigned int it = 0; it < iterations; ++it) {
        unsigned int i = it % input.size();

        float softmax_normalization = 0.0f;
        for (unsigned int k = 0; k < input.size(); ++k) {
            if (k == i) continue;
            cv::Mat temp1,temp2;
            input[i].convertTo(temp1, CV_32FC1);
            input[k].convertTo(temp2, CV_32FC1);
            cv::Mat temp = temp1 *A - temp2 *A;
            softmax_normalization += std::exp(-temp.dot(temp));
        }

        std::vector<float> softmax;
        for (unsigned int k = 0; k < input.size(); ++k) {
            if (k == i) softmax.push_back(0.0);
            else {
                cv::Mat temp1, temp2;
                input[i].convertTo(temp1, CV_32FC1);
                input[k].convertTo(temp2, CV_32FC1);
                cv::Mat temp = temp1 * A - temp2 * A;
                softmax.push_back(std::exp(-temp.dot(temp)) / softmax_normalization);
            }
        }

        float p = 0.0f;
        for (unsigned int k = 0; k < softmax.size(); ++k) {
            if (label[k] == label[i]) p += softmax[k];
        }

        cv::Mat first_term = cv::Mat::zeros(input[0].total(), input[0].total(), CV_32FC1);
        cv::Mat second_term = cv::Mat::zeros(input[0].total(), input[0].total(), CV_32FC1);
        for (unsigned int k = 0; k < input.size(); ++k) {
            if (k == i) continue;
            cv::Mat xik = input[i] - input[k];
            xik.convertTo(xik, CV_32FC1);
            cv::Mat temp = xik.t() * xik;
            temp.convertTo(temp, CV_32FC1);
            cv::Mat term = softmax[k] * temp;

            first_term += term;
            if (label[k] == label[i]) { second_term += term;}
        }
        first_term *= p;

        A += learning_rate *  (first_term - second_term) * A;
    }

    return A;
}

#endif // NCA_HPP