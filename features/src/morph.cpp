#include "morph.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;

auto* morph::matToArray(const Mat& image) {
    Mat output;
    if (image.channels() != 1) {
        cvtColor(image, output, COLOR_BGR2GRAY);
    }
    else {
        output = image.clone();
    }
   int totalElements = static_cast<int>(output.total()) * output.channels();
    if (!output.isContinuous()) {
        output = output.clone();
    }

    // flatten the Mat.
    Mat flat = output.reshape(1, totalElements);
    unsigned char* ptr = new uchar[totalElements];
    for (int i = 0; i < totalElements; i++) {
        ptr[i] = flat.at<uchar>(i, 0);
    }
    return ptr;
}

// compute opening-closing by reconstruction from image
// example:https://de.Mathworks.com/help/images/marker-controlled-watershed-segmentation.html
Mat morph::CaculateMP(const Mat& src, int morph_size) {
    //convert img to grayscale
    Mat dst;
    if (src.channels() != 1) {
        cvtColor(src, dst, COLOR_BGR2GRAY);
    }
    else {
        src.copyTo(dst);
        dst.convertTo(dst, CV_8UC1);
    }

    Mat element = getStructuringElement(MORPH_RECT, cv::Size(morph_size ,morph_size ));

    //openning
    Mat open;
    cv::morphologyEx(dst, open, cv::MORPH_OPEN, element);
    //cv::imwrite("opening.png", open);

    //erode and reconstruct ( opening-by-reconstruction )
    Mat erosion = Mat(Size(dst.size()), dst.type());
    erode(dst, erosion, element);
    Mat  Iobr = imReconstruct(erosion, dst);
    //cv::imwrite("opening_by_reconstruction.png", Iobr);

    //closing
    Mat close;
    cv::morphologyEx(dst, close, cv::MORPH_CLOSE, element);
    //cv::imwrite("closing.png", close);

     //closing-by-Reconstruction
    Mat dilation = Mat(Size(dst.size()), dst.type());
    dilate(dst, dilation,element);
    Mat Icbr = imReconstruct(255-dilation, 255-dst);
    Icbr = 255 - Icbr;
    //cv::imwrite("closing_by_reconstruction.png", Icbr);

    Mat output;
    output.push_back(open);
    output.push_back(Iobr);
    output.push_back(close);
    output.push_back(Icbr);

    if (!output.isContinuous()) {
        output = output.clone();
    }
    return output;
}

cv::Mat morph::imReconstruct(const cv::Mat& marker, const cv::Mat& mask) {
    
    Mat output;
    if(marker.size() == mask.size()){
        auto* ptr_marker = matToArray(marker);
        auto* ptr_mask = matToArray(mask);
        Reconstruct(ptr_marker, ptr_mask, mask.cols, mask.rows);
        //restore cv Mat
        output = Mat(mask.rows, mask.cols, mask.type(), ptr_marker);
    }
    return output;
}

cv::Mat morph::imRegionalMax(const cv::Mat& src) {
    Mat temp,output;
    if (src.channels() != 1) {
        cvtColor(src, temp, COLOR_BGR2GRAY);
    }
    else {
        src.copyTo(temp);
        temp.convertTo(temp, CV_8UC1);
    }
    bool* ptr_output = new bool[temp.total()];
    auto* ptr_src = matToArray(temp);
    int mean = cv::mean(temp)[0];
    Regmax(ptr_src,ptr_output,src.cols,src.rows);
    output = Mat(src.rows, src.cols, CV_8UC1, ptr_output);
    return output;
}

