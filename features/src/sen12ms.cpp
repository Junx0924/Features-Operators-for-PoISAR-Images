#include "opencv2/core/core.hpp"
#include "sen12ms.hpp"
#include "Geotiff.cpp"
#include <string>
#include <iostream>
#include <fstream>

 
std::array<unsigned char, 17>  IGBP_label = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 };
// LCCS_LC, LCCS_LU,LCCS_SH merged into LCCS
std::array<unsigned char, 26>  LCCS_label = { 11,12,13,14,15,16,10,21,22,30,31,32,40,41,43,42,27,50,36,9,25,35,51,2,1,3 };

/*===================================================================
 * Function: GetFalseColorImage
 *
 * Summary:
 *   Generate false color image from SAR data;
 *
 * Arguments:
 *   Mat src - 2 channel matrix(values in dB) from tiff file
 *   bool normed - normalized to 0-255 
 *
 * Returns:
 *   3 channel matrix: R: VV, G:VH, B: VV/VH
=====================================================================
*/
Mat sen12ms::GetFalseColorImage(const Mat& src, bool normed) {
    vector<Mat>  Channels;
    split(src, Channels);

    Mat R = cv::abs(Channels[0]); //VV
    Mat G = cv::abs(Channels[1]);  //VH
    
    

    Mat B = Mat::zeros(src.rows, src.cols, CV_32FC1); //VV/VH
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (Channels[1].at<float>(i, j) != 0) {
                B.at<float>(i, j) =  Channels[0].at<float>(i, j)  /  Channels[1].at<float>(i, j) ;
            }
        }
    }
    Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC3);
    vector<Mat> temp;
    temp.push_back(B);
    temp.push_back(G);
    temp.push_back(R);
    merge(temp, dst);
    if (normed) {
        cv::normalize(dst, dst, 0, 255, NORM_MINMAX);
        dst.convertTo(dst, CV_8UC3);
    }
    return dst;
}


/*===================================================================
 * Function: GetLabelClass
 *
 * Summary:
 *   Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
 *   Generate IGBP, LCCS from the ground truth 
 *
 * Arguments:
 *   Mat src - 4 channel matrix from groundtruth file
 *   Mat& IGBP - Destination Mat for IGBP class
 *   Mat& LCCS - Destination Mat for LCCS class
 *
 * Returns:
 *   3 channel matrix: R: VV, G:VH, B: VV/VH
=====================================================================
*/
void sen12ms::GetClassCategory(const Mat& lc, Mat& IGBP, Mat& LCCS) {
   
    vector<Mat> temp(lc.channels());
    split(lc, temp);

    IGBP = temp[0];
    Mat LCCS_LC = temp[1];
    Mat LCCS_LU = temp[2];
    Mat LCCS_SH = temp[3];

   
    for (int i = 0; i < lc.rows; i++) {
        for (int j = 0; j < lc.cols; j++) {
            if (LCCS_LC.at<unsigned char>(i, j) != 0) {
                LCCS.at<unsigned char>(i, j) = LCCS_LC.at<unsigned char>(i, j);
            }
            else if (LCCS_LU.at<unsigned char>(i, j) != 0) {
                LCCS.at<unsigned char>(i, j) = LCCS_LU.at<unsigned char>(i, j);
            }
            else {
                LCCS.at<unsigned char>(i, j) = LCCS_SH.at<unsigned char>(i, j);
            }
        }
    }
}
 
/*===================================================================
 * Function: GetPolStatistic
 *
 * Summary:
 *   Compute min, max, mean, std, median of mask area
 *
 * Arguments:
 *   Mat src - matrix of PolSAR data ( VV, VH for each channel)
 *   const Mat& mask - single channel mask matrix 
 *
 * Returns:
 *   Single channel Mat of Size(src.channels(), 5)
=====================================================================
*/
Mat sen12ms::GetPolStatistic(const Mat& src, const Mat& mask) {

    Mat stat;
    
    for (int i =0; i<src.channels(); i++){
        Mat result= Mat(1, 5, CV_32FC1);

        Mat src_temp = Mat(src.rows, src.cols, CV_32FC1);
        extractChannel(src, src_temp, i);
       
        // put the mask area into a vector
        vector<float> srcWithMask;
        for (int x = 0; x < src.rows; x++) {
            for (int y = 0; y < src.cols; y++) {
                if (mask.at<unsigned char>(x, y) == 0) {
                    continue;
                }
                else {
                    srcWithMask.push_back(src_temp.at<float>(x, y));

                }
            }
        }
        //median
        int size = srcWithMask.size();
        std::sort(srcWithMask.begin(), srcWithMask.end());
        if (size % 2 == 0)
        {
            result.at<float>(0,4) = (srcWithMask[size / 2 - 1] + srcWithMask[size / 2]) / 2;
        }
        else
        {
            result.at<float>(0,4) = srcWithMask[size / 2];
        }

        double min,  max;
        cv::minMaxIdx(srcWithMask, &min, &max);

        //min
        result.at<float>(0,0) = srcWithMask[0];
        //max
        result.at<float>(0, 1) = srcWithMask[size-1];

        cv::Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
        cv::meanStdDev(srcWithMask, mean, stddev);
        //mean
        result.at<float>(0, 2) = mean[0];
        //stddev
        result.at<float>(0, 3) = stddev[0];

        stat.push_back(result);
    }
    return stat;
}

/*===================================================================
 * Function: GetMask
 *
 * Summary:
 *   Create Masks for each class category
 *
 * Arguments:
 *   Mat & lc - 4 channel matrix from groundtruth file
 *   vector<Mat> &IGBP_mask - Destination Mask Mat for IGBP class
 *   vector<Mat> &LCCS_mask - Destination Mask Mat for  LCCS class
 *   vector<unsigned char>IGBP_list  - record each IGBP class value in the masks  
 *   vector<unsigned char> LCCS_list - record each LCCS class value in the masks  

 * Returns:
 *  void
=====================================================================
*/
void sen12ms::GetMask(const Mat& lc, vector<Mat>& list_masks,  vector<unsigned char> &list_classValue, MaskType mask_type) {
    Mat igbp = Mat(lc.rows, lc.cols, CV_8UC1);
    Mat lccs = Mat(lc.rows, lc.cols, CV_8UC1);
    // merge different LCCS class channels to one channel
    GetClassCategory(lc, igbp, lccs);
    
    //get IGBP mask
    if(mask_type ==MaskType::IGBP){
        for (int i = 0; i < IGBP_label.size(); i++) {
            vector<std::pair<int, int> > ind;
            if (FindLandClass(igbp, ind, IGBP_label[i])) {
                list_classValue.push_back(IGBP_label[i]);
                Mat tmp = Mat::zeros(lc.rows, lc.cols, CV_8UC1);
                for (auto const& p : ind) {
                    tmp.at<unsigned char>(p.first, p.second) = IGBP_label[i];
                }
                list_masks.push_back(tmp);
            }
        }
    }
    // get LCCS_mask
    if (mask_type == MaskType::IGBP) {
        for (int i = 0; i < LCCS_label.size(); i++) {
            vector<std::pair<int, int> > ind;
            if (FindLandClass(lccs, ind, LCCS_label[i])) {
                list_classValue.push_back(LCCS_label[i]);
                Mat tmp = Mat::zeros(lc.rows, lc.cols, CV_8UC1);
                for (auto const& p : ind) {
                    tmp.at<unsigned char>(p.first, p.second) = LCCS_label[i];
                }
                list_masks.push_back(tmp);
            }
        }
    }
}

/*===================================================================
 * Function: FindLandClass
 *
 * Summary:
 *   check if cetain class type existed in a class category
 *
 * Arguments:
 *   Mat & src - IGBP mat or LCCS mat
 *   vector<std::pair<int, int> > &ind - record the index of the class type
 *   const int &landclass: value of the class type

 * Returns:
 *  bool
=====================================================================
*/
 bool sen12ms::FindLandClass(const Mat& src, vector<std::pair<int, int> > &ind, const unsigned char&landclass) {
     bool flag = false;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<unsigned char>(i,j) == landclass) {
                ind.push_back(std::make_pair(i, j));
                flag = true;
            }
        }
    }
    return flag;
}

 /*===================================================================
 * Function: GetHistWithMask
 *
 * Summary:
 *   Caculate the historgram vector of a mat with mask
 *
 * Arguments:
 *   Mat & src - IGBP mat or LCCS mat
 *   const Mat& mask -  single channel mask matrix 
 *   int minVal - the min of bin boundaries
 *   int maxVal - the max of bin boundaries
 *   int histSize 
 *   bool normed - normalized to make the sum become 1
 * Returns:
 *  Mat of Size(1,histSize)
=====================================================================
*/
 Mat sen12ms::GetHistWithMask(const Mat& src, const Mat& mask, int minVal, int maxVal, int histSize, bool normed)
 {
     Mat result;
     // Set the ranges.
     float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
     const float* histRange = { range };
     // calc histogram with mask
     calcHist(&src, 1, 0, mask, result, 1, &histSize, &histRange, true, false);
     // normalize
     if (normed) {
         result /= (int)src.total();
     }
     return result.reshape(1, 1);
 }

/*===================================================================
 * Function: GeneratePNG
 *
 * Summary:
 *   convert tiff files to png format for images and masks
 *   the list of RGB images are in list_images.txt
 *   the list of IGBP masks are in list_IGBP_masks.txt
 *   the list of LCCS masks are in list_LCCS_masks.txt
 *
 * Arguments:
 *   string outputpath - the folder path to store all the pngs
 *   MaskType mask_type - IGBP or LCCS
 * Returns:
 *  void
=====================================================================
*/
 void sen12ms::GeneratePNG(const string &outputpath, MaskType mask_type) {
     
     fstream list_images;
     string list_images_path = outputpath + "\\" + "list_images.txt";
     list_images.open(list_images_path, ios::out | ios::trunc);

     fstream listofmasks;

     if(mask_type ==MaskType::IGBP){
     string list_IGBP_path = outputpath + "\\" + "list_IGBP_masks.txt";
     listofmasks.open(list_IGBP_path, ios::out | ios::trunc);
     } else if(mask_type == MaskType::LCCS) {
         string list_LCCS_path = outputpath + "\\" + "list_LCCS_masks.txt";
         listofmasks.open(list_LCCS_path, ios::out | ios::trunc);
     }

     if (list_images.is_open()) {
         for (const auto &tp: s1FileList) {
                 if (tp.empty()) { cout << "empty line find" << endl; break; }
                 //get the filename from path without extension
                 string base_filename = tp.substr(tp.find_last_of("/\\") + 1);
                 size_t position = base_filename.find(".");
                 string fileName = (string::npos == position) ? base_filename : base_filename.substr(0, position);

                 Mat patch =  ReadTiff(tp);
                 Mat colorImg =  GetFalseColorImage(patch, true);
                 string outputpng = outputpath + "\\" + fileName + ".png";
                 imwrite(outputpng, colorImg);
                 list_images << outputpng << endl;
         }
         list_images.close();
     }


     if (listofmasks.is_open()) {
         for (const auto &tp: lcFileList) {
                 if (tp.empty()) { cout << "empty line find" << endl; break; }
                 //get the filename from path without extension
                 string base_filename = tp.substr(tp.find_last_of("/\\") + 1);
                 size_t position = base_filename.find(".");
                 string fileName = (string::npos == position) ? base_filename : base_filename.substr(0, position);
                  
                 //Replace String In Place
                 size_t pos = 0;
                 string search = "_lc_", replace = "_s1_";
                 while ((pos = fileName.find(search, pos)) != std::string::npos) {
                     fileName.replace(pos, search.length(), replace);
                     pos += replace.length();
                 }

                 Mat lc_mat =  ReadTiff(fileName);

                 vector<Mat> list_mask;
                 vector<unsigned char> list_classValue; //store the class value
                  GetMask(lc_mat, list_mask, list_classValue, mask_type);
                 string outputpng;
                 string outputFolder;
                 if (mask_type != MaskType::LCCS) {
                     outputFolder = outputpath + "\\" + fileName + "_IGBP";
                 }
                 else {
                     outputFolder = outputpath + "\\" + fileName + "_LCCS";
                 }
                  
                 int status = _mkdir(outputFolder.c_str());
                 if (status < 0) {
                     cout << "failed to create LCCS mask folder for p" << fileName << endl;
                     break;
                 }
                 else {
                     for (int j = 0; j < list_classValue.size(); j++) {
                         outputpng = outputFolder + "\\" + to_string(list_classValue[j]) + ".png";
                         imwrite(outputpng, list_mask[j]);
                         listofmasks << outputpng << endl;
                     }
                 }
         }
         listofmasks.close();
     }
     cout << "Generate PNG files done." << endl;
 }

 


 /*===================================================================
 * Function: LoadDataToMemeory
 *
 * Summary:
 *   load tiff files to vector<Mat> list_images, vector<vector<Mat>> list_masks,vector<vector<unsigned char>>list_classValue
 *
 * Arguments:
 *   int BatchSize  
 *   MaskType mask_type - IGBP or LCCS
 *   
 * Returns:
 *  void
=====================================================================
*/
 void sen12ms::LoadDataToMemeory(int BatchSize, MaskType mask_type) {

     if (BatchSize > s1FileList.size()) { 
         cout << "BatchSize must be smaller than " << s1FileList.size() << endl;
         exit(-1); 
     }
     
     for (auto const& s1File : s1FileList) {
         Mat s1_mat =  ReadTiff(s1File);
         Mat unnomarlizedImg = GetFalseColorImage(s1_mat, false);
         list_images->push_back(unnomarlizedImg);
         if (list_images->size() >= BatchSize) { break; }
     }
     
     for (auto const& lcFile : lcFileList) {
         Mat lc_mat =  ReadTiff(lcFile);
         vector<Mat> masks;
         vector<unsigned char> classValue; //store the class value
          GetMask(lc_mat, masks, classValue, mask_type);
          list_masks->push_back(masks);
          list_classValue->push_back(classValue);
          if (list_masks->size() >= BatchSize) { break; }
     }
     cout << "Load " << list_images->size() << " images and its masks to memory" << endl;
 }


 /*===================================================================
 * Function: ProcessData
 *
 * Summary:
 *   process data to fee torch dataset
 *
 * Arguments:
 *   vector<Mat>& imageOfMaskArea - stores the image pixels within mask area
 *   vector<unsigned char>& classValue  - store the class type for each image
 *
 * Returns:
 *  void
=====================================================================
*/
 void sen12ms::ProcessData(vector<Mat>& imageOfMaskArea, vector<unsigned char>& classValue) {
     int count = 0;
    if (list_images->empty()) { cout << "Please load data to memory first! " << endl;  exit(-1); }
     for (int i = 0; i < list_images->size(); i++) {
         Mat temp = list_images->at(i);
         Mat output = Mat(Size(temp.size()), CV_32FC3);
         for(int j=0;j< list_masks->at(i).size();j++){
             Mat mask = list_masks->at(i)[j];
             unsigned char class_value = list_classValue->at(i)[j];
             bitwise_and(temp, temp, output, mask);

             //for testing
             //normalize(mask, mask, 0, 255, NORM_MINMAX);
             //cout << "class value: " << to_string(class_value) << "," << GetClassName(class_value, MaskType::IGBP) << endl;
             //imshow("mask ", mask);
             //waitKey(0);
             //imshow("image of mask area:", output);
             //waitKey(0);

             imageOfMaskArea.push_back(output);
             classValue.push_back(class_value);
         }
     }
 }

 /*===================================================================
 * Function: LoadFileList
 *
 * Summary:
 *   Load tiff file list
 *
 * Arguments:
 *   string s1FileList - the txt file of all s1 files path
 *   string lcFileList - the txt file list of all lc files path
 *
 * Returns:
 *  void
=====================================================================
*/
 void sen12ms::LoadFileList(const string& s1FileListPath, const string& lcFileListPath) {
     fstream s1File;
     s1File.open(s1FileListPath, ios::in);

     std::ifstream lcFile(lcFileListPath);
     const std::string lcFileString((std::istreambuf_iterator<char>(lcFile)), std::istreambuf_iterator<char>());


     if (s1File.is_open()) {
         string s1FilePath;
         while (getline(s1File, s1FilePath)) {
             size_t pos = 0;
             if (s1FilePath.empty()) { cout << "empty line find" << endl; break; }
             // if the path is not tiff file
             if (s1FilePath.find("tif", pos) == std::string::npos) { continue; }
              
             // to make sure each s1File is corresponding to lcFile
             // get the lc filename, only need to replace all the _s1_ to _lc_
             string lcFileName = s1FilePath.substr(s1FilePath.find_last_of("/\\") + 1);
             //Replace replace all the _s1_ to _lc
             string search = "_s1_", replace = "_lc_";
             while ((pos = lcFileName.find(search, pos)) != std::string::npos) {
                 lcFileName.replace(pos, search.length(), replace);
                 pos += replace.length();
             }

             // get the absolute file path of this lc file from lcFileString
             string tmp = lcFileString.substr(0, lcFileString.find(lcFileName));
             string lcFilePath = tmp.substr(tmp.find_last_of("\n") + 1) + lcFileName;
             if (!lcFilePath.empty()) {
                 s1FileList.push_back(s1FilePath);
                 lcFileList.push_back(lcFilePath);
             }
         }
         s1File.close();
     }
     cout << "list size of s1 files: " << s1FileList.size() << endl;
     cout << "list size of lc files: " << lcFileList.size() << endl;
 }


 /*===================================================================
 * Function: GetClassName
 *
 * Summary:
 *  get the class name from class value
 *
 * Arguments:
 *   int classValue 
 *   MaskType mask_type: choose in IGBP or LCCS
 *
 * Returns:
 *  void
=====================================================================
*/
 string sen12ms::GetClassName(unsigned char classValue, MaskType mask_type){
     string class_name;
     std::map<unsigned char, string> IGBP = {
       {1,"Evergreen Needleleaf Forests"},
{2,"Evergreen Broadleaf Forests"},
{3,"Deciduous Needleleaf Forests"},
{4,"Deciduous Broadleaf Forests"},
{5,"Mixed Forests"},
{6,"Closed (Dense) Shrublands"},
{7,"Open (Sparse) Shrublands"},
{8,"Woody Savannas"},
{9,"Savannas"},
{10,"Grasslands "},
{11,"Permanent Wetlands"},
{12,"Croplands"},
{13,"Urban and Built-Up Lands"},
{14,"Cropland/Natural Vegetation Mosaics"},
{15,"Permanent Snow and Ice"},
{16,"Barren"},
{17,"Water Bodies"}
     };

     std::map<unsigned char, string> LCCS = {
{1,"Barren"},
{2,"Permanent Snow and Ice"},
{3,"Water Bodies"},
{9,"Urban and Built-Up Lands"},
{10,"Dense Forests"},
{11,"Evergreen Needleleaf Forests"},
{12,"Evergreen Broadleaf Forests"},
{13,"Deciduous Needleleaf Forests"},
{14,"Deciduous Broadleaf Forests"},
{15,"Mixed Broadleaf/Needleleaf Forests"},
{16,"Mixed Broadleaf Evergreen/Deciduous Forests"},
{21,"Open Forests "},
{22,"Sparse Forests"},
{25,"Forest/Cropland Mosaics"},
{27,"Woody Wetlands"},
{30,"Natural Herbaceous"},
{30,"Grasslands"},
{31,"Dense Herbaceous"},
{32,"Sparse Herbaceous"},
{35,"Natural Herbaceous/Croplands Mosaics"},
{36,"Herbaceous Croplands"},
{40,"Shrublands"},
{41,"Closed (Dense) Shrublands"},
{42,"Shrubland/Grassland Mosaics"},
{43,"Open (Sparse) Shrublands"},
{50,"Herbaceous Wetlands"},
{51,"Tundra"}
     };
 
     if (mask_type == MaskType::IGBP) {
         class_name = IGBP[classValue];
     }
     else {
         class_name = LCCS[classValue];
     }
     return class_name;
 }

 Mat sen12ms::ReadTiff(string filepath) {
     const char* file = filepath.c_str();
     GeoTiff* tiff = new GeoTiff(file);
     Mat tiff_mat = tiff->GetMat().clone();
     delete tiff;
     return tiff_mat;
 }