/*Implementing complete processing chain*
*step 1: Read data - from oberpfaffenhofen - rat, label, image
*step 2: Simple feature extraction
*step 3: Train a classifier (KNN/RF/CNN)
*step 4: Apply trained classifier to test data
*step 5: Visualize - PCA/tSNE? and Evaluate data
*/

#include <iostream>
#include <opencv2/opencv.hpp>

#include "Data.h"
#include "Training.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	cout << "In main!!" << endl;
	int numOfClasses;
	numOfClasses = 10;
	//Object of class
	Data data;
	//load RAT
	data.loadData(argv[1]);
	cout << "Data loaded" << endl;
	//load RGB image
	data.loadImage(argv[2]);
	cout << "Image loaded" << endl;
	//load labels
	data.loadLabels(argv[3], numOfClasses);
	cout << "Labels loaded" << endl;
	waitKey(0);
	return 0;	
}


