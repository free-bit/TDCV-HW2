
#include <opencv2/opencv.hpp>

#include "HOGDescriptor.h"

int main(){ 
	cv::Mat im = cv::imread("../../data/task1/obj1000.jpg");

	if (im.empty())
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	HOGDescriptor hog;
	std::vector<float> feat;
	
	// Set parameters:
	hog.setWinSize(cv::Size(64, 64));
	hog.setBlockSize(cv::Size(16, 16));
	hog.setBlockStep(cv::Size(8, 8));
	hog.setCellSize(cv::Size(8, 8));
	hog.setWinStride(cv::Size(0, 0));
	hog.setPadSize(cv::Size(0, 0));

	hog.printParams();

	hog.detectHOGDescriptor(im, feat, true);

	return 0;
}