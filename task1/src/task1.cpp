
#include <opencv2/opencv.hpp>

#include "HOGDescriptor.h"





int main(){ 
	cv::Mat im = cv::imread("../../data/task1/obj1000.jpg");

	if (im.empty())
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//Fill Code here
	/*int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	//Prepare windows
	namedWindow("Original", WINDOW_AUTOSIZE);
	namedWindow("Rotation", WINDOW_AUTOSIZE);
	namedWindow("Padding", WINDOW_AUTOSIZE);
	namedWindow("Sobel", CV_WINDOW_AUTOSIZE);

	//Initialize matrices for image manipulations
	Mat im_gray;
	Mat grad;
	Mat im_rot;
	Mat im_pad;

	//90 degree rotation
	rotate(im, im_rot, ROTATE_90_CLOCKWISE);

	//Padding with size 50px
	int border = 50;
	copyMakeBorder(im, im_pad, border, border,border, border, BORDER_REPLICATE);

	//Compute Gradients
	//Apply Gaussian 3x3 Kernel
	GaussianBlur(im, im, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//Convert to gray
	cvtColor(im, im_gray, CV_BGR2GRAY);
	//Kernels in x and y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	//Apply derivative filter in x
	Sobel(im_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//Apply derivative filter in y
	Sobel(im_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//Weight gradient based on absolute partial gradients
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	//Disply images
	imshow("Original", im);
	imshow("Rotation", im_rot);
	imshow("Padding", im_pad);
	imshow("Sobel", grad);*/

	HOGDescriptor hog;
	std::vector<float> feat;
	hog.setWinSize(cv::Size(64, 64));
	hog.setBlockSize(cv::Size(32, 32));
	hog.setBlockStep(cv::Size(16, 16));
	hog.setCellSize(cv::Size(8, 8));
	// hog.setPadSize();

	hog.detectHOGDescriptor(im, feat, cv::Size(0, 0), true);


	cv::waitKey(0);
	return 0;

    /*
    	* Create instance of HOGDescriptor and initialize
    	* Compute HOG descriptors
    	* visualize
    */


}