#include "HOGDescriptor.h"
#include <iostream>

void HOGDescriptor::initDetector() {
    // Initialize hog detector
	//initialize default parameters(win_size, block_size, block_step,....)
	win_size = cv::Size(64, 64);

	//Fill other parameters here
	//Included
	block_size = cv::Size(16, 16);
	cell_size = cv::Size(4, 4);
	block_step = cv::Size(2, 2);

	hog_detector = cv::HOGDescriptor(win_size, block_size, block_step, cell_size, 9);
    //Fill code here
    is_init = true;

}


void HOGDescriptor::visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor) {
    // Fill code here (already provided)
	visualizeHOG(img, feats, hog_detector, scale_factor);
}

void HOGDescriptor::detectHOGDescriptor(cv::Mat &im, std::vector<float> &feat, cv::Size sz, bool show) {
	//im: image
	//feat: descriptors
	//sz: window stride: multiple of block stride
    if (!is_init) {
        initDetector();
    }

   // Fill code here
	/* pad your image
	* resize your image
	* use the built in function "compute" to get the HOG descriptors
	*/

	//int border = 50;
	//cv::copyMakeBorder(im, im_pad, border, border, border, border, cv::BORDER_REPLICATE);
	cv::resize(im, im, cv::Size(64, 64), cv::INTER_CUBIC);
	hog_detector.compute(im, feat, sz, cv::Size(0, 0));

	if (show) {
		visualizeHOG(im, feat, hog_detector, 3);
	}




}

//returns instance of cv::HOGDescriptor
cv::HOGDescriptor & HOGDescriptor::getHog_detector() {
     //Fill code here
	return this->hog_detector;
}

