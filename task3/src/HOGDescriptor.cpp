#include "HOGDescriptor.h"
#include "hog_visualization.h"
#include <iostream>

void HOGDescriptor::printParams(){
	std::cout<<"Parameters:"<<std::endl;
	std::cout<<"Window Size: "<<win_size<<std::endl;
	std::cout<<"Window Stride: "<<win_stride<<std::endl;
	std::cout<<"Block Size: "<<block_size<<std::endl;
	std::cout<<"Block Step: "<<block_step<<std::endl;
	std::cout<<"Cell Size: "<<cell_size<<std::endl;
	std::cout<<"Pad Size: "<<pad_size<<std::endl;
}

// Default initializer
void HOGDescriptor::initDetector() {
    // Initialize hog detector
	//initialize default parameters(win_size, block_size, block_step,....)
	win_size = cv::Size(64, 64);
	block_size = cv::Size(16, 16);
	block_step = cv::Size(8, 8);
	cell_size = cv::Size(8, 8);
	win_stride = cv::Size(0, 0);
	pad_size = cv::Size(0, 0);

	hog_detector = cv::HOGDescriptor(win_size, block_size, block_step, cell_size, 9);
    is_init = true;
}

// Use references to prevent OpenCV from copying hog_detector all the time!
void HOGDescriptor::visualizeHOG(cv::Mat &img, std::vector<float> &feats, cv::HOGDescriptor &hog_detector, int scale_factor) {
    // Fill code here (already provided)
	// Call function from the global scope. Otherwise, it calls the method itself and infinite recursion happens.
	::visualizeHOG(img, feats, hog_detector, scale_factor);
}

void HOGDescriptor::detectHOGDescriptor(cv::Mat &im, std::vector<float> &feat, bool show) {
	//im: image
	//feat: descriptors
	//sz: window stride: multiple of block stride
    if (!is_init) {
        initDetector();
    }

	cv::resize(im, im, cv::Size(64, 64));
	hog_detector.compute(im, feat, win_stride, pad_size);

	if (show) {
		visualizeHOG(im, feat, hog_detector, 10);
	}
}

// Returns instance of cv::HOGDescriptor
cv::HOGDescriptor & HOGDescriptor::getHog_detector() {
	return this->hog_detector;
}

