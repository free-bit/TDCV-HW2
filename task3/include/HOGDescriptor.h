

#ifndef RF_HOGDESCRIPTOR_H
#define RF_HOGDESCRIPTOR_H


#include <opencv2/opencv.hpp>
#include <vector>

class HOGDescriptor {

public:

    HOGDescriptor() {

		initDetector();
        // parameter to check if descriptor is already initialized or not
        is_init = false;
    };


    void setWinSize(cv::Size win_size) {
        //Fill
		this->win_size = win_size;
        
    }

    cv::Size getWinSize(){
        //Fill
		return this->win_size;
    }

    void setBlockSize(cv::Size block_size) {
        //Fill
		this->block_size = block_size;
    }

    void setBlockStep(cv::Size block_step) {
       //Fill
		this->block_step = block_step;
    }

    void setCellSize(cv::Size cell_size) {
      //Fill
		this->cell_size = cell_size;
    }

    void setPadSize(cv::Size sz) {
        auto pad_size = sz;
    }


    void initDetector();

    void visualizeHOG(cv::Mat &img, std::vector<float> &feats, cv::HOGDescriptor &hog_detector, int scale_factor);

    void detectHOGDescriptor(cv::Mat &im, std::vector<float> &feat, cv::Size sz, bool show);

    ~HOGDescriptor() {};


private:
    cv::Size win_size;
	cv::Size block_size;
	cv::Size block_step;
	cv::Size cell_size;

    cv::HOGDescriptor hog_detector;
public:
    cv::HOGDescriptor & getHog_detector();

private:
    bool is_init;
};

#endif //RF_HOGDESCRIPTOR_H