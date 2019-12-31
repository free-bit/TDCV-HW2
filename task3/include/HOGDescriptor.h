

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

    void printParams();

    void setWinSize(cv::Size win_size) {
        //Fill
		this->win_size = win_size;
        this->hog_detector.winSize = this->win_size;
    }

    cv::Size getWinSize(){
        //Fill
		return this->win_size;
    }

    void setBlockSize(cv::Size block_size) {
        //Fill
		this->block_size = block_size;
        this->hog_detector.blockSize = this->block_size;
    }

    void setBlockStep(cv::Size block_step) {
       //Fill
		this->block_step = block_step;
        this->hog_detector.blockStride = this->block_step;
    }

    void setCellSize(cv::Size cell_size) {
      //Fill
		this->cell_size = cell_size;
        this->hog_detector.cellSize = this->cell_size;
    }

    void setPadSize(cv::Size pad_size) {
        this->pad_size = pad_size;
    }

    void setWinStride(cv::Size win_stride){
        this->win_stride = win_stride;
    }


    void initDetector();

    void visualizeHOG(cv::Mat &img, std::vector<float> &feats, cv::HOGDescriptor &hog_detector, int scale_factor);

    void detectHOGDescriptor(cv::Mat &im, std::vector<float> &feat, bool show);

    ~HOGDescriptor() {};


private:
    cv::Size win_size;
    cv::Size win_stride;
	cv::Size block_size;
	cv::Size block_step;
	cv::Size cell_size;
    cv::Size pad_size;

    cv::HOGDescriptor hog_detector;
public:
    cv::HOGDescriptor & getHog_detector();

private:
    bool is_init;
};

#endif //RF_HOGDESCRIPTOR_H