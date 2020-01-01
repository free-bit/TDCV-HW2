#include <regex>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
  #define TRAIN "..\\..\\data\\task3\\train"
  #define TEST "..\\..\\data\\task3\\test"
#else //LINUX
  #define TRAIN "../../data/task3/train"
  #define TEST "../../data/task3/test"
#endif

#define OUTPUT "../../aug_data/task3/train"

void augmentImages(std::string path, std::string output_path){
	std::cout<<"augmentImages runs..."<<std::endl;
	// Read the files from test or train directory

	// Get all files under path recursively, store paths on filepaths vector
	std::vector<std::string> filepaths;
	cv::glob(path, filepaths, true);

	// Allocate space for column vector
	int n_rows = filepaths.size();
	
	// Prepare regex for ground truth labels
	std::regex re(".*(/\\d\\d/)(.*)(\\.jpg)");
	std::smatch re_match;
		
	// For filepath in filepaths
	for(int i = 0; i < n_rows; i++){
        std::regex_match(filepaths[i], re_match, re);
		    std::string class_name = re_match[1].str();
        std::string name = re_match[2].str();
        std::string ext = re_match[3].str();
        // std::cout<<"I: "<<filepaths[i]<<std::endl;
        // std::cout<<"O: "<<output_path<<class_name<<name<<ext<<std::endl;
        // std::cout<<class_name<<std::endl;
        // Read image in the path
        cv::Mat im = cv::imread(filepaths[i]);
        cv::Mat im_tmp;

        // Scale 1.5x, 0.5x, 2x
        // cv::resize(im, im_tmp, cv::Size(im.size().height * 1.5,im.size().width * 1.5), cv::INTER_CUBIC);
        // cv::imwrite(output_path + class_name + name + "_scale_one_and_half" + ext, im_tmp);
        // cv::resize(im, im_tmp, cv::Size(im.size().height * 0.5,im.size().width * 0.5), cv::INTER_CUBIC);
        // cv::imwrite(output_path + class_name + name + "_scale_half" + ext, im_tmp);
        // cv::resize(im, im_tmp, cv::Size(im.size().height * 2,im.size().width * 2), cv::INTER_CUBIC);
        // cv::imwrite(output_path + class_name + name + "_scale_twice" + ext, im_tmp);

        // Rotate 90, 180, 270
        // cv::rotate(im, im_tmp, cv::ROTATE_90_CLOCKWISE);    //90
        // cv::imwrite(output_path + class_name + name + "_rot90" + ext, im_tmp);
        // cv::rotate(im_tmp, im_tmp, cv::ROTATE_90_CLOCKWISE);//180
        // cv::imwrite(output_path + class_name + name + "_rot180" + ext, im_tmp);
        // cv::rotate(im_tmp, im_tmp, cv::ROTATE_90_CLOCKWISE);//270
        // cv::imwrite(output_path + class_name + name + "_rot270" + ext, im_tmp);

        // Gaussian blur with kernels 5x5, 11x11, 19x19
        cv::GaussianBlur(im, im_tmp, cv::Size(5, 5), 0, 0);
        cv::imwrite(output_path + class_name + name + "_blur_5x5" + ext, im_tmp);
        cv::GaussianBlur(im, im_tmp, cv::Size(11, 11), 0, 0);
        cv::imwrite(output_path + class_name + name + "_blur_11x11" + ext, im_tmp);
        cv::GaussianBlur(im, im_tmp, cv::Size(19, 19), 0, 0);
        cv::imwrite(output_path + class_name + name + "_blur_19x19" + ext, im_tmp);
	}
	std::cout<<"augmentImages finishes."<<std::endl;
}

int main(){
    std::string path(TRAIN);
    std::string output_path(OUTPUT);
    augmentImages(path, output_path);
    return 0;
}