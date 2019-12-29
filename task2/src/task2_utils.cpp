#include "task2_utils.h"
#include "HOGDescriptor.h"
#include <regex>

void readFiles(std::string path) {
	// Read the files from test or train directory
	// Get labels of images
	// Get HOG descriptors of images (+ GT)
	
	// Get all files under path recursively, store paths on filepaths vector
	std::vector<std::string> filepaths;
	cv::glob(path, filepaths, true);
	
	// Prepare regex for ground truth labels
	std::regex label_re(".*/\\d(\\d)/.*");
	std::smatch label_match;
	
	// Array of ground truth values per file (TODO: Consider taking this as input)
	std::vector<int> gt_labels;

	// Array of HoG features per file (row-wise) (TODO: Consider taking this as input)
	std::vector<std::vector<float>> feats;
	
	//TODO remove later (Taking 3 pictures for testing purposes)
	//std::vector<std::string> sub_set = std::vector<std::string>(filepaths.begin(), filepaths.begin()+3);

	// For filepath in filepaths
	for(const auto &filepath : sub_set){
		// Extract label and store on gt_labels
		std::regex_match(filepath, label_match, label_re);
		std::ssub_match class_idx = label_match[1];
		gt_labels.push_back(std::stoi(class_idx.str()));
		// std::cout<<"Class: "<<gt_labels.back()<<std::endl;		

		// Read image in the path
		// std::cout<<filepath<<std::endl;
		cv::Mat im = cv::imread(filepath);

		// Get HoG of the image (TODO: HoG parameters can be set here)
		HOGDescriptor hog;
		std::vector<float> feat;
		hog.detectHOGDescriptor(im, feat, cv::Size(0, 0), false);
		feats.push_back(feat);
	}
}

std::vector<int> perm(int size) {
	std::vector<int> indexes;

	for (int i = 0; i < size; ++i)
		indexes.push_back(i);
	std::random_shuffle(indexes.begin(), indexes.end());

	return indexes;
}