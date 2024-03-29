#include "task2_utils.h"
#include "HOGDescriptor.h"
#include <regex>

/*
Read the files from given path (test or train directory)
Get labels of images
Get HOG descriptors of images
Return ground truth labels and extracted features
*/
void readFiles(std::string path, cv::Mat &gt_labels, cv::Mat &feats) {
	std::cout<<"readFiles runs..."<<std::endl;

	// Get all files under path recursively, store paths on filepaths vector
	std::vector<std::string> filepaths;
	cv::glob(path, filepaths, true);

	// Allocate space for column vector
	int n_rows = filepaths.size();
	
	// Prepare regex for ground truth labels
	std::regex label_re(".*/\\d(\\d)/.*");
	std::smatch label_match;
		
	// For filepath in filepaths
	for(int i = 0; i < n_rows; i++){
		// Extract label and store on gt_labels
		std::regex_match(filepaths[i], label_match, label_re);
		std::ssub_match class_idx = label_match[1];
		gt_labels.push_back(std::stoi(class_idx.str()));
		
		// Read image in the path
		cv::Mat im = cv::imread(filepaths[i]);

		// Get HoG of the image
		HOGDescriptor hog; // Using default parameters
		std::vector<float> feat;
		hog.detectHOGDescriptor(im, feat, false); // false: Don't visiualize
		cv::Mat feat_mat = cv::Mat(feat).clone();
        feat_mat.convertTo(feat_mat, CV_32F);
		feat_mat = feat_mat.reshape(1,1);
		feats.push_back(feat_mat);
	}
	std::cout<<"readFiles finishes."<<std::endl;
}

// Create a shuffled index array based on given size (values in range: [0-size-1])
std::vector<int> perm(int size) {
	std::vector<int> indexes;
	// Create index array
	for (int i = 0; i < size; ++i)
		indexes.push_back(i);
	// Shuffle indices
	std::random_shuffle(indexes.begin(), indexes.end());

	return indexes;
}