#ifndef TASK2_H
#define TASK2_H

#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

void readFiles(std::string path) {
	// Read the files from test and train directory
	// Get labels of images
	// Get HOG descriptors of images (+ GT)
	cv::imread(path);

}

std::vector<int> perm(int size) {
	std::vector<int> indexes;

	for (int i = 0; i < size; ++i)
		indexes.push_back(i);
	std::random_shuffle(indexes.begin(), indexes.end());

	return indexes;
}
#endif
