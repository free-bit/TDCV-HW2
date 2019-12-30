#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

void readFiles(std::string path, cv::Mat &gt_labels, cv::Mat &feats);

std::vector<int> perm(int size);

#endif
