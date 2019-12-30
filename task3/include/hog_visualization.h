#ifndef HOG_V_H
#define HOG_V_H
#include <opencv2/opencv.hpp>
#include <vector>

void visualizeHOG(cv::Mat &img, std::vector<float> &feats, cv::HOGDescriptor &hog_detector, int scale_factor = 3);
/*
 * img - the image used for computing HOG descriptors. **Attention here the size of the image should be the same as the window size of your cv::HOGDescriptor instance **
 * feats - the hog descriptors you get after calling cv::HOGDescriptor::compute
 * hog_detector - the instance of cv::HOGDescriptor you used
 * scale_factor - scale the image *scale_factor* times larger for better visualization
 */
#endif