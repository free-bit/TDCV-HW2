#include "RandomForest.h"
#include "task2.h"

RandomForest::RandomForest()
{
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
    :mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount), mMaxCategories(maxCategories)
{
   /*
     construct a forest with given number of trees and initialize all the trees with the
     given parameters
   */

	for (int i = 0; i < treeCount; i++) {
		mTrees.push_back(cv::ml::DTrees::create());

		mTrees[i]->setMaxDepth(mMaxDepth);
		mTrees[i]->setCVFolds(mCVFolds);
		mTrees[i]->setMinSampleCount(mMinSampleCount);
		mTrees[i]->setMaxCategories(mMaxCategories);
	}

}

RandomForest::~RandomForest()
{
}

void RandomForest::setTreeCount(int treeCount)
{
    // Fill
	this->mTreeCount = treeCount;

}

void RandomForest::setMaxDepth(int maxDepth)
{
    mMaxDepth = maxDepth;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFolds)
{
    // Fill
	this->mCVFolds = cvFolds;


}

void RandomForest::setMinSampleCount(int minSampleCount)
{
    // Fill
	this->mMinSampleCount = minSampleCount;
}

void RandomForest::setMaxCategories(int maxCategories)
{
    // Fill
	this->mMaxCategories = maxCategories;
}



void RandomForest::train(cv::Mat input_array,int layout, cv::Mat train_labels)
{
    // Fill
	// input = training images (rowwise)
	// layout = Rowwise 
	// traing_labels as vector

	int set_size = input_array.rows / mMaxCategories;

	for (int i = 0; i < mTreeCount; i++) {
		std::vector<int> perm_indices = perm(input_array.rows);

		cv::Mat sub_input, sub_labels;

		for (int j = 0; j < set_size; j++) {
			sub_input.push_back(input_array.at<float>(perm_indices[j]));
			sub_labels.push_back(train_labels.at<float>(perm_indices[j]));
		}
		mTrees[i]->train(sub_input, layout,sub_labels);
	}
}

float RandomForest::predict(cv::Mat test_images)
{
    // Fill
	// We assumed: DT predicts us class labels 0...5 as float?

	std::vector<int> voted_preds(test_images.rows);
	std::vector<float> confidences(test_images.rows);
	for (int i = 0; i < test_images.rows; i++) {

		std::vector<int> preds(mMaxCategories);
		for (int j = 0; j < mTreeCount; j++) {
			cv::Mat label;
			mTrees[j]->predict(test_images.at<float>(i), label);
			preds[label.at<float>(0,0)]++;//for each class count instances voted by trees
		}
		int max = -1;
		int max_index = -1;
		for (int j = 0; j < mMaxCategories; j++) {
			if (preds[j]>max) {
				max = preds[j];
				max_index = j;
			}
		}
		voted_preds[i] = max_index;
		confidences[i] = max / mTreeCount;
		

	}



}

