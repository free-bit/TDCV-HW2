#include "RandomForest.h"
#include "task2_utils.h"

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



void RandomForest::train(cv::Ptr<cv::ml::TrainData> &data)
{
    // Fill
	// input = training images (rowwise)
	// layout = Rowwise 
	// traing_labels as vector

    cv::Mat feats = data->getSamples();
    cv::Mat gt_labels = data->getResponses();

	int set_size = feats.rows / mMaxCategories;

	for (int i = 0; i < mTreeCount; i++) {
		std::vector<int> perm_indices = perm(feats.rows);

		cv::Mat sub_input, sub_labels;

		for (int j = 0; j < set_size; j++) {
			sub_input.push_back(feats.row(perm_indices[j]));
			sub_labels.push_back(gt_labels.row(perm_indices[j]));
		}
		mTrees[i]->train(sub_input, cv::ml::ROW_SAMPLE, sub_labels);
	}
}

std::vector<float> RandomForest::predict(cv::Mat &feats, cv::Mat &voted_preds)
{
    // Fill
	// We assumed: DT predicts us class labels 0...5 as float?

	std::vector<int> voted_preds_vec(feats.rows);
	std::vector<float> confidences(feats.rows);
	for (int i = 0; i < feats.rows; i++) {

		std::vector<int> preds(mMaxCategories);
		for (int j = 0; j < mTreeCount; j++) {
			cv::Mat label;
			mTrees[j]->predict(feats.row(i), label);
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
		voted_preds_vec[i] = max_index;
		confidences[i] = max / mTreeCount;
	}
	voted_preds = cv::Mat(voted_preds_vec).clone();
    voted_preds.convertTo(voted_preds, CV_32F);
	voted_preds = voted_preds.reshape(1, 1);
	return confidences;
}
