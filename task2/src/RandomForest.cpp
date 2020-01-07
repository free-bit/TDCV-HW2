#include "RandomForest.h"
#include "task2_utils.h"
#include <fstream>

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
	this->mCVFolds = cvFolds;
}

void RandomForest::setMinSampleCount(int minSampleCount)
{
	this->mMinSampleCount = minSampleCount;
}

void RandomForest::setMaxCategories(int maxCategories)
{
	this->mMaxCategories = maxCategories;
}

void RandomForest::train(cv::Ptr<cv::ml::TrainData> &data)
{
	std::cout<<"RandomForest::train runs..."<<std::endl;
	// input = training images (rowwise)
	// layout = Rowwise 
	// traing_labels as vector

    cv::Mat feats = data->getSamples();
    cv::Mat gt_labels = data->getResponses();
	// set_size is the size of the subset of the training data
	int set_size = feats.rows / 5; // taking %20 of training data

	// For each tree in the forest
	for (int i = 0; i < mTreeCount; i++) {
		// Get permuted indices in the same size as the size of training data
		std::vector<int> perm_indices = perm(feats.rows);

		cv::Mat sub_input, sub_labels;

		// Take a subset of size set_size from permuted training data
		for (int j = 0; j < set_size; j++) {
			sub_input.push_back(feats.row(perm_indices[j]));
			sub_labels.push_back(gt_labels.row(perm_indices[j]));
		}
		// Train i-th tree
		mTrees[i]->train(sub_input, cv::ml::ROW_SAMPLE, sub_labels);
	}
	std::cout<<"RandomForest::train finishes."<<std::endl;
}

/*
Perform prediction on one or more image descriptors based on the model trained
Return a list of predictions and a list of confidence values such that each list contain one value per image
*/
std::vector<float> RandomForest::predict(cv::Mat &feats, cv::Mat &voted_preds)
{
	std::vector<int> voted_preds_vec(feats.rows);
	std::vector<float> confidences(feats.rows);
	// For each image descriptor (each row of feats is the feature of a different image)
	for (int i = 0; i < feats.rows; i++) {

		std::vector<int> preds(mMaxCategories);
		// Perform prediction & voting with all trees in the forest 
		for (int j = 0; j < mTreeCount; j++) {
			cv::Mat label;
			mTrees[j]->predict(feats.row(i), label);
			preds[label.at<float>(0,0)]++; //for each class count instances voted by trees
		}
		// Keep the index (=label) which received the highest number of votes along with the number of votes it received
		int max = -1;
		int max_index = -1;
		for (int j = 0; j < mMaxCategories; j++) {
			if (preds[j]>max) {
				max = preds[j];
				max_index = j;
			}
		}
		voted_preds_vec[i] = max_index;			   // Per image keep the predicted (top voted) class index
		confidences[i] = max / (float) mTreeCount; // Per image keep the confidence for top voted class
	}
	voted_preds = cv::Mat(voted_preds_vec).clone();
    voted_preds.convertTo(voted_preds, CV_32F);
	voted_preds = voted_preds.reshape(1, 1);
	return confidences;
}

// Save the state of each decision tree
void RandomForest::saveTrees(std::string path){
	for (int i = 0; i < mTreeCount; i++) {
		mTrees[i]->save(path + std::string("tree_") + std::to_string(i) + std::string(".yml"));
	}
}

// Save the parameters of forest
void RandomForest::saveParams(std::string path){
	std::ofstream file(path + std::string("params.txt"));
	file << mTreeCount << " " << mMaxDepth << " " << mCVFolds << " " << mMinSampleCount << " " << mMaxCategories << std::endl;
}

// Save the state of forest (parameters + states of the trees)
void RandomForest::saveModel(std::string path){
	std::cout<<"RandomForest::saveModel runs..."<<std::endl;
	saveParams(path);
	saveTrees(path);
	std::cout<<"RandomForest::saveModel finishes."<<std::endl;
}

// Load parameters and trees, overrides existing forest
void RandomForest::loadModel(std::string path){
	std::cout<<"RandomForest::loadModel runs..."<<std::endl;
	std::vector<std::string> filenames;
	cv::utils::fs::glob_relative(path, std::string("*.txt"), filenames);

	std::ifstream file(path + filenames[0]);
  	std::string line;
	std::getline(file, line);
    std::stringstream linestream(line);
	linestream >> mTreeCount >> mMaxDepth >> mCVFolds >> mMinSampleCount >> mMaxCategories;

	std::cout<<std::endl;
	printParams();
	std::cout<<std::endl;

	filenames.clear();
	cv::utils::fs::glob_relative(path, std::string("tree_*.yml"), filenames);

	mTrees.clear();
	cv::Ptr<cv::ml::DTrees> tree;
	for(int i = 0; i < filenames.size(); i++){
		tree = cv::ml::DTrees::load(path + filenames[i]);
		mTrees.push_back(tree);
	}
	std::cout<<"RandomForest::loadModel finishes."<<std::endl;
}

void RandomForest::printParams(){
	std::cout << "Parameters:" << std::endl;
	std::cout << "Tree Count: " << mTreeCount << std::endl;
	std::cout << "Max Categories: " << mMaxCategories << std::endl;
	std::cout << "Max Depth (for all trees): " << mMaxDepth << std::endl;
	std::cout << "CV Folds (for all trees): " << mCVFolds << std::endl;
	std::cout << "Min Sample Count (for all trees): " << mMinSampleCount << std::endl;
}