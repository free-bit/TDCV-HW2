
#include <opencv2/opencv.hpp>
#include <iostream>


#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task2_utils.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
  #define TRAIN "..\\..\\data\\task2\\train"
  #define TEST "..\\..\\data\\task2\\test"
#else //LINUX
  #define TRAIN "../../data/task2/train"
  #define TEST "../../data/task2/test"
#endif

// using namespace std;

template<class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> data) {

    cv::Mat feats = data->getSamples();
    cv::Mat gt_labels = data->getResponses();
    cv::Mat pred_labels;

    classifier->predict(feats, pred_labels);//feats: contain hog per row, pred_labels is a col vector of labels

    
    float accuracy = 0;
    float total_instances = pred_labels.cols;

    for(int j = 0; j<total_instances; j++){
      int gt_label = gt_labels.at<int>(0, j);
      int pred_label = pred_labels.at<float>(0, j);
      std::cout<<"Eval: Image-"<<j<<" with gt: "<<gt_label<<" pred: "<<pred_label<<std::endl;
      if (gt_label == pred_label)
        accuracy++;
    }
    accuracy /= total_instances;
    std::cout<<"Accuracy: "<<accuracy<<std::endl;
};


void testDTrees() {

    int num_classes = 6;

    /* 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a single Decision Tree and evaluate the performance 
      * Experiment with the MaxDepth parameter, to see how it affects the performance
    */
    // Create and init one decision tree
    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
    dtree->setMaxDepth(20);
		dtree->setCVFolds(1);
		dtree->setMinSampleCount(1);
		dtree->setMaxCategories(6);

    // Load train images and labels
    std::string train_path(TRAIN);
    cv::Mat train_gt_labels;
    cv::Mat train_feats;
    readFiles(train_path, train_gt_labels, train_feats);
    cv::Ptr<cv::ml::TrainData> train_data =  cv::ml::TrainData::create(train_feats, cv::ml::ROW_SAMPLE, train_gt_labels);

    // Train the tree
    dtree->train(train_data);

    // Load test images and labels
    std::string test_path(TEST);
    cv::Mat test_gt_labels;
    cv::Mat test_feats;
    readFiles(test_path, test_gt_labels, test_feats);
    cv::Ptr<cv::ml::TrainData> test_data =  cv::ml::TrainData::create(test_feats, cv::ml::ROW_SAMPLE, test_gt_labels);

    performanceEval<cv::ml::DTrees>(dtree, train_data);
    performanceEval<cv::ml::DTrees>(dtree, test_data);

}


void testForest(){

    int treeCount = 15, maxDepth = 20, CVFolds = 1, minSampleCount = 1, maxCategories = 6;
    /* 
      * 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance 
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */
    // Create and init random forest
    cv::Ptr<RandomForest> forest = new RandomForest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories);
    
    // Load train images and labels
    std::string train_path(TRAIN);
    cv::Mat train_gt_labels;
    cv::Mat train_feats;
    readFiles(train_path, train_gt_labels, train_feats);
    cv::Ptr<cv::ml::TrainData> train_data =  cv::ml::TrainData::create(train_feats, cv::ml::ROW_SAMPLE, train_gt_labels);

    // Train the forest
    forest->train(train_data);

    // Load test images and labels
    std::string test_path(TEST);
    cv::Mat test_gt_labels;
    cv::Mat test_feats;
    readFiles(test_path, test_gt_labels, test_feats);
    cv::Ptr<cv::ml::TrainData> test_data =  cv::ml::TrainData::create(test_feats, cv::ml::ROW_SAMPLE, test_gt_labels);

    performanceEval<RandomForest>(forest, train_data);
    performanceEval<RandomForest>(forest, test_data);
}


int main(){
    //testDTrees();
    testForest();
    return 0;
}
