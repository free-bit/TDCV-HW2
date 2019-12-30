
#include <opencv2/opencv.hpp>
#include <iostream>


#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task2_utils.h"
#include <tuple>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
  #define TRAIN "..\\..\\data\\task3\\train"
  #define TEST "..\\..\\data\\task3\\test"
#else //LINUX
  #define TRAIN "../../data/task3/train"
  #define TEST "../../data/task3/test"
#endif
#define BACKGROUND 3

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


void slidingWindow(RandomForest &forest, cv::Mat &im, int size, int stride){

  std::vector<std::vector<std::tuple<int, int, float>>> box_descriptions(4, std::vector<std::tuple<int, int, float>>());
  std::vector<std::vector<std::vector<int>>> valid_boxes(4, std::vector<std::vector<int>>());

  HOGDescriptor hog;
	hog.setWinSize(cv::Size(64, 64));
	hog.setBlockSize(cv::Size(32, 32));
	hog.setBlockStep(cv::Size(16, 16));
	hog.setCellSize(cv::Size(8, 8));

  cv::Rect box(0 ,0, size, size);

  //remove
  // cv::Mat test_im = im.clone();
  //
  
  for(int i = 0; i + size < im.rows; i += stride){
    box.y = i;
    for(int j = 0; j + size < im.cols; j += stride){
      box.x = j;
      std::vector<float> crop_feat;
      cv::Mat crop = im(box);
      hog.detectHOGDescriptor(crop, crop_feat, cv::Size(0, 0), false);

      cv::Mat crop_feat_mat = cv::Mat(crop_feat).clone();
      crop_feat_mat.convertTo(crop_feat_mat, CV_32F);
		  crop_feat_mat = crop_feat_mat.reshape(1, 1);

      cv::Mat pred; // will receive a single value
      std::vector<float> confidence = forest.predict(crop_feat_mat, pred); // will receive a single value
      int label = pred.at<float>(0, 0);
      if (label != BACKGROUND){
        std::tuple<int, int, float> box_description;
        std::get<0>(box_description) = j;
        std::get<1>(box_description) = i;
        std::get<2>(box_description) = confidence[0];
        box_descriptions[label].push_back(box_description);
      }
      //remove
      // cv::rectangle(test_im, box, cv::Scalar(0, 0, 255), 3);
      // cv::imshow("", test_im);
      // cv::waitKey(0);
      //
    }
  }
  // For each class
  for(int i = 0; i < box_descriptions.size(); i++){
    // For each box detecting the class
    for(int j = 0; j < box_descriptions[i].size(); j++){
      int x1 = std::get<0>(box_descriptions[i][j]);
      int y1 = std::get<1>(box_descriptions[i][j]);
      float conf1 = std::get<2>(box_descriptions[i][j]);
      cv::Rect box1(x1 ,y1, size, size);
      // take a second box from the list
      for(int k = j + 1; k < box_descriptions[i].size(); k++){
        int x2 = std::get<0>(box_descriptions[i][k]);
        int y2 = std::get<1>(box_descriptions[i][k]);
        float conf2 = std::get<2>(box_descriptions[i][k]);
        cv::Rect box2(x2 ,y2, size, size); 
        // Non overlapping box1 & box2
        if (box2.x+size < box1.x || box1.x+size < box2.x || box1.y+size < box2.y || box2.y+size < box1.y){
          std::vector<int> pair1 = {box1.x, box1.y};
          valid_boxes[i].push_back(pair1);
          std::vector<int> pair2 = {box2.x, box2.y};
          valid_boxes[i].push_back(pair2);
          cv::putText(im, "No overlap", cv::Point(box1.x,box1.y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,200,250));
          cv::putText(im, "No overlap", cv::Point(box2.x,box2.y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(250,0,200));
        }
        //Overlap happens
        else{
          if(conf1 >= conf2){
            std::vector<int> pair = {box1.x, box1.y};
            valid_boxes[i].push_back(pair);
            std::cout<<"Overlap in class:"<<i<<std::endl;
          }
          else{
            std::vector<int> pair = {box2.x, box2.y};
            valid_boxes[i].push_back(pair);
            std::cout<<"Overlap in class:"<<i<<std::endl;
          }
        }
      }
    }
  }

  //
  cv::Scalar red(255, 0, 0);
  cv::Scalar green(0, 255, 0);
  cv::Scalar blue(0, 0, 255);
  std::vector<cv::Scalar> colors = {red, green, blue};
  int i = 0;
  for(const auto &classes : valid_boxes){
    for(const auto &box_params : classes){
      cv::Rect box(box_params[0], box_params[1], size, size);
      cv::rectangle(im, box, colors[i], 3);
    }
    i++;
  }
  cv::imshow("", im);
  cv::waitKey(0);
  //
}

/*
1) Crop: OK
-> HOG: OK
2) Predict: OK
3) Check if background: 3?
4) If not background save object index, box top-left, (we know the size), confidence: OK
5) If bounding boxes overlap take the highest confidence OK
for each class:
  if overlap:
    remove overlap based on confidence

draw
  
*/


int main(){
    //TODO: Read one test image for now, later read all
    std::string path("../../data/task3/test/0000.jpg");
    cv::Mat im = cv::imread(path);
    //

    int treeCount = 15, maxDepth = 20, CVFolds = 1, minSampleCount = 1, maxCategories = 6;
    RandomForest forest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories);    

    // Load train images and labels
    std::string train_path(TRAIN);
    cv::Mat train_gt_labels;
    cv::Mat train_feats;
    readFiles(train_path, train_gt_labels, train_feats);
    cv::Ptr<cv::ml::TrainData> train_data =  cv::ml::TrainData::create(train_feats, cv::ml::ROW_SAMPLE, train_gt_labels);

    // Train the forest
    forest.train(train_data);

    slidingWindow(forest, im, 200, 25);
    //testDTrees();
    //testForest();
    return 0;
}
