#include <opencv2/opencv.hpp>
#include <iostream>


#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task2_utils.h"
#include <fstream>
#include <tuple>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
  #define TRAIN "..\\..\\data\\task3\\train"
  #define TEST "..\\..\\data\\task3\\test"
  #define GT "..\\..\\data\\task3\\gt"
#else //LINUX
  #define TRAIN "../../data/task3/train"
  #define TEST "../../data/task3/test"
  #define GT "../../data/task3/gt"
#endif
#define BACKGROUND 3
#define NUM_CLASS 3
#define IOU_THRESH 0.5
#define CONF_RESOLUTION 1.0    //40
#define LOWER_CONF_THRESH 0.6  //0.5
#define HIGHER_CONF_THRESH 1.0 //0.9
#define VISUAL_INDEX 0

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

void slidingWindow(RandomForest &forest, cv::Mat &im, int size, int stride, std::vector<std::tuple<int, int, float, int>> &best_fit, float conf_thresh){

  std::vector<std::vector<std::tuple<int, int, float>>> box_descriptions(NUM_CLASS, std::vector<std::tuple<int, int, float>>());

  HOGDescriptor hog;
	hog.setWinSize(cv::Size(64, 64));
	hog.setBlockSize(cv::Size(32, 32));
	hog.setBlockStep(cv::Size(16, 16));
	hog.setCellSize(cv::Size(8, 8));

  cv::Rect box(0 ,0, size, size);
  
  for(int i = 0; i + size < im.rows; i += stride){
    box.y = i;
    for(int j = 0; j + size < im.cols; j += stride){
      box.x = j;
      std::vector<float> crop_feat;
      cv::Mat crop = im(box);
      hog.detectHOGDescriptor(crop, crop_feat, false);

      cv::Mat crop_feat_mat = cv::Mat(crop_feat).clone();
      crop_feat_mat.convertTo(crop_feat_mat, CV_32F);
		  crop_feat_mat = crop_feat_mat.reshape(1, 1);

      cv::Mat pred; // will receive a single value
      std::vector<float> confidence = forest.predict(crop_feat_mat, pred); // will receive a single value
      int label = pred.at<float>(0, 0);
      if (label != BACKGROUND){
        std::tuple<int, int, float> box_description;
        std::get<0>(box_description) = j; //x
        std::get<1>(box_description) = i; //y
        std::get<2>(box_description) = confidence[0];
        box_descriptions[label].push_back(box_description);
      }
    }
  }
  // For each class
  for(int i = 0; i < box_descriptions.size(); i++){
    int max_x = -1, max_y = -1;
    float max_conf = 0;
    // For each box found for the class
    for(int j = 0; j < box_descriptions[i].size(); j++){
      std::tuple<int, int, float> box_description = box_descriptions[i][j];
      int x = std::get<0>(box_description);
      int y = std::get<1>(box_description);
      float conf = std::get<2>(box_description);

      // Get the box with the highest confidence
      if (conf > max_conf){
        max_x = x;
        max_y = y;
        max_conf = conf;
        // Compare the current box for ith class with the max one found from other scales
        float best_conf = std::get<2>(best_fit[i]);
        if (max_conf > best_conf && max_conf > conf_thresh){
          std::get<0>(best_fit[i]) = max_x;
          std::get<1>(best_fit[i]) = max_y;
          std::get<2>(best_fit[i]) = max_conf;
          std::get<3>(best_fit[i]) = size;
        }
      }
    }
  }
}

std::vector<std::tuple<int, int, float, int>> tryDifferentScale(RandomForest &forest, cv::Mat &im, int start_size, int end_size, int size_step, int box_stride, float conf_thresh){
  int x = -1, y = -1, conf = 0.0, size = 0;
  std::vector<std::tuple<int, int, float, int>> best_fit(3, std::tuple<int, int, float, int>(x, y, conf, size)); //Per file return 3 box parameters
  for(int size = start_size; size <= end_size; size += size_step){
    slidingWindow(forest, im, size, box_stride, best_fit, conf_thresh);
  }
  //TODO: Remove later: Print highest confidences achieved over scales
  // printf("Class-%d, confidence: %.2f\n", 0, std::get<2>(best_fit[0]));
  // printf("Class-%d, confidence: %.2f\n", 1, std::get<2>(best_fit[1]));
  // printf("Class-%d, confidence: %.2f\n", 2, std::get<2>(best_fit[2]));
  //
  return best_fit;
}

std::vector<std::vector<std::tuple<int, int, float, int>>> tryDifferentConfThresh(RandomForest &forest, cv::Mat &im, int start_size, int end_size, int size_step, int box_stride){
  // Keep best bounding boxes for all confidence thresholds
  std::vector<std::vector<std::tuple<int, int, float, int>>> best_fits;
  // Keep best bounding box for a specific confidence threshold
  std::vector<std::tuple<int, int, float, int>> best_fit;
  for(float conf_thresh = LOWER_CONF_THRESH; conf_thresh <= HIGHER_CONF_THRESH; conf_thresh += 1/CONF_RESOLUTION){
    best_fit = tryDifferentScale(forest, im, start_size, end_size, size_step, box_stride, conf_thresh);
    best_fits.push_back(best_fit);
  }
  return best_fits;
}

std::vector<cv::Rect> readGT(std::string path){
  std::vector<cv::Rect> gt_boxes_per_file;
  std::ifstream file(path);
  std::string line;

  while(std::getline(file, line)){
      std::stringstream linestream(line);
      std::string data;
      int cls, x_top, y_top, x_bot, y_bot;

      linestream >> cls >> x_top >> y_top >> x_bot >> y_bot;
      gt_boxes_per_file.push_back(cv::Rect(x_top, y_top, x_bot-x_top, y_bot-y_top));
  }
  return gt_boxes_per_file;
}

int getIOU(const cv::Rect &pred_box, const cv::Rect &gt_box, float thresh){
  float i = (pred_box & gt_box).area();
  float u = (pred_box | gt_box).area();
  //std::cout<<"Intersection/Union area: "<<(i / u)<<std::endl;
  return (i / u) > thresh;
}

int main(){
    cv::Scalar red(0, 0, 255);  //Class-0
    cv::Scalar green(0, 255, 0);//Class-1
    cv::Scalar blue(255, 0, 0); //Class-2
    std::vector<cv::Scalar> colors = {red, green, blue};
    std::string path(GT);
    std::vector<std::string> filepaths;
    cv::glob(path, filepaths, true);
    std::vector<std::vector<cv::Rect>> gt_boxes;
    for(int i = 0; i < filepaths.size(); i++){
      std::vector<cv::Rect> boxes_per_image = readGT(filepaths[i]);
      gt_boxes.push_back(boxes_per_image);
    }

    // Initialize forest
    int treeCount = 200, maxDepth = 20, CVFolds = 1, minSampleCount = 1, maxCategories = 4;
    RandomForest forest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories);    

    // Load train images and labels
    std::string train_path(TRAIN);
    cv::Mat train_gt_labels;
    cv::Mat train_feats;
    readFiles(train_path, train_gt_labels, train_feats);
    cv::Ptr<cv::ml::TrainData> train_data =  cv::ml::TrainData::create(train_feats, cv::ml::ROW_SAMPLE, train_gt_labels);

    // Train the forest
    forest.train(train_data);

    cv::Mat im;
    path = TEST;
    std::vector<std::string> imagepaths;
    cv::glob(path, imagepaths, true);
    std::vector<std::vector<std::tuple<int, int, float, int>>> best_fits;

    std::ofstream file("../pc_data.txt");
    int start_size = 70, end_size = 210, size_step = 10, box_stride = 10;
    // Create vector of IOU threshold values
    std::vector<std::tuple<float, int, int, int>> evaluate_pr; //float: threshold, int-1: tp, int-2: fp, int-3: fn
    for(float i = LOWER_CONF_THRESH; i <= HIGHER_CONF_THRESH; i += 1/CONF_RESOLUTION){
      evaluate_pr.push_back(std::tuple<float, int, int, int>(i, 0, 0, 0));
    }

    // For each image: i is the file index
    for(int i = 0; i < imagepaths.size(); i++){
      printf("Detecting image-%d...\n", i);
      im = cv::imread(imagepaths[i]);
      best_fits = tryDifferentConfThresh(forest, im, start_size, end_size, size_step, box_stride);
      // For each class: j is the class index
      for(int j = 0; j < gt_boxes[i].size(); j++){
        cv::rectangle(im, gt_boxes[i][j], cv::Scalar(0, 0, 0), 2); //TODO: Drawing gt remove later
        cv::Rect visual_box;
        // For each confidence threshold
        for(int k = 0; k < best_fits.size(); k++){
          int x = std::get<0>(best_fits[k][j]);
          int y = std::get<1>(best_fits[k][j]);
          int size = std::get<3>(best_fits[k][j]);
          // If the box is valid
          if (x != -1){
            cv::Rect pred_box(x, y, size, size);
            if(k == VISUAL_INDEX)
              visual_box = pred_box;
            // If it detects correctly under thresh
            if (getIOU(pred_box, gt_boxes[i][j], IOU_THRESH))
              std::get<1>(evaluate_pr[k]) += 1;
            // If it detects incorrectly under thresh
            else
              std::get<2>(evaluate_pr[k]) += 1;
          }
          else
            std::get<3>(evaluate_pr[k]) += 1;
        }
        cv::rectangle(im, visual_box, colors[j], 2); // Drawing prediction from one of the confidence thresholds 
      }
      cv::imshow("", im);
      cv::waitKey(500); // Wait 0.5 secs

      printf("Detecting image-%d completed.\n", i);
    }
    int tp, fp, fn;
    float conf_thresh, prec, rec;
    for(const auto &pr_thresh : evaluate_pr){
      conf_thresh = std::get<0>(pr_thresh);
      tp = std::get<1>(pr_thresh);
      fp = std::get<2>(pr_thresh);
      fn = std::get<3>(pr_thresh);
      prec = tp / (float) (tp + fp);
      rec = tp / (float) (tp + fn);
      std::cout<<"TP: "<<tp<<std::endl;
      std::cout<<"FP: "<<fp<<std::endl;
      std::cout<<"FN: "<<fn<<std::endl;
      // std::cout<<"P: "<<prec<<std::endl;
      // std::cout<<"R: "<<rec<<std::endl;
      file<<conf_thresh<<" "<<prec<<" "<<rec<<std::endl;
    }
    return 0;
}