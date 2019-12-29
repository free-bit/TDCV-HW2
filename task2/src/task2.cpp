
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

using namespace std;

template<class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> data) {

	/* 

		Fill Code 	

	*/

};






void testDTrees() {

    int num_classes = 6;

    /* 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a single Decision Tree and evaluate the performance 
      * Experiment with the MaxDepth parameter, to see how it affects the performance
    */



    // performanceEval<cv::ml::DTrees>(tree, train_data);
    // performanceEval<cv::ml::DTrees>(tree, test_data);

}


void testForest(){

    int num_classes = 6;

    /* 
      * 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance 
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */

    // performanceEval<RandomForest>(forest, train_data);
    // performanceEval<RandomForest>(forest, test_data);
}


int main(){
    string train_path(TRAIN);
    readFiles(train_path);

    // string test_path(TEST);
    // readFiles(test_path);
    
    testDTrees();
    testForest();
    return 0;
}
