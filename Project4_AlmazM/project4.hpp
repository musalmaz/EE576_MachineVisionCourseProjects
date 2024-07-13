/*
Date:26/04/2024
Developed by: Musa Almaz
Project: Project 4 - representation, learning and recognition
Summary: This is the header file for the tasks of the Project 4.
Basically includes functions for extracting features of images using SIFT, training one class SVM's and
classifying test images. RGB and Depth images are used.
*/
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace fs = std::filesystem;
using namespace cv::xfeatures2d;

enum Data_Type {
    rgb_image = 0,
    depth_image = 1
};

class Project4 {
    public:
        

        /**
         * @brief Helper function to print paths (for debugging)
         * 
         * @param paths A set of pair object name and its related path
         */
        void printPaths(const std::vector<std::pair<std::string, std::string>>& paths);

        // Function to collect image paths, leaving out one folder per object for testing
        /**
         * @brief Collecting image paths and related images from a directory
         * 
         * @param data_type RGB image or depth image
         * @return std::pair<std::vector<std::pair<std::string, std::string>>, std::vector<std::pair<std::string, std::string>>> 
         */
        std::pair<std::vector<std::pair<std::string, std::string>>, std::vector<std::pair<std::string, std::string>>> collectImagePaths(Data_Type data_type);

        /**
         * @brief Main function for the project
         * 
         */
        void run();



    private:

        /**
         * @brief SIFT feature extraction
         * 
         * @param image Input image for SIFT
         * @param descriptors Output of the SIFT
         * @return std::vector<cv::KeyPoint> Keypoint output of the SIFT
         */
        std::vector<cv::KeyPoint> extractSIFTFeatures(const cv::Mat& image, cv::Mat& descriptors);

        /**
         * @brief Creates bow representation of images
         * 
         * @param imagePaths Image paths as string
         * @return std::vector<cv::Mat> Representations of the images
         */
        std::vector<cv::Mat> createBOWRepresentation(const std::vector<std::string>& imagePaths);

        /**
         * @brief Creates bow representation of images
         * 
         * @param images vector of images
         * @return std::vector<cv::Mat> Representation of images
         */
        std::vector<cv::Mat> createBOWRepresentation(const std::vector<cv::Mat>& images);

        /**
         * @brief Function for training one class SVM's
         * 
         * @param descriptors Bow descriptors of an object
         * @param objectName Object name that will be trained as string
         * @return cv::Ptr<cv::ml::SVM> Trained SVM
         */
        cv::Ptr<cv::ml::SVM> trainOneClassSVM(const std::vector<cv::Mat>& descriptors, const std::string& objectName);

        /**
         * @brief Function for classifying images
         * 
         * @param svm Trained SVM
         * @param descriptors Test image descriptors
         * @param threshold Threshold value for comparison of the prediction result
         * @return std::vector<int> Prediction value for each descriptor
         */
        std::vector<int> classifyImages(const cv::Ptr<cv::ml::SVM>& svm, const std::vector<cv::Mat>& descriptors, float threshold);

        /**
         * @brief Function to mask RGB image
         * 
         * @param orgImagePath RGB image path
         * @param maskPath Mask image path
         * @return cv::Mat Masked image
         */
        cv::Mat bitwiseAnd(std::string orgImagePath, std::string maskPath);

        /**
         * @brief Function to converting PCD data to image
         * 
         * @param filename Path of the data
         * @return cv::Mat Converted image
         */
        cv::Mat convertPCDtoMat(const std::string& filename);

        /**
         * @brief Function for converting images CV_8U
         * 
         * @param img Input image that will be checked
         * @return cv::Mat COnerted image
         */
        cv::Mat correctImageType(cv::Mat img);

        /**
         * @brief Function to compute precision, recall, accuracy
         * 
         * @param descriptors Descriptors of the test images
         * @param svmTrains Trained SVM's
         * @param object_names Object names of the tested images
         * @param allTestImagesName Vector of the names of test images
         */
        void computeMetrics(const std::vector<cv::Mat>& descriptors,
            const std::map<std::string, cv::Ptr<cv::ml::SVM>>& svmTrains,
            const std::vector<std::string>& object_names,
            const std::vector<std::string>& allTestImagesName, Data_Type data_type);

        /**
         * @brief Configuring the dataset
         * 
         * @param train_or_test_set Pairs of dataset with object name and file path
         * @return std::map<std::string, std::vector<std::string>> Vector of file paths for each object
         */
        std::map<std::string, std::vector<std::string>> organizePathsByObject(std::vector<std::pair<std::string, std::string>> train_or_test_set);

        /**
         * @brief Function for plotting ROC curve for each object
         * 
         * @param precisions Set of precision values
         * @param recalls    Set of recall values
         * @param thresholds Set of threshold values
         * @param className  Name of the related class
         */
        void plotPRCurve(const std::vector<double>& precisions, const std::vector<double>& recalls,
                 const std::vector<double>& thresholds, const std::string& className, Data_Type data_type);

        /**
         * @brief Function to configure dataset
         * 
         * @param mappedImages Map of object name and related images
         * @return std::vector<std::pair<std::string, cv::Mat>> Set of pair for image and its related object name
         */
        std::vector<std::pair<std::string, cv::Mat>> flattenTestImages(const std::map<std::string, std::vector<cv::Mat>>& mappedImages);

        int num_objects = 5;

        std::string datasetDir = "../rgbd-dataset";
        std::string datasetDirDepth = "../depth-dataset";

        std::vector<std::pair<std::string, std::string>> trainingPaths;
        std::vector<std::pair<std::string, std::string>> testPaths;

        std::vector<std::pair<std::string, std::string>> trainingPathsDepth;
        std::vector<std::pair<std::string, std::string>> testPathsDepth;

        std::map<std::string, std::vector<fs::path>> objectFolders;

        std::map<std::string, std::vector<cv::Mat>> trainImages;
        std::map<std::string, std::vector<cv::Mat>> testImages;

        std::map<std::string, std::vector<cv::Mat>> trainImagesDepth;
        std::map<std::string, std::vector<cv::Mat>> testImagesDepth;

        std::vector<std::pair<std::string, cv::Mat>> flattenedTestImages;
        std::vector<std::pair<std::string, cv::Mat>> flattenedTestImagesDepth;

        std::vector<cv::Mat> allTestImages;
        std::vector<std::string> allTestImagesNames;

        std::vector<cv::Mat> allTestImagesDepth;
        std::vector<std::string> allTestImagesNamesDepth;


        int dictionarySize = 100;

         // Threshold values to try
        std::vector<double> thresholds = {0.2, 0.3, 0.4};
};
