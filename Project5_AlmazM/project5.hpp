/*
Date:18/05/2024
Developed by: Musa Almaz
Project: Project 5 - Optical Flow and Tracking
Summary: This is the header file for the tasks of the Project 5.
Basically includes functions for computing and visualizing optical flow and tracking.
*/
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;


class Project5 {

    public:

      /**
       * @brief Construct a new Project 5 object
       * 
       */
      Project5();

      /**
       * @brief Check the given index in the image and masks files
       * 
       * @param n 
       * @return true 
       * @return false 
       */
      bool checkIndexes(int n);

      /**
       * @brief Gets the images of indexes n and n+1 also masks
       * 
       * @param n Index of the image
       */
      void getImages(int n);

      /**
       * @brief The main function of the project
       * 
       */
      void run();



    private:
      
      /**
       * @brief Displays nth and (n+1)th images on the grid
       * 
       * @param img1 The image of index n
       * @param img2 The image of index (n+1)
       */
       void displayImagesInGrid_Task1(const cv::Mat& img1, const cv::Mat& img2);

        /**
         * @brief Function to load and display an image from a specific path
         * 
         * @param path 
         * @return cv::Mat 
         */
        cv::Mat loadImage(const std::string& path);

      /**
       * @brief Gets the indexes of the images in given directory
       * 
       * @param directory 
       * @return std::vector<int> Vector of indexes
       */
        std::vector<int> getImageIndexes(const std::string& directory);

      /**
       * @brief Gets the image path of given index and directory
       * 
       * @param directory
       * @param index 
       * @return std::string 
       */
        std::string getImagePathFromIndex(const std::string& directory, int index);

      /**
       * @brief Calculates the optical flow with calcOpticalFlowFarneback function
       * 
       * @param img1 
       * @param img2 
       * @param mask1 
       * @param mask2 
       * @param flowVisualization The image that the optical flow draw
       * @return cv::Point2f 
       */
        cv::Point2f applyOpticalFlow(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask1, const cv::Mat& mask2, cv::Mat& flowVisualization);

      /**
       * @brief Calculates the motion with calcOpticalFlowPyrLK function
       * 
       * @param img1 
       * @param img2 
       * @param mask1 
       * @param mask2 
       * @param trackVisualization The visualization image of the tracking result
       * @return * cv::Point2f 
       */
        cv::Point2f applyTracking(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask1, const cv::Mat& mask2, cv::Mat& trackVisualization);

      /**
       * @brief Function to compute norm of difference between two points
       * 
       * @param flow1 
       * @param flow2 
       * @return double 
       */
        double computeNormDifference(const cv::Point2f& flow1, const cv::Point2f& flow2);


        std::string imgDir = "../Data/tum_freiburg3_sitting_static/";
        std::string maskDir = "../Data/tum_freiburg3_sitting_static/masks/";

        std::vector<int> img_set_indexes;
        std::vector<int> mask_set_indexes;

        int total_file_number = 0;
        int number_of_images = 0;

        const std::string windowName = "Image Grid";

        std::vector<cv::Mat> images;

        cv::Mat gridImage;

        const int max_img_width = 640;
        const int max_img_height = 480;

        int flow_grid_size = 20;



};


