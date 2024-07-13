/*
Date:10/03/2024
Developed by: Musa Almaz
Project: Project 2 -  Computing Homography Matrix
Summary: This is the header file for the tasks of the Project 2.
Basically includes functions for reading image, displaying image, and mouse events.
Additionally for computing the hompgraphy matrix based on the related points on the consequtive images.
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

class BasicOpencv{
    public:
        /*
        Date:10/03/2024
        Developed by: Musa Almaz
        Summary: This method is for finding homography matrix using the built-in funtions of OpenCV.
        Input: The vector of points for the two images
        Output: 
        Additional info: This uses findHomography and drawMatches functions of OpenCV
        */ 
        void findHomographyMap(const std::vector<cv::Point2d>& points1, 
                       const std::vector<cv::Point2d>& points2);

        /*
        Date:10/03/2024
        Developed by: Musa Almaz
        Summary: This method is for finding homography matrix using the psoude-inverse method
        Input: The vector of points for the two images
        Output: 
        Additional info: This uses psoude-inverse method to calculate the homography matrix.
        */ 
        void calculateHomographyMap(const std::vector<cv::Point2d>& points1, 
                        const std::vector<cv::Point2d>& points2);
        /*
        Date:28/02/2024
        Developed by: Musa Almaz
        Summary: This method is for reading the image from the directory
        Input: The index of the image in the directory
        Output: Returns the readed image
        Additional info:
        */ 
        cv::Mat readImage(int NumberofImage);

        /*
        Date:28/02/2024
        Developed by: Musa Almaz
        Summary: This method is for displaying the image
        Input: An image as cv::Mat object
        Output: 
        Additional info: It creates the first two cell of the grid
        */
        void displayImage();

        /*
        Date:28/02/2024
        Developed by: Musa Almaz
        Summary: This method is for calling the mouseCallback function
        Input: Event, x and y point that the curser touched, flag
        Output: 
        Additional info:
        */
        static void mouseCallbackStatic(int event, int x, int y, int flags, void* userdata);
        /*
        Date:28/02/2024
        Developed by: Musa Almaz
        Summary: This is the constructor of the class
        Input: No input
        Output: No output
        Additional info: This changes the value of the "numberofImages"
        */
        BasicOpencv();

        int numberofImages = 0; // total number of images in dataset
        int selected_image;

        std::vector<cv::Point2d> first_image_points;
        std::vector<cv::Point2d> second_image_points;

    private:

        /*
        Date:28/02/2024
        Developed by: Musa Almaz
        Summary: This is for the printing the size of the given image
        Input: An image as cv::Mat object
        Output: No output
        Additional info: 
        */
        std::array<int, 2> getDimensions(cv::Mat image);

        /*
        Date:28/02/2024
        Developed by: Musa Almaz
        Summary: This is actual mouseCallback function
        Input: Event, x and y coordintes, flags 
        Output: No output
        Additional info: This actions when the left mouse button pressed
        */
        void mouseCallback(int event, int x, int y, int flags, void* userdata);

        /*
        Date:28/02/2024
        Developed by: Musa Almaz
        Summary: This is updates the grid based on the pressing the mouse
        Input: x and y coordinate of the image, flag for first cell or second cell
        Output: No output
        */
        void updateGridWithROI(int x, int y, bool isFirstImage);
        
        std::string dataFilePath = "../Data"; //Data path of the images
        std::vector<std::string> filenames;
        

        cv::Mat gridImage; // 2x2 grid image
        cv::Mat resizedImage; // Resized image
        cv::Mat resize_second_image;

        int max_rows = 400;  // Define the max size of the rows, if the image is large, this is the cell row size
        int max_cols = 500;  // Define the max size of the cols, if the image is large, this is the cell column size

        int N1_prime;  // Cell size of the 2x2 grid (row)
        int N2_prime;  // Cell size of the 2x2 grid (column)

        const int ROISize = 50; // Example fixed size of ROI
        int halfSize = ROISize / 2;

        cv::Rect box;
        bool drawing_box = false;

        static BasicOpencv* instance;  // Global instance pointer

        

        std::vector<cv::Point2d> first_image_prob_points;
        std::vector<cv::Point2d> second_image_prob_points;

};