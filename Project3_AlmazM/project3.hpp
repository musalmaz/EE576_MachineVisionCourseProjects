/*
Date:03/04/2024
Developed by: Musa Almaz
Project: Project 3 - Segmentation and Representation
Summary: This is the header file for the tasks of the Project 3.
Basically includes functions for segmenting the image, drawing contours and representing the image.
*/
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
// #include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp> 
// #include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <map>
#include <iostream>
#include <iostream>
#include <vector>
#include <stack>
#include <numeric>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

using namespace cv;

class Project3{
    public:
        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for drawing boundaries of object and the related region
        Input: The image that will be edited
        Output: No output
        Additional info:
        */ 
        void drawBoundaries(Mat& image);

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for segmenting the green region
        Input: The original image 
        Output: The segmented image
        Additional info:
        */
        Mat segmentGreenRegion(const Mat &image);

        /*
        Date: 03/04/2024
        Developed by: Musa Almaz
        Summary: Applies morphological closing to the segmented image.
        Input: A colorful segmented image (const Mat&).
        Output: Processed image (Mat) after applying morphological closing.
        Additional info: Uses dilation followed by erosion to close gaps.
        */
        Mat closeImage(const Mat &segmentedImage);

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for finding contours of binary image
        Input: The binary image 
        Output: No output
        Additional info: It edits the private variable "contours"
        */
        void findContoursInRegion(const Mat& binaryImage);

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for finding the object inside green region
        Input: No input
        Output: No output
        Additional info:
        */
        void findLargestObject();

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for rotating the image based on the rotation angle of the object
        Input: The original image 
        Output: The rotated image
        Additional info:
        */
        Mat rotateImage(const Mat& image);

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for extracting object from the image and rotationg it
        Input: The original image 
        Output: The rotated object
        Additional info:
        */
        Mat extractAndRotateObject(const Mat& image);

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is callback function of trackbar
        Input: The related HSV values
        Output: 
        Additional info:
        */
        static void on_trackbar(int, void*);

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for finding parameters for HSV filtering
        Input: The original image
        Output: 
        Additional info:
        */
        void applyInteractiveHSVFilter(const Mat &image);

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for getting the relative path of the images
        Input: 
        Output: The related path of all images in a vector
        Additional info: It uses directory path
        */
        std::vector<std::string> getImagePaths();

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for changing the relative path as desired
        Input: The original path taht will be changed
        Output: The changed path as desired
        Additional info: It uses oldDir and newDir
        */
        std::string replaceDirectoryName(const std::string& originalPath);

        /*
        Date:03/04/2024
        Developed by: Musa Almaz
        Summary: This method is for saving the image to given path
        Input: The path of the image that will be saved and the image
        Output: No output
        Additional info:
        */
        bool saveImage(const std::string& savePath, const cv::Mat& canvas);

        /*
        Date: 03/04/2024
        Developed by: Musa Almaz
        Summary: Extracts keypoints and descriptors from a set of images using feature detection.
        Input: A vector of images (const std::vector<Mat>&).
        Output: Fills two vectors, one with keypoints and the other with descriptors (std::vector<std::vector<KeyPoint>>& and std::vector<Mat>& respectively).
        Additional info: Uses a feature extractor like ORB or SIFT for feature detection and description.
        */
        void extractFeatures(const std::vector<Mat>& images, std::vector<std::vector<KeyPoint>>& keypoints, std::vector<Mat>& descriptors);

        /*
            Date: 03/04/2024
            Developed by: Musa Almaz
            Summary: Creates a visual vocabulary (bag of words) using KMeans clustering on the descriptors.
            Input: A vector of descriptors (const std::vector<Mat>&) and the desired number of words in the vocabulary (int).
            Output: The visual vocabulary as a Mat object.
            Additional info: This vocabulary is used for creating a BoW representation of the images.
        */
        Mat createVocabulary(const std::vector<Mat>& descriptors, int numWords);

        /*
            Date: 03/04/2024
            Developed by: Musa Almaz
            Summary: Builds a Bag of Words representation of an image descriptor.
            Input: A single image descriptor (const Mat&) and the vocabulary (const Mat&).
            Output: The BoW representation of the image as a Mat object.
            Additional info: Converts the set of keypoints into a histogram of word occurrences.
        */
        Mat buildBOWRepresentation(const Mat& descriptor, const Mat& vocabulary);

        /*
            Date: 03/04/2024
            Developed by: Musa Almaz
            Summary: Calculates the dissimilarity between two BoW descriptors.
            Input: Two BoW descriptors (const Mat&).
            Output: The dissimilarity score (double).
            Additional info: Uses L2 norm for calculating dissimilarity.
        */
        double calculateDissimilarity(const Mat& bow1, const Mat& bow2);

        /*
            Date: 03/04/2024
            Developed by: Musa Almaz
            Summary: Calculates a dissimilarity matrix between different classes based on their average BoW descriptors.
            Input: A map associating class labels with their average BoW descriptors (const std::map<int, Mat>&).
            Output: A dissimilarity matrix (Mat).
            Additional info: Each matrix element represents the average dissimilarity between two classes.
        */
        void calculateDissimilarityMatrix(const std::map<int, Mat>& averageBOWDescriptors);

        /*
            Date: 03/04/2024
            Developed by: Musa Almaz
            Summary: It maps classnames to integers
            Input: 
            Output: A map of classnames and their responding numbers
            Additional info: 
        */
        std::map<std::string, int> mapDirectoriesToLabels();

        /*
            Date: 03/04/2024
            Developed by: Musa Almaz
            Summary: It returns the class label
            Input: 
            Output: class label as an integer
            Additional info: 
        */
        int getClassLabel(const std::filesystem::directory_entry& classEntry);

        std::string newDir = "LearningData";


        

    private:
        static Mat hsvImage;
        static Mat filteredImage;
        static int lowerH, lowerS, lowerV;
        static int upperH, upperS, upperV;
        static Mat originalImage;

        Scalar outerColor = Scalar(0, 100, 255);
        Scalar innerColor = Scalar(100, 255, 0);

        std::vector<std::vector<Point>> contours;

        RotatedRect objectRect;

        std::string directoryPath = "../components";

        std::string oldDir = "components";

        std::map<std::string, int> directoryLabels;

        int numClasses = 13;
};