/*
Date:03/04/2024
Developed by: Musa Almaz
Project: Project 3 - Segmentation and Representation
Summary: This is the main file for the tasks of the Project 3.
*/
#include "project3.hpp"


int main() {
    
    Project3 project;

    // Mat image = imread("../components/L5050/5050_60.jpg");
    // project.applyInteractiveHSVFilter(image);

    std::vector<std::string> imagePaths = project.getImagePaths();
    for (const auto& imagePath : imagePaths) {
        // std::cout << imagePath << std::endl;
        
        Mat image = imread(imagePath, IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Image cannot be loaded!" << std::endl;
            return -1;
        }

        Mat segmentedImage = project.segmentGreenRegion(image);

        imwrite("segmented.png",segmentedImage);

        // Mat closedImage = project.closeImage(segmentedImage);
        Mat grayImage, binaryImage;
        cvtColor(segmentedImage, grayImage, COLOR_BGR2GRAY);
        threshold(grayImage, binaryImage, 1, 255, THRESH_BINARY);

        project.findContoursInRegion(binaryImage);

        Mat boundaryMarkedImage = Mat::zeros(image.size(), image.type());
        project.drawBoundaries(boundaryMarkedImage);
        imwrite("boundaryimage.png", boundaryMarkedImage);
        project.findLargestObject();

        Mat rotatedImage = project.rotateImage(image);
        Mat rotatedObject = project.extractAndRotateObject(image);
        imwrite("roated.png", rotatedObject);
        // Create a canvas for the 2x2 grid
        int rows = image.rows;
        int cols = image.cols;
        Mat canvas = Mat::zeros(rows * 2, cols * 2, image.type());

        // Copy each image into its respective position
        // Copy original image
        image.copyTo(canvas(Rect(0, 0, cols, rows)));
        segmentedImage.copyTo(canvas(Rect(cols, 0, cols, rows)));
        boundaryMarkedImage.copyTo(canvas(Rect(0, rows, cols, rows)));
        rotatedObject.copyTo(canvas(Rect(cols, rows, cols, rows)));

        std::string savePath = project.replaceDirectoryName(imagePath);
        // std::cout << savePath << std::endl;

        namedWindow("2x2 Grid Display", WINDOW_AUTOSIZE);
        imshow("2x2 Grid Display", canvas);
        waitKey(1000);

        if (project.saveImage(savePath, canvas)) {
            std::cout << "Image saved successfully." << std::endl;
        }

    }

    // // Representation part
    // std::vector<int> numWordsList = {50, 100, 200}; // Three alternative values of N

    // project.mapDirectoriesToLabels();
    // // Loop for each N
    // for (int N : numWordsList) {
    //     // Placeholder for storing descriptors from all images
    //     std::vector<Mat> allDescriptors;
    //     std::map<int, std::vector<Mat>> classFeatures;

    //     // Iterate through each class directory
    //     for (const auto& classEntry : fs::directory_iterator(project.newDir)) {
    //         if (fs::is_directory(classEntry)) {
    //             int classLabel = project.getClassLabel(classEntry);
    //             std::vector<Mat> images = project.getImagePaths(classEntry.path());

    //             // Extract features from images
    //             std::vector<std::vector<KeyPoint>> keypoints;
    //             std::vector<Mat> descriptors;
    //             project.extractFeatures(images, keypoints, descriptors);

    //             // Accumulate descriptors
    //             allDescriptors.insert(allDescriptors.end(), descriptors.begin(), descriptors.end());

    //             // Store descriptors for each class
    //             classFeatures[classLabel] = descriptors;
    //         }
    //     }

    //     // Create vocabulary
    //     Mat vocabulary = project.createVocabulary(allDescriptors, N);

    //     // Build BoW representations and calculate average descriptor for each class
    //     std::map<int, Mat> averageBOWDescriptors;
    //     for (auto& [classLabel, descriptors] : classFeatures) {
    //         Mat sumDescriptors;
    //         for (const Mat& desc : descriptors) {
    //             sumDescriptors += project.buildBOWRepresentation(desc, vocabulary);
    //         }
    //         averageBOWDescriptors[classLabel] = sumDescriptors / descriptors.size();
    //     }

    //     // Calculate dissimilarity matrix
    //     project.calculateDissimilarityMatrix(averageBOWDescriptors);
    // }


    return 0;
}