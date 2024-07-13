/*
Date:03/04/2024
Developed by: Musa Almaz
Project: Project 3 - Segmentation and Representation
Summary: This is the source file for the tasks of the Project 3.
*/
#include "project3.hpp"


Mat Project3::hsvImage;
Mat Project3::filteredImage;
Mat Project3::originalImage;
int Project3::lowerH = 0;
int Project3::lowerS = 0;
int Project3::lowerV = 0;
int Project3::upperH = 180;
int Project3::upperS = 255;
int Project3::upperV = 255;


void Project3::drawBoundaries(Mat& image) {
    // Check if we have enough contours
    if (contours.size() < 2) {
        std::cerr << "Not enough contours found to distinguish between outer boundary and object." << std::endl;
        return;
    }

    // Sort contours based on area in descending order (largest first)
    std::vector<std::vector<Point>> sortedContours = contours;
    sort(sortedContours.begin(), sortedContours.end(), [](const std::vector<Point>& a, const std::vector<Point>& b) {
        return contourArea(a, false) > contourArea(b, false);
    });

    // Outer boundary is the largest contour
    std::vector<Point> outerBoundary = sortedContours[0];

    // The object boundary is the second-largest contour
    std::vector<Point> objectBoundary = sortedContours[1];

    // Draw the outer boundary in green
    drawContours(image, std::vector<std::vector<Point>>{outerBoundary}, -1, Scalar(0, 255, 0), 2);

    // Draw the object boundary in blue
    drawContours(image, std::vector<std::vector<Point>>{objectBoundary}, -1, Scalar(255, 0, 0), 2);
}

Mat Project3::closeImage(const Mat &segmentedImage) {
    // Convert to grayscale and then to binary
    Mat grayImage, binaryImage;
    cvtColor(segmentedImage, grayImage, COLOR_BGR2GRAY);
    threshold(grayImage, binaryImage, 1, 255, THRESH_BINARY);

    Mat closedImage;
    int morphSize = 4; // Adjust this size according to your needs
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morphSize + 1, 2 * morphSize + 1), Point(morphSize, morphSize));
    morphologyEx(binaryImage, closedImage, MORPH_CLOSE, element);

    return closedImage;
}

void Project3::findContoursInRegion(const Mat& binaryImage) {
    // std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(binaryImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
}

void Project3::findLargestObject() { 
    if (contours.size() < 2) {
        return;
    }

    // Find the two largest contours based on contour area
    std::vector<int> indices(contours.size()); // Vector with the same size as contours
    iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ... 

    // Partial sort to get the indices of the two largest contours
    partial_sort(indices.begin(), indices.begin() + 2, indices.end(), [&](int a, int b) {
        return contourArea(contours[a], false) > contourArea(contours[b], false);
    });

    // Get the second largest contour by index
    const auto& secondLargestContour = contours[indices[1]];

    // Ensure that the points are in the correct type
    std::vector<Point2f> contourFloat;
    contourFloat.reserve(secondLargestContour.size());
    for (const Point& p : secondLargestContour) {
        contourFloat.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }

    // Calculate the minimum area rectangle for the second largest contour
    objectRect = minAreaRect(contourFloat);
}

Mat Project3::rotateImage(const Mat& image) {
    float angle = objectRect.angle;
    if (angle < -45.0) angle += 90.0; // Correct the angle

    // Get the rotation matrix 
    Mat rotationMatrix = getRotationMatrix2D(objectRect.center, angle, 1.0);

    // Perform the affine transformation (rotation)
    Mat rotatedImage;
    warpAffine(image, rotatedImage, rotationMatrix, image.size(), INTER_CUBIC);
    
    return rotatedImage;
}
Mat Project3::extractAndRotateObject(const Mat& image) {
    // Ensure the objectRect has been calculated
    if (objectRect.size.width <= 0 || objectRect.size.height <= 0) {
        std::cerr << "Object rectangle not set. Call findLargestObject first." << std::endl;
        return Mat();
    }

    // Get the points of the rotated rectangle
    Point2f rectPoints[4];
    objectRect.points(rectPoints);

    // Convert the points to a contour (vector of points)
    std::vector<Point> contourPoly(rectPoints, rectPoints + 4);

    // Create mask for the object
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    drawContours(mask, std::vector<std::vector<Point>>{contourPoly}, 0, Scalar(255), FILLED);

    // Extract the object
    Mat object;
    image.copyTo(object, mask);

    // Rotate the object
    float angle = objectRect.angle;
    if (angle < -45.0) angle += 90.0;

    Mat rotationMatrix = getRotationMatrix2D(objectRect.center, -angle, 1.0);
    Mat rotatedObject;
    warpAffine(object, rotatedObject, rotationMatrix, image.size(), INTER_CUBIC);

    return rotatedObject;
}

Mat Project3::segmentGreenRegion(const Mat &image) {

    if(image.empty()) {
        std::cerr << "Error: Image cannot be loaded!" << std::endl;
        return Mat();
    }

    // Convert to HSV color space
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Define the range of green color in HSV
    Scalar lowerGreen(40, 45, 41); // Adjusted values 40 29 83
    Scalar upperGreen(105, 193, 240);         // 105 143 240

    // Threshold the image to get only green colors
    Mat greenMask;
    inRange(hsvImage, lowerGreen, upperGreen, greenMask);

    // Create a colorful filtered image
    Mat colorfulFilteredImage;
    bitwise_and(image, image, colorfulFilteredImage, greenMask);

    return colorfulFilteredImage;
}

// Callback function for trackbar event
void Project3::on_trackbar(int, void*) {
    Scalar lowerHSV(lowerH, lowerS, lowerV);
    Scalar upperHSV(upperH, upperS, upperV);

    // Generate the mask
    Mat mask;
    inRange(hsvImage, lowerHSV, upperHSV, mask);

    // Create a colorful filtered image
    Mat colorfulFilteredImage;
    bitwise_and(originalImage, originalImage, colorfulFilteredImage, mask);

    imshow("Colorful Filtered Image", colorfulFilteredImage);
}

void Project3::applyInteractiveHSVFilter(const Mat &image) {
    originalImage = image;
    // Convert the image to HSV color space
    cvtColor(originalImage, hsvImage, COLOR_BGR2HSV);

    // Create window
    namedWindow("Filtered Image", WINDOW_AUTOSIZE);

    // Create trackbars
    createTrackbar("Lower H", "Filtered Image", &lowerH, 180, on_trackbar);
    createTrackbar("Lower S", "Filtered Image", &lowerS, 255, on_trackbar);
    createTrackbar("Lower V", "Filtered Image", &lowerV, 255, on_trackbar);
    createTrackbar("Upper H", "Filtered Image", &upperH, 180, on_trackbar);
    createTrackbar("Upper S", "Filtered Image", &upperS, 255, on_trackbar);
    createTrackbar("Upper V", "Filtered Image", &upperV, 255, on_trackbar);

    // Initial call to update the display
    on_trackbar(0, 0);

    // Wait for a key press
    while (true) {
        char key = (char)waitKey(1);
        if (key == 27) break; // Exit on ESC
    }

    destroyWindow("Filtered Image");
}

std::vector<std::string> Project3::getImagePaths() {
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::recursive_directory_iterator(directoryPath)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg" || entry.path().extension() == ".png")) {
            imagePaths.push_back(entry.path().string());
        }
    }
    return imagePaths;
}

std::string Project3::replaceDirectoryName(const std::string& originalPath) {
    size_t pos = originalPath.find(oldDir);
    if (pos != std::string::npos) {
        // Replace old directory name with new directory name
        return originalPath.substr(0, pos) + newDir + originalPath.substr(pos + oldDir.length());
    }
    return originalPath; // Return the original path if the old directory name is not found
}

bool Project3::saveImage(const std::string& savePath, const cv::Mat& canvas) {
    if (canvas.empty()) {
        std::cerr << "Error: Image data is empty." << std::endl;
        return false;
    }

    fs::path path(savePath);

    // Check and create the parent directory if it doesn't exist
    if (!fs::exists(path.parent_path())) {
        if (!fs::create_directories(path.parent_path())) {
            std::cerr << "Error: Could not create directories." << std::endl;
            return false;
        }
    }

    // Save the image
    if (!cv::imwrite(savePath, canvas)) {
        std::cerr << "Error: Could not write image to the path provided." << std::endl;
        return false;
    }

    return true;
}

// // Function to extract keypoints and descriptors
// void Project3::extractFeatures(const std::vector<Mat>& images, std::vector<std::vector<KeyPoint>>& keypoints, std::vector<Mat>& descriptors) {
//     Ptr<Feature2D> extractor = xfeatures2d::SIFT::create();   // Choose SIFT or SURF
//     for (const Mat& img : images) {
//         std::vector<KeyPoint> kp;
//         Mat desc;
//         extractor->detectAndCompute(img, Mat(), kp, desc);
//         keypoints.push_back(kp);
//         descriptors.push_back(desc);
//     }
// }

// // Function to create vocabulary using KMeans
// Mat Project3::createVocabulary(const std::vector<Mat>& descriptors, int numWords) {
//     Mat allDescriptors;
//     for (const Mat& desc : descriptors) {
//         allDescriptors.push_back(desc.t());
//     }
//     allDescriptors = allDescriptors.reshape(1, 0);

//     Mat vocabulary;
//     kmeans(allDescriptors, numWords, vocabulary, 
//             TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0), 
//             10, KMEANS_RANDOM_CENTERS);
//     return vocabulary.t();
// }

// // Function to build BOW representation
// Mat Project3::buildBOWRepresentation(const Mat& descriptor, const Mat& vocabulary) {
//     BOWImgDescriptorExtractor extractor(SIFT::create()); // Choose SIFT or SURF
//     extractor.setVocabulary(vocabulary);
//     Mat bowDescriptor;
//     extractor.compute(descriptor, Mat(), bowDescriptor);
//     return bowDescriptor.clone();
// }

// // Function to calculate dissimilarity between two BOW representations
// double Project3::calculateDissimilarity(const Mat& bow1, const Mat& bow2) {
//     return norm(bow1, bow2, NORM_L2);
// }

// void Project3::calculateDissimilarityMatrix(const std::map<int, Mat>& averageBOWDescriptors){
//     Mat dissimilarityMatrix = Mat::zeros(numClasses, numClasses, CV_64F);
//     for (int i = 0; i < numClasses; i++) {
//         for (int j = 0; j < numClasses; j++) {
//             if (i != j) {
//                 double dissimilarity = 0.0;
//                 int Mi = classFeatures[i].size();
//                 int Mj = classFeatures[j].size();
//                 for (int k = 0; k < Mi; k++) {
//                     for (int l = 0; l < Mj; l++) {
//                         dissimilarity += calculateDissimilarity(averageBOWDescriptors[i], averageBOWDescriptors[j]);
//                     }
//                 }
//                 dissimilarityMatrix.at<double>(i, j) = dissimilarity / (Mi * Mj);
//             }
//         }
//     }

//     // Print or save the dissimilarity matrix
//     std::cout << dissimilarityMatrix << std::endl;
// }

// std::map<std::string, int> Project3::mapDirectoriesToLabels() {
    
//     int label = 0;  // Starting label

//     for (const auto& entry : std::filesystem::directory_iterator(newDir)) {
//         if (std::filesystem::is_directory(entry)) {
//             std::string dirName = entry.path().filename().string(); // Get the directory name
//             directoryLabels[dirName] = label++;
//         }
//     }

//     return directoryLabels;
// }

// // Function to get the class label based on the directory entry
// int Project3::getClassLabel(const std::filesystem::directory_entry& classEntry) {
//     std::string dirName = classEntry.path().filename().string();
//     auto it = directoryLabels.find(dirName);
//     if (it != directoryLabels.end()) {
//         return it->second;  // Return the found label
//     } else {
//         return -1;  // Return -1 or another indicator if the directory name is not found in the map
//     }
// }