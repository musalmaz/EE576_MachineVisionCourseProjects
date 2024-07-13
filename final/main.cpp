#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

// Global variables for trackbar positions
int lowH = 0, highH = 179, lowS = 0, highS = 255, lowV = 0, highV = 255;

// Callback function for trackbar events
void on_trackbar(int, void* userdata) {
    Mat* hsv = reinterpret_cast<Mat*>(userdata);

    // Threshold the HSV image within the user-defined range
    Mat mask;
    inRange(*hsv, Scalar(lowH, lowS, lowV), Scalar(highH, highS, highV), mask);

    // Display the masked image
    imshow("Object Detection", mask);
}

// Function to add trackbars and initialize the image processing
void createTrackbarsAndProcess(Mat& image) {
    namedWindow("Control", WINDOW_AUTOSIZE); // Create a window called "Control"

    // Create trackbars in "Control" window
    createTrackbar("Low Hue", "Control", &lowH, 179, on_trackbar, &image);
    createTrackbar("High Hue", "Control", &highH, 179, on_trackbar, &image);
    createTrackbar("Low Sat", "Control", &lowS, 255, on_trackbar, &image);
    createTrackbar("High Sat", "Control", &highS, 255, on_trackbar, &image);
    createTrackbar("Low Value", "Control", &lowV, 255, on_trackbar, &image);
    createTrackbar("High Value", "Control", &highV, 255, on_trackbar, &image);

    // Initial call to the trackbar handler
    on_trackbar(0, &image);
}

int main(int argc, char **argv){

    cv::Mat apple_img = cv::imread("../apple.jpg");

    Mat hsvImage;
    cvtColor(apple_img, hsvImage, COLOR_BGR2HSV);

    // createTrackbarsAndProcess(hsvImage);
    // Define the range of the hue for the object (adjust these values)
    Mat1b mask1, mask2;
    inRange(hsvImage, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
    inRange(hsvImage, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);

    Mat1b mask = mask1 | mask2;
   
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Check if any contours are found
    if (contours.empty()) {
        std::cout << "No object found!" << std::endl;
    }

   // Find the bounding rectangle for the first contour
    Rect objectBoundingRect = boundingRect(contours[0]);

    // Output dimensions
    cout << "Object Length Apple (Height): " << objectBoundingRect.height << " pixels" << endl;
    cout << "Object Width Apple: " << objectBoundingRect.width << " pixels" << endl;

    // Calculate and output the mean hue of the object
    Scalar meanVal = mean(hsvImage, mask);
    cout << "Mean Hue Apple: " << meanVal[0] << endl; // Displaying the hue component

    // banana part

    Mat banana_img = imread("../banana.jpg"); // Assuming apple_img is already loaded with an image.
    Mat hsvImageB;
    cvtColor(banana_img, hsvImageB, COLOR_BGR2HSV);

    // Define the range of the hue for the object (adjust these values)
    Scalar lowerBound(30, 50, 50); // Example for green objects
    Scalar upperBound(80, 255, 255);

    // Threshold the HSV image to get only the object colors
    Mat maskB;
    inRange(hsvImageB, Scalar(20, 100, 100), Scalar(30, 255, 255), maskB);

    // Find contours
    vector<vector<Point>> contoursB;
    findContours(maskB, contoursB, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find the bounding rectangle for the first contour
    Rect objectBoundingRect_ = boundingRect(contoursB[0]);

    // Output dimensions
    cout << "Object Length banana (Height): " << objectBoundingRect_.height << " pixels" << endl;
    cout << "Object Width Banana: " << objectBoundingRect_.width << " pixels" << endl;

    // Calculate and output the mean hue of the object
    Scalar meanVal_ = mean(hsvImageB, maskB);
    cout << "Mean Hue Banana: " << meanVal_[0] << endl; // Displaying the hue component

    // pear part 

    Mat pear_img = imread("../pear.jpg"); // Assuming apple_img is already loaded with an image.
    Mat hsvImageP;
    cvtColor(pear_img, hsvImageP, COLOR_BGR2HSV);

    // createTrackbarsAndProcess(hsvImageP);
    // Threshold the HSV image to get only the object colors
    Mat maskP;
    inRange(hsvImageP, Scalar(20, 100, 100), Scalar(30, 255, 255), maskP);

    // Find contours
    vector<vector<Point>> contoursP;
    findContours(maskP, contoursP, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find the bounding rectangle for the first contour
    Rect objectBoundingRect_p = boundingRect(contoursP[0]);

    // Output dimensions
    cout << "Object Length pear (Height): " << objectBoundingRect_p.height << " pixels" << endl;
    cout << "Object Width pear: " << objectBoundingRect_p.width << " pixels" << endl;

    // Calculate and output the mean hue of the object
    Scalar meanVal_p = mean(hsvImageP, maskP);
    cout << "Mean Hue pear: " << meanVal_p[0] << endl; // Displaying the hue component

    cv::imshow("apple", mask);
    cv::imshow("banana", maskB);
    cv::imshow("banana", maskP);
    cv::waitKey(0);

    return 0;
}