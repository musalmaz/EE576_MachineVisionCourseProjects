/*
Date:28/02/2024
Developed by: Musa Almaz
Project: Project 1 - Basic OpenCV
Summary: Source file of the "project1.hpp"
*/
#include "project1.hpp"

// Initialize the static member outside the class (in your source file)
BasicOpencv* BasicOpencv::instance = nullptr;

BasicOpencv::BasicOpencv(){
    instance = this;  // Set the global instance pointer

    // Go to directory and get the number of images
    // Iterate over the directory and store image filenames
    for (const auto & entry : fs::directory_iterator(dataFilePath)){
        numberofImages++;
        filenames.push_back(entry.path());
    }
}
cv::Mat BasicOpencv::readImage(int NumberofImage){
    // Load the nth image
    std::string filename = filenames[NumberofImage - 1];
    cv::Mat image = cv::imread(filename);

    if (image.empty()) {
        std::cerr << "Could not open or find the image." << std::endl;
    }

    //cv::imshow("image", image);
    //cv::waitKey(0);

    return image;
}

void BasicOpencv::mouseCallbackStatic(int event, int x, int y, int flags, void* userdata) {
    if (instance) {
        instance->mouseCallback(event, x, y, flags, userdata);
    }
}

void BasicOpencv::updateGridWithROI(int x, int y, bool isFirstImage) {


    // Calculate ROI bounds, ensuring they are within the image
    int roiX = std::max(0, x - halfSize);
    int roiY = std::max(0, y - halfSize);

    roiX = std::min(roiX, N2_prime - ROISize);
    roiY = std::min(roiY, N1_prime - ROISize);
    //std::cout << "roi x y : " << roiX << " " << roiY << "\n";
    cv::Rect roi(roiX, roiY, ROISize, ROISize);

    // Extract ROI and place it in the appropriate cell of the grid
    cv::Mat cell;
    if (isFirstImage) {
        cell = resizedImage(roi).clone();
        cv::resize(cell, cell, cv::Size(N2_prime, N1_prime));
        cell.copyTo(gridImage(cv::Rect(0, N1_prime, N2_prime, N1_prime))); // 3rd cell
    } else {
        cell = resizedRotatedImage(roi).clone();
        cv::resize(cell, cell, cv::Size(N2_prime, N1_prime));
        cell.copyTo(gridImage(cv::Rect(N2_prime, N1_prime, N2_prime, N1_prime))); // 4th cell
    }

    cv::imshow("Image Grid", gridImage); // Refresh the grid display
}

void BasicOpencv::mouseCallback(int event, int x, int y, int flags, void* userdata){
    if (event != cv::EVENT_LBUTTONDOWN) return;

    std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    // Check if click is on the first or second image
    if (x < N2_prime && y < N1_prime) {
        // Click is on the first image
        updateGridWithROI(x, y, true);
    } else if (x >= N2_prime && y < N1_prime) {
        // Click is on the second image
        x -= N2_prime; // Adjust x-coordinate for the second image
        updateGridWithROI(x, y, false);
    }
    
    
}


std::array<int, 2> BasicOpencv::getDimensions(cv::Mat image){
    std::cout << "Size of the image is : " << image.rows << " " << image.cols << "\n";
    return {image.rows, image.cols};
}

void BasicOpencv::displayImage(cv::Mat image){
    // Create a window
    cv::namedWindow("Image Grid", cv::WINDOW_AUTOSIZE);
    
    N1_prime = getDimensions(image)[0];
    N2_prime = getDimensions(image)[1];

    if(N1_prime > max_rows){
        N1_prime = max_rows;
    }
    if(N2_prime > max_cols){
        N2_prime = max_cols;
    }

    // Create an empty grid image of size 2N1' x 2N2'
    gridImage = cv::Mat(2 * N1_prime, 2 * N2_prime, image.type(), cv::Scalar::all(0));

    // Resize the original image if it's larger than N1' x N2'
    cv::resize(image, resizedImage, cv::Size(N2_prime, N1_prime));

    // Place the resized image in the first cell of the grid
    cv::Mat firstCell = gridImage(cv::Rect(0, 0, N2_prime, N1_prime));
    resizedImage.copyTo(firstCell);

    // Rotate the original image and place it in the second cell
    cv::Mat rotatedImage;
    cv::rotate(resizedImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
    // Ensure the rotated image fits into the cell (this might require another resize)
    
    cv::resize(rotatedImage, resizedRotatedImage, cv::Size(N2_prime, N1_prime));

    cv::Mat secondCell = gridImage(cv::Rect(N2_prime, 0, N2_prime, N1_prime));
    resizedRotatedImage.copyTo(secondCell);

    cv::setMouseCallback("Image Grid", mouseCallbackStatic, this);

    // Display the grid image and wait for a mouse event
    while (true) {
        cv::imshow("Image Grid", gridImage);

        // Check for 'Enter' key press
        int key = cv::waitKey(20); // Wait for 20 ms
        if (key == 13 || key == 10) { // 13 is Enter key on Windows, 10 on Unix
            break; // Exit the loop
        }
    }
}