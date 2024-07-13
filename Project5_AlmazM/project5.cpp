/*
Date:18/05/2024
Developed by: Musa Almaz
Project: Project 5 - Optical Flow and Tracking
Summary: This is the source file for the tasks of the Project 5.
Basically includes function definitions.
*/
#include "project5.hpp"

Project5::Project5(){
    img_set_indexes = getImageIndexes(imgDir);
    number_of_images = total_file_number;
    mask_set_indexes = getImageIndexes(maskDir);
    std::cout <<"There is : " << number_of_images << " images, and "<<total_file_number - number_of_images << " mask images found. \n";

}

cv::Mat Project5::loadImage(const std::string& path){
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        exit(1);
    }
    return image;
}

std::vector<int> Project5::getImageIndexes(const std::string& directory) {
    std::vector<int> indexes;

    // Iterate through the directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            if (extension == ".jpg" || extension == ".png") {
                // Extract index from filename
                std::string indexStr;
                for (char ch : filename) {
                    if (std::isdigit(ch)) {
                        indexStr += ch;
                    }
                }
                if (!indexStr.empty()) {
                    indexes.push_back(std::stoi(indexStr));
                    total_file_number++;
                }
                // std::cout << "File: " << filename << " Index: " << indexStr << "\n";
            }
        }
    }

    // std::cout << "Total images found: " << file_number << std::endl;
    return indexes;
}


std::string Project5::getImagePathFromIndex(const std::string& directory, int index) {
    std::string filename;
    std::string path;

    // Possible image extensions
    std::vector<std::string> extensions = {".png", ".jpg"};

    // Create a filename based on the index
    filename = std::to_string(index);

    // Try all possible extensions
    for (const auto& ext : extensions) {
        path = directory + filename + ext;

        // Check if the file exists in the directory
        if (fs::exists(path)) {
            // std::cout << "Found: " << path << std::endl;
            return path;
        }
    }

    // If none of the files were found, return an empty string
    std::cout << "Image with index " << index << " not found in " << directory << std::endl;
    return "";
}

void Project5::displayImagesInGrid_Task1(const cv::Mat& img1, const cv::Mat& img2){

    // Resize images for uniformity
    cv::resize(img1, img1, cv::Size(max_img_width, max_img_height));
    cv::resize(img2, img2, cv::Size(max_img_width, max_img_height));

    // Create a large image to combine the smaller images
    gridImage.create(2 * max_img_height, 2 * max_img_width, img1.type()); // Adjusted to stack horizontally

    // Place images in their respective grid positions
    img1.copyTo(gridImage(cv::Rect(0, 0, max_img_width, max_img_height)));
    img2.copyTo(gridImage(cv::Rect(max_img_width, 0, max_img_width, max_img_height)));

}

// Function to compute average flow in masked regions
cv::Point2f computeAverageFlow(const cv::Mat& flow, const cv::Mat& mask1, const cv::Mat& mask2) {
    cv::Point2f averageFlow(0.0f, 0.0f);
    int count = 0;

    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            if (mask1.at<uchar>(y, x) > 0 && mask2.at<uchar>(y, x) > 0) {
                cv::Point2f flow_at_point = flow.at<cv::Point2f>(y, x);
                averageFlow += flow_at_point;
                count++;
            }
        }
    }

    if (count > 0) {
        averageFlow.x /= count;
        averageFlow.y /= count;
    }

    return averageFlow;
}
cv::Point2f Project5::applyOpticalFlow(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask1, const cv::Mat& mask2, cv::Mat& flowVisualization) {
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(img1, img2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    // Mask img1 before converting to BGR for visualization
    cv::Mat maskedImg1;
    img1.copyTo(maskedImg1, mask1);
    cv::cvtColor(maskedImg1, flowVisualization, cv::COLOR_GRAY2BGR);


    // Loop through each pixel or sub-sampling grid
    for (int y = 0; y < flow.rows; y += flow_grid_size) {
        for (int x = 0; x < flow.cols; x += flow_grid_size) {
            // Get the flow vector at this position
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);

            // Draw an arrow from the current position to the position dictated by the flow vector
            cv::arrowedLine(flowVisualization, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), cv::Scalar(0, 255, 0), 1, 8, 0, 0.3);
        }
    }

    // Compute and return the average flow (optional if only visualization is needed)
    cv::Point2f averageFlow = computeAverageFlow(flow, mask1, mask2);
    return averageFlow;
}


cv::Point2f Project5::applyTracking(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask1, const cv::Mat& mask2, cv::Mat& trackVisualization) {
    // Detect features in the first image using mask1
    std::vector<cv::Point2f> points1, points2;
    cv::goodFeaturesToTrack(img1, points1, 100, 0.01, 10, mask1);
    std::vector<uchar> status;
    std::vector<float> err;
    
    // Track these features in the second image
    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, err, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), 0, 1e-4);

    // Prepare to calculate average flow
    cv::Point2f averageTrackFlow(0.0f, 0.0f);
    int trackCount = 0;

    cv::Mat maskedImg1;
    img1.copyTo(maskedImg1, mask1);
    cv::cvtColor(maskedImg1, trackVisualization, cv::COLOR_GRAY2BGR);

    // Draw individual flow vectors only where they are valid in both masks
    for (size_t i = 0; i < points1.size(); i++) {
        if (status[i] && mask2.at<uchar>(points2[i]) > 0 && mask1.at<uchar>(points1[i]) > 0) {
            cv::Point2f flowVec = points2[i] - points1[i];
            averageTrackFlow += flowVec;
            trackCount++;
            
            // Draw an arrow from the original to the new position
            cv::arrowedLine(trackVisualization, points1[i], points2[i], cv::Scalar(255, 0, 0), 1, 8, 0, 0.3);
        }
    }

    // Calculate average flow vector
    if (trackCount > 0) {
        averageTrackFlow.x /= trackCount;
        averageTrackFlow.y /= trackCount;
    }

    return averageTrackFlow;
}
// Function to compute norm of difference between two points
double Project5::computeNormDifference(const cv::Point2f& flow1, const cv::Point2f& flow2) {
    return cv::norm(flow1 - flow2);
}
void Project5::run(){
    // // Load the consecutive images
    cv::Mat img1_org = images[0];
    cv::Mat img2_org = images[1];

    displayImagesInGrid_Task1(img1_org, img2_org);

    // Convert to grayscale
    cv::Mat img1, img2;
    cv::cvtColor(img1_org, img1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2_org, img2, cv::COLOR_BGR2GRAY);


    // Load mask and ensure it is a single channel 8-bit image
    cv::Mat mask1 = images[2];
    cv::Mat mask2 = images[3];
    
    if (mask1.type() != CV_8U) {
        cv::cvtColor(mask1, mask1, cv::COLOR_BGR2GRAY);
        mask1.convertTo(mask1, CV_8U);
    }
    if (mask2.type() != CV_8U) {
        cv::cvtColor(mask2, mask2, cv::COLOR_BGR2GRAY);
        mask2.convertTo(mask2, CV_8U);
    }

    // Apply optical flow algorithm
    cv::Mat flowVisualization;
    cv::Point2f averageFlow = applyOpticalFlow(img1, img2, mask1, mask2, flowVisualization);

    // Apply tracking algorithm
    cv::Mat trackVisualization;
    cv::Point2f averageTrackFlow = applyTracking(img1, img2, mask1, mask2, trackVisualization);

    cv::imwrite("flow_visualization.png", flowVisualization);
    cv::imwrite("track_visualization.png", trackVisualization);
    // Compute the norm of the difference between the two average flows
    double normDiff = computeNormDifference(averageFlow, averageTrackFlow);
    std::cout << "Norm of the difference: " << normDiff << std::endl;

    // Resize images for uniformity
    cv::resize(flowVisualization, flowVisualization, cv::Size(max_img_width, max_img_height));
    cv::resize(trackVisualization, trackVisualization, cv::Size(max_img_width, max_img_height));

    flowVisualization.copyTo(gridImage(cv::Rect(0, max_img_height, max_img_width, max_img_height)));
    trackVisualization.copyTo(gridImage(cv::Rect(max_img_width, max_img_height, max_img_width, max_img_height)));

    cv::imwrite("grid_image.png", gridImage);
    images.clear();
    // Display the grid
    cv::imshow(windowName, gridImage);
    while (true) {
        int key = cv::waitKey(1); // Wait for a key press with a short delay

        // Check if the space key (ASCII value 32) is pressed
        if (key == ' ') {
            std::cout << "Space key pressed, closing window...\n";
            cv::destroyWindow(windowName); // Close the window
            break;
        } 
    }

}

void Project5::getImages(int n){
    // Load the consecutive images
    cv::Mat img1_org = loadImage(getImagePathFromIndex(imgDir, n));
    cv::Mat img2_org = loadImage(getImagePathFromIndex(imgDir, n + 1));

    // Load mask and ensure it is a single channel 8-bit image
    cv::Mat mask1 = loadImage(getImagePathFromIndex(maskDir, n));
    cv::Mat mask2 = loadImage(getImagePathFromIndex(maskDir, n + 1));

    images.push_back(img1_org);
    images.push_back(img2_org);
    images.push_back(mask1);
    images.push_back(mask2);

}

bool Project5::checkIndexes(int n){
    // Check if n and n+1 are in img_set_indexes
    bool img_check = std::find(img_set_indexes.begin(), img_set_indexes.end(), n) != img_set_indexes.end() &&
                     std::find(img_set_indexes.begin(), img_set_indexes.end(), n + 1) != img_set_indexes.end();

    // Check if n and n+1 are in mask_set_indexes
    bool mask_check = std::find(mask_set_indexes.begin(), mask_set_indexes.end(), n) != mask_set_indexes.end() &&
                      std::find(mask_set_indexes.begin(), mask_set_indexes.end(), n + 1) != mask_set_indexes.end();

    // Return true if both checks are true
    return img_check && mask_check;
}
