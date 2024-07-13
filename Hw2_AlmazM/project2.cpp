/*
Date:10/03/2024
Developed by: Musa Almaz
Project: Project 2 - Computing Homography Matrix
Summary: Source file of the "project2.hpp"
*/
#include "project2.hpp"

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
        cell = resize_second_image(roi).clone();
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
        cv::Point2d p;
        p.x = x;
        p.y = y;
        first_image_prob_points.push_back(p);
        updateGridWithROI(x, y, true);
    } else if (x >= N2_prime && y < N1_prime) {
        // Click is on the second image
        cv::Point2d p;
        x -= N2_prime; // Adjust x-coordinate for the second image
        p.x = x;
        p.y = y;
        second_image_prob_points.push_back(p);
        updateGridWithROI(x, y, false);
    }
    
    
}


std::array<int, 2> BasicOpencv::getDimensions(cv::Mat image){
    std::cout << "Size of the image is : " << image.rows << " " << image.cols << "\n";
    return {image.rows, image.cols};
}

void BasicOpencv::displayImage(){
    // Create a window
    cv::namedWindow("Image Grid", cv::WINDOW_AUTOSIZE);
    cv::Mat first_image =readImage(selected_image);
    N1_prime = getDimensions(first_image)[0];
    N2_prime = getDimensions(first_image)[1];

    if(N1_prime > max_rows){
        N1_prime = max_rows;
    }
    if(N2_prime > max_cols){
        N2_prime = max_cols;
    }

    // Create an empty grid image of size 2N1' x 2N2'
    gridImage = cv::Mat(2 * N1_prime, 2 * N2_prime, first_image.type(), cv::Scalar::all(0));

    // Resize the first image if it's larger than N1' x N2'
    
    cv::resize(first_image, resizedImage, cv::Size(N2_prime, N1_prime));

    // Place the resized image in the first cell of the grid
    cv::Mat firstCell = gridImage(cv::Rect(0, 0, N2_prime, N1_prime));
    resizedImage.copyTo(firstCell);

    cv::Mat second_image = readImage(selected_image + 1);
    
    cv::resize(second_image, resize_second_image, cv::Size(N2_prime, N1_prime));

    cv::Mat secondCell = gridImage(cv::Rect(N2_prime, 0, N2_prime, N1_prime));
    resize_second_image.copyTo(secondCell);

    cv::setMouseCallback("Image Grid", mouseCallbackStatic, this);

    // Display the grid image and wait for a mouse event
    while (true) {
        cv::imshow("Image Grid", gridImage);

        
        int key = cv::waitKey(20); // Wait for 20 ms
        if (key == 's' || key == 'S') {
            std::cout << "S key is pressed, points are saved." << std::endl;
            if (!first_image_prob_points.empty()) {
                first_image_points.push_back(first_image_prob_points.back());
            }
            if (!second_image_prob_points.empty()) {
                second_image_points.push_back(second_image_prob_points.back());
            }
            // Clear the probable points
            first_image_prob_points.clear();
            second_image_prob_points.clear();
            // Print for debug
            for (const auto& point : first_image_points) {
                std::cout << "Selected first image points: " << point.x << " " << point.y << "\n";
            }
        }
        // Check for 'Enter' key press
        if (key == 13 || key == 10) { // 13 is Enter key on Windows, 10 on Unix
            break; // Exit the loop
        }
    }
}

void BasicOpencv::calculateHomographyMap(const std::vector<cv::Point2d>& points1, const std::vector<cv::Point2d>& points2){
    // Check if there are enough points
    int N = points1.size();
    if (N < 4 || points2.size() != N) {
        std::cerr << "Need at least 4 pairs of corresponding points." << std::endl;
        return;
    }
    // To find the Homography matrix (h) we need to solve: A * h = a
    // Then solve for h using pseudo-inverse method
    // Construct the A and a matrix
    // This part - for loop - is writed with ChatGPT
    cv::Mat A(2 * N, 8, CV_32F);
    cv::Mat a(2 * N, 1, CV_32F);
    for (int i = 0; i < N; ++i) {
        float x = static_cast<float>(first_image_points[i].x);
        float y = static_cast<float>(first_image_points[i].y);
        float x_prime = static_cast<float>(second_image_points[i].x);
        float y_prime = static_cast<float>(second_image_points[i].y);
        A.at<float>(2 * i, 0) = x;
        A.at<float>(2 * i, 1) = y;
        A.at<float>(2 * i, 2) = 1;
        A.at<float>(2 * i, 3) = 0;
        A.at<float>(2 * i, 4) = 0;
        A.at<float>(2 * i, 5) = 0;
        A.at<float>(2 * i, 6) = -x * x_prime;
        A.at<float>(2 * i, 7) = -y * x_prime;
        a.at<float>(2 * i) = x_prime;
        A.at<float>(2 * i + 1, 0) = 0;
        A.at<float>(2 * i + 1, 1) = 0;
        A.at<float>(2 * i + 1, 2) = 0;
        A.at<float>(2 * i + 1, 3) = x;
        A.at<float>(2 * i + 1, 4) = y;
        A.at<float>(2 * i + 1, 5) = 1;
        A.at<float>(2 * i + 1, 6) = -x * y_prime;
        A.at<float>(2 * i + 1, 7) = -y * y_prime;
        a.at<float>(2 * i + 1) = y_prime;
    }

    // Solve for h using the pseudo-inverse method
    cv::Mat h = (A.t() * A).inv() * A.t() * a;

    // Reshape h into a 3x3 matrix
    cv::Mat H = (cv::Mat_<float>(3, 3) << h.at<float>(0), h.at<float>(1), h.at<float>(2),
                                    h.at<float>(3), h.at<float>(4), h.at<float>(5),
                                    h.at<float>(6), h.at<float>(7), 1);

    std::cout << "Calculated Homography Matrix H:" << H <<  std::endl;

    // Load images 
    cv::Mat image1 = readImage(selected_image);
    cv::Mat image2 = readImage(selected_image + 1);

    // Concatenate images horizontally into a 1x2 grid
    cv::Mat grid;
    cv::hconcat(image1, image2, grid);

    // Apply homography to points in the first image to find corresponding points in the second image
    std::vector<cv::Point2f> matched_points;
    for (const auto& point : first_image_points) {
        cv::Mat homogenous_point = (cv::Mat_<float>(3, 1) << point.x, point.y, 1);
        cv::Mat transformed_point = H * homogenous_point;
        transformed_point /= transformed_point.at<float>(2);  // Convert back to Cartesian coordinates
        matched_points.push_back(cv::Point2f(transformed_point.at<float>(0), transformed_point.at<float>(1)));
    }

    float total_error = 0.0;
    cv::RNG random_number_gen;
    // Draw circles and lines on the grid
    for (size_t i = 0; i < N; ++i) {
        // Generate random color
        cv::Scalar color(random_number_gen.uniform(0, 255), random_number_gen.uniform(0, 255), random_number_gen.uniform(0, 255));
        // Draw circles
        circle(grid, first_image_points[i], 5, color, 1);  //  circle on first image
        circle(grid, matched_points[i] + cv::Point2f(image1.cols, 0), 5, color, 1);  //  circle on second image
        // Draw lines
        line(grid, first_image_points[i], matched_points[i] + cv::Point2f(image1.cols, 0), color, 1);  //  line between circles
        // Calculate error (Euclidean distance)
        float error = cv::norm(static_cast<cv::Point2f>(second_image_points[i]) - matched_points[i]);
        total_error += error;
    }

    // Compute average error
    float average_error = total_error / N;
    std::cout << "Calculated Average Error: " << average_error << std::endl;
    // cv::imwrite("task2.png", grid);
    while(1){
        // Display the grid
        cv::imshow("Calculated Matches (Task 2)", grid);

        int key = cv::waitKey(20);
        // Check for 'Enter' key press
        if (key == 13 || key == 10) { // 13 is Enter key on Windows, 10 on Unix
            break; // Exit the loop
        }

    }
    

}

void BasicOpencv::findHomographyMap(const std::vector<cv::Point2d>& points1, const std::vector<cv::Point2d>& points2){
      // Check if there are enough points
    if (points1.size() < 4 || points2.size() < 4) {
        std::cerr << "Need at least 4 pairs of corresponding points." << std::endl;
        return;
    }

    // Compute the Homography Matrix
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);
    std::cout << "H: " << H << "\n";
    // Transform points from the first image
    std::vector<cv::Point2d> transformedPoints1;
    cv::perspectiveTransform(points1, transformedPoints1, H);
    // Prepare data for drawMatches
    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < transformedPoints1.size(); ++i) {
        matches.push_back(cv::DMatch(i, i, 0));
    }

    // Draw matches
    cv::Mat imgMatches;
    cv::Mat first_image = readImage(selected_image);
    cv::Mat second_image = readImage(selected_image + 1);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;

    // Convert cv::Point2d to cv::Point2f, then to cv::KeyPoint
    for (const auto& pt : points1) {
        keypoints1.push_back(cv::KeyPoint(static_cast<float>(pt.x), static_cast<float>(pt.y), 1.0f));
    }
    for (const auto& pt : points2) {
        keypoints2.push_back(cv::KeyPoint(static_cast<float>(pt.x), static_cast<float>(pt.y), 1.0f));
    }

    // Now use keypoints1 and keypoints2 in drawMatches
    cv::drawMatches(first_image, keypoints1, second_image, keypoints2, matches, imgMatches);

    // Compute the average error
    double error = 0.0;
    for (size_t i = 0; i < transformedPoints1.size(); ++i) {
        error += cv::norm(transformedPoints1[i] - points2[i]);
    }
    error /= transformedPoints1.size();

    std::cout << "Average Error: " << error << std::endl;

    // Show the matches
    //cv::imwrite("task3.png", imgMatches);
    cv::imshow("Matches (Task 3)", imgMatches);
    cv::waitKey(0);                 

}                      