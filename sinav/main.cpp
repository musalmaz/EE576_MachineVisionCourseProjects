#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char **argv){

    Mat org_image = cv::imread("../RectWGlitch.jpg");

    // threshold(org_image, org_image, 0, 255, THRESH_BINARY);

    Mat normalized_image;
    org_image.convertTo(normalized_image, CV_32F, 1.0 / 255.0);

    Mat gridImage = cv::Mat(2 * normalized_image.rows , normalized_image.cols, normalized_image.type(), cv::Scalar::all(0));

    // Create a structuring element (SE) 
    int morph_size = 15; 
    Mat element = getStructuringElement( 
        MORPH_ELLIPSE, Size(2 * morph_size + 1, 
                         2 * morph_size + 1), 
        Point(morph_size, morph_size)); 
    Mat erod, dill; 
  
    // For Dilation 
    dilate(normalized_image, dill, element, 
           Point(-1, -1), 1);

    erode(dill, erod, element, 
          Point(-1, -1), 1); 

    // cv::Mat firstCell = gridImage(cv::Rect(0, 0, normalized_image.rows, normalized_image.cols));
    // org_image.copyTo(firstCell);

    // cv::Mat secondCell = gridImage(cv::Rect(normalized_image.rows, 0, normalized_image.rows, normalized_image.cols));
    // dill.copyTo(secondCell);

    imshow("norm", normalized_image);
    imshow("dilation", dill);
    imshow("erosion", erod);
    waitKey(0);

    return 0;
}