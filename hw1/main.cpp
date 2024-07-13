/*
Date:28/02/2024
Developed by: Musa Almaz
Project: Project 1 - Basic OpenCV
Summary: Main file to run the tasks
*/
#include "project1.hpp"

int main(int argc, char ** argv){

    BasicOpencv task1;

    std::cout << "Number of images in the dataset is : " << task1.numberofImages << "\n";
    int n;
    
    while(true){
        std::cout << "Enter the number of the image :\n";
        std::cin >> n;

        // Check the n for valid
        if(n >= task1.numberofImages){
            std::cout << "The input is not valid, exiting. \n";
            break;
        }
        cv::Mat image = task1.readImage(n);
        task1.displayImage(image);

        
    }


}