/*
Date:10/03/2024
Developed by: Musa Almaz
Project: Project 2 - Finding Homography Matrix
Summary: Main file to run the tasks
*/
#include "project2.hpp"

int main(int argc, char ** argv){

    BasicOpencv task;

    std::cout << "Number of images in the dataset is : " << task.numberofImages << "\n";
    int n;
    
    // while(true){
    std::cout << "Enter the number of the image :\n";
    std::cin >> n;
    task.selected_image = n;

    cv::Mat image = task.readImage(n);
    task.displayImage();

        
    task.calculateHomographyMap(task.first_image_points, task.second_image_points);
    std::cout << std::endl;
    task.findHomographyMap(task.first_image_points, task.second_image_points);


}