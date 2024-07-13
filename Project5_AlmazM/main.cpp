/*
Date:18/05/2024
Developed by: Musa Almaz
Project: Project 5 - Optical Flow and Tracking
Summary: This is the main file for the project.
*/
#include "project5.hpp"

int main(int argc, char **argv){


    Project5 project;
    while (true){
        std::cout << "Enter valid input : \n";
        int n;
        std::cin >> n;
        bool result = project.checkIndexes(n);
        if (result == false){
            std::cout << "Input is invalid, exiting ... \n";
            break;
        }
        project.getImages(n);
        // Run the code
        project.run();
    }
    
    
    return 0;
}