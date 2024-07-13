/*
Date:26/04/2024
Developed by: Musa Almaz
Project: Project 4 - representation, learning and recognition
Summary: This is the source file for the tasks of the Project 4.
Basically includes function definitions.
*/
#include "project4.hpp"


void Project4::printPaths(const std::vector<std::pair<std::string, std::string>>& paths) {
    for (const auto& path : paths) {
        std::cout << path.first << " " << path.second << std::endl;
    }
}

cv::Mat Project4::bitwiseAnd(std::string orgImagePath, std::string maskPath){
    // Load the image and the mask
    cv::Mat image = cv::imread(orgImagePath, cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);

    if (image.empty() || mask.empty()) {
        std::cerr << "Error loading images!" << std::endl;
    }

    // Ensure the mask is binary
    cv::Mat binaryMask;
    cv::threshold(mask, binaryMask, 128, 255, cv::THRESH_BINARY);

    // Apply the bitwise AND
    cv::Mat result;
    cv::bitwise_and(image, image, result, mask);
    return result;
}

std::pair<std::vector<std::pair<std::string, std::string>>, std::vector<std::pair<std::string, std::string>>> Project4::collectImagePaths(Data_Type data_type) {
    std::string selected_dir;
    if(data_type == rgb_image){
        selected_dir = datasetDir;
    }
    else if(data_type == depth_image){
        // selected_dir = datasetDirDepth;
        selected_dir = datasetDir;
    }
    fs::path directory(selected_dir);

    int objectCount = 0;

    // Step 1: Traverse the dataset directory and collect objects
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_directory()) {
            std::string objectName = entry.path().filename().string();
            std::vector<fs::path> subfolders;
            std::cout << "object name " << objectName << "\n";

            // Step 2: Traverse subfolders within each object folder
            for (const auto& subfolderEntry : fs::directory_iterator(entry.path())) {
                if (subfolderEntry.is_directory()) {
                    subfolders.push_back(subfolderEntry.path());
                    std::cout << "sub : " << subfolderEntry.path() << "\n";
                }
            } 

            if (!subfolders.empty()) {
                objectFolders[objectName] = subfolders;
                objectCount++;
            }

            if (objectCount == num_objects) {
                break;  // Limit the number of processed objects
            }
        }
    }

    // Step 3: Distribute subfolders into training and testing
    for (auto& [objectName, folders] : objectFolders) {
        if (folders.size() > 1) {
            std::sort(folders.begin(), folders.end());
            fs::path testFolder = folders.back();
            // std::cout << "Test folder path: " << testFolder.string() << std::endl;
            folders.pop_back();  // Remove the last folder for testing

            // Collect image paths for training
            for (const auto& folder : folders) {
                // std::cout << "rest : " << folder.string() << "\n";
                for (const auto& file : fs::directory_iterator(folder)) {
                    
                    if(data_type == rgb_image){
                        std::string fileName = file.path().filename().string();
                        std::string maskName = fileName;
                        size_t pos = maskName.find("_crop");
                        if (pos != std::string::npos) {
                            maskName.replace(pos, 5, "_maskcrop");

                            // Check if the mask file exists in the same directory
                            fs::path maskPath = file.path().parent_path() / maskName;
                            if (fs::exists(maskPath)) {
                                // std::cout << "Mask path : " << maskPath.string() << "\n";
                                // std::cout << "File path : " << file.path().string() << "\n";
                                cv::Mat andedResult = bitwiseAnd(file.path().string(), maskPath.string());
                                trainImages[objectName].push_back(andedResult);
                            }

                        }
                        if (file.path().extension() == ".png" && file.path().filename().string().find("_crop") != std::string::npos) {
                            trainingPaths.push_back({file.path().string(), objectName});
                        }
                    }
                    else if(data_type == depth_image){
                        trainingPathsDepth.push_back({file.path().string(), objectName});
                        // Convert depth data to cv::Mat object
                        // cv::Mat depth_img = convertPCDtoMat(file.path().string());
                        // trainImagesDepth[objectName].push_back(depth_img);
                        if (file.path().extension() == ".png" && file.path().filename().string().find("_depthcrop") != std::string::npos){
                            cv::Mat depthImg = cv::imread(file.path().string());
                            trainImagesDepth[objectName].push_back(depthImg);
                        }
                    }
                }
            }

            // Collect image paths for testing
            for (const auto& file : fs::directory_iterator(testFolder)) {
                
                if(data_type == rgb_image){
                    std::string fileName = file.path().filename().string();
                        std::string maskName = fileName;
                        size_t pos = maskName.find("_crop");
                        if (pos != std::string::npos) {
                            maskName.replace(pos, 5, "_maskcrop");

                            // Check if the mask file exists in the same directory
                            fs::path maskPath = file.path().parent_path() / maskName;
                            if (fs::exists(maskPath)) {
                                // std::cout << "Mask path : " << maskPath.string() << "\n";
                                // std::cout << "File path : " << file.path().string() << "\n";
                                cv::Mat andedResult = bitwiseAnd(file.path().string(), maskPath.string());
                                testImages[objectName].push_back(andedResult);
                            }

                        }
                    if (file.path().extension() == ".png" && file.path().filename().string().find("_crop") != std::string::npos) {
                        testPaths.push_back({file.path().string(), objectName});
                    }
                }
                else if(data_type == depth_image){
                    testPathsDepth.push_back({file.path().string(), objectName});
                    // Convert depth data to cv::Mat object
                    // cv::Mat depth_img = convertPCDtoMat(file.path().string());
                    // testImagesDepth[objectName].push_back(depth_img);
                    if (file.path().extension() == ".png" && file.path().filename().string().find("_depthcrop") != std::string::npos){
                        cv::Mat depthImg = cv::imread(file.path().string());
                        testImagesDepth[objectName].push_back(depthImg);
                    }
                }
            }

        } else {
            std::cerr << "Not enough subfolders for object " << objectName << " to separate training and testing.\n";
        }
    }

    return {trainingPaths, testPaths};
}

std::map<std::string, std::vector<std::string>> Project4::organizePathsByObject(std::vector<std::pair<std::string, std::string>> train_or_test_set)
{
    std::map<std::string, std::vector<std::string>> objectPaths;

    for (const auto& entry : train_or_test_set) {
        const std::string& path = entry.first;
        const std::string& objectName = entry.second;
        objectPaths[objectName].push_back(path);
    }

    return objectPaths;
}


std::vector<cv::KeyPoint> Project4::extractSIFTFeatures(const cv::Mat& image, cv::Mat& descriptors) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    return keypoints;
}

std::vector<cv::Mat> Project4::createBOWRepresentation(const std::vector<std::string>& imagePaths) {
    // Parameters
    int dictionarySize = 10;

    // BOW trainer setup
    cv::TermCriteria tc(cv::TermCriteria::MAX_ITER, 100, 0.001);
    cv::BOWKMeansTrainer bowTrainer(dictionarySize, tc, 1, cv::KMEANS_PP_CENTERS);

    // Feature extractor and descriptor setup using SIFT
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    cv::BOWImgDescriptorExtractor bowExtractor(sift, matcher);

    // Extract SIFT descriptors from each image and add to BOW trainer
    for (const std::string& path : imagePaths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            cv::Mat descriptors;
            std::vector<cv::KeyPoint> keypoints = extractSIFTFeatures(image, descriptors);
            if (!descriptors.empty()) {
                bowTrainer.add(descriptors);
            }
        }
    }

    // Cluster the SIFT descriptors to create the vocabulary
    cv::Mat vocabulary = bowTrainer.cluster();
    bowExtractor.setVocabulary(vocabulary);
    std::vector<cv::Mat> bowDescriptor;
    // Extract BOW descriptors for each image using the vocabulary
    for (const std::string& path : imagePaths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat bowDescriptor_;
            sift->detect(image, keypoints);
            bowExtractor.compute(image, keypoints, bowDescriptor_);
            bowDescriptor.push_back(bowDescriptor_);

            // `bowDescriptor` now contains the BOW representation of the image
            // This can be used for training a classifier or other purposes
        }
    }
    return bowDescriptor;
}

std::vector<cv::Mat> Project4::createBOWRepresentation(const std::vector<cv::Mat>& images) {
    // Parameters
    int dictionarySize = 50;

    // BOW trainer setup
    cv::TermCriteria tc(cv::TermCriteria::MAX_ITER, 100, 0.001);
    cv::BOWKMeansTrainer bowTrainer(dictionarySize, tc, 1, cv::KMEANS_PP_CENTERS);

    // Feature extractor and descriptor setup using SIFT
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    cv::BOWImgDescriptorExtractor bowExtractor(sift, matcher);

    // Extract SIFT descriptors from each image and add to BOW trainer
    for (const cv::Mat& image : images) {
        // cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        cv::Mat img = correctImageType(image);
        if (!img.empty()) {
            cv::Mat descriptors;
            std::vector<cv::KeyPoint> keypoints = extractSIFTFeatures(img, descriptors);
            if (!descriptors.empty()) {
                bowTrainer.add(descriptors);
            }
        }
    }
    std::cout << "VOCABULARY\n";
    // Cluster the SIFT descriptors to create the vocabulary
    cv::Mat vocabulary = bowTrainer.cluster();
    bowExtractor.setVocabulary(vocabulary);
    std::vector<cv::Mat> bowDescriptor;
    // Extract BOW descriptors for each image using the vocabulary
    for (const cv::Mat& image : images) {
        // cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        cv::Mat img = correctImageType(image);
        if (!img.empty()) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat bowDescriptor_;
            sift->detect(img, keypoints);
            bowExtractor.compute(img, keypoints, bowDescriptor_);
            cv::Mat normalized_bowDescriptor_;
            cv::normalize(bowDescriptor_, normalized_bowDescriptor_, 0, 1, cv::NORM_MINMAX);
            bowDescriptor.push_back(normalized_bowDescriptor_);

            // `bowDescriptor` now contains the BOW representation of the image
            // This can be used for training a classifier or other purposes
        }
    }
    return bowDescriptor;
}

cv::Ptr<cv::ml::SVM> Project4::trainOneClassSVM(const std::vector<cv::Mat>& descriptors, const std::string& objectName) {
    cv::Mat labels = cv::Mat::ones(descriptors.size(), 1, CV_32S); // Labels for one-class SVM

    for (const auto& desc : descriptors) {
        if (desc.empty()) {
            std::cerr << "Error: One of the descriptors is empty." << std::endl;
            return nullptr; // Or handle the error as appropriate
        }
        // std::cout << "Descriptor size: " << desc.size() << ", type: " << desc.type() << std::endl;
    }

    // Check for consistent dimensions and types
    if (!descriptors.empty()) {
        int cols = descriptors.front().cols;
        int type = descriptors.front().type();
        for (const auto& desc : descriptors) {
            if (desc.cols != cols || desc.type() != type) {
                std::cerr << "Error: Descriptors have inconsistent sizes or types." << std::endl;
                return nullptr; // Or handle the error as appropriate
            }
        }
    }

    // Convert vector of Mats to a single Mat
    cv::Mat trainingData;
    std::vector<cv::Mat> reshapedDescriptors;
    for (const cv::Mat& desc : descriptors) {
        reshapedDescriptors.push_back(desc.reshape(1, 1)); // Reshape to ensure it is a row vector
    }
    cv::vconcat(reshapedDescriptors, trainingData);
    // cv::vconcat(descriptors, trainingData);

    // Set up SVM parameters
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::ONE_CLASS);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setNu(0.1);  // Nu parameter for one-class SVM, typically between 0.1 and 0.5
    svm->setGamma(0.002);
    // Train the SVM
    svm->train(trainingData, cv::ml::ROW_SAMPLE, labels);

    // Save the trained model
    // svm->save("svm_" + objectName + ".xml");
    return svm;
}

std::vector<int> Project4::classifyImages(const cv::Ptr<cv::ml::SVM>& svm, const std::vector<cv::Mat>& descriptors, float threshold) {
    std::vector<int> predictions;
    if (!svm.empty()) {
        for (const auto& desc : descriptors) {
            if (!desc.empty()) {
                float response = svm->predict(desc, cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
                // std::cout << "Response : " << response << "\n";
                predictions.push_back(response > threshold ? 1 : -1);
            } else {
                std::cerr << "Descriptor is empty.\n";
            }
        }
    } else {
        std::cerr << "SVM model is not loaded.\n";
    }
    return predictions;
}

void Project4::computeMetrics(const std::vector<cv::Mat>& descriptors,const std::map<std::string, cv::Ptr<cv::ml::SVM>>& svmTrains,const std::vector<std::string>& object_names, const std::vector<std::string>& allTestImagesName, Data_Type data_type) {
    // Classify images at each threshold and compute metrics
    for(int j = 0; j < object_names.size(); j++){
        std::vector<double> precisions;
        std::vector<double> recalls;
        for (float threshold : thresholds) {
            
            std::cout << "Testing : " << object_names[j] << " for the threshold : " << threshold << "\n";
            
            std::vector<int> predictions = classifyImages(svmTrains.at(object_names[j]), descriptors, threshold);
            double precision, recall, accuracy;
            // computeMetrics(trueLabels[object.first], predictions, precision, recall, accuracy);
            int tp = 0, fp = 0, fn = 0, tn = 0;
            std::cout << "Prediction size : " << predictions.size()  << "\n";
            for (size_t i = 0; i < predictions.size(); ++i) {
                if (predictions[i] == 1 && allTestImagesName[i] == object_names[j]) tp++;
                if (predictions[i] == 1 && allTestImagesName[i] != object_names[j]) fp++;
                if (predictions[i] == -1 && allTestImagesName[i] == object_names[j]) fn++;
                if (predictions[i] == -1 && allTestImagesName[i] != object_names[j]) tn++;
            }
            std::cout << "TP : " << tp << " FP : " << fp << " FN : " << fn << " TN : " << tn << "\n";
            precision = (tp + fp) > 0 ? tp / static_cast<double>(tp + fp) : 0;
            recall = (tp + fn) > 0 ? tp / static_cast<double>(tp + fn) : 0;
            accuracy = (tp + tn) > 0 ? (tp + tn) / static_cast<double>(tp + tn + fp + fn) : 0;

            // Output results
            std::cout << "Precision: " << precision << ", Recall: " << recall << " , Accuracy : " << accuracy << std::endl;
            precisions.push_back(precision);
            recalls.push_back(recall);
        }
        plotPRCurve(precisions, recalls, thresholds, object_names[j], data_type);
    }
}

cv::Mat Project4::convertPCDtoMat(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return cv::Mat();
    }

    cv::Mat image(cloud->height, cloud->width, CV_32FC1);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        auto& p = cloud->points[i];
        image.at<float>(i / cloud->width, i % cloud->width) = p.z; // Assuming Z is depth
    }
    // cv::Mat normalizedDepth;
    // cv::normalize(image, normalizedDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1); // Normalize for display
    // cv::imshow("Depth Image", normalizedDepth);
    // cv::waitKey(0);

    return image;
}

cv::Mat Project4::correctImageType(cv::Mat image){
    // Check if image needs conversion to 8-bit
    if (image.type() != CV_8U) {
        double minVal, maxVal;
        cv::minMaxIdx(image, &minVal, &maxVal); // Find min and max values
        cv::Mat scaledImage;
        image.convertTo(scaledImage, CV_8UC1, 255 / (maxVal - minVal), -minVal * 255 / (maxVal - minVal));
        return scaledImage;
    }
    return image;
}


void Project4::run(){ 
    std::cout << " \t ...RGB data part started ...... \n";

    std::pair<std::vector<std::pair<std::string, std::string>>, std::vector<std::pair<std::string, std::string>>> paths = collectImagePaths(rgb_image);

    std::map<std::string, cv::Ptr<cv::ml::SVM>> svmTrains;

    std::vector<std::string> object_names;

    for (const auto& object : trainImages) {
        std::cout << "Training object : " << object.first << std::endl;
        object_names.push_back(object.first);
        // for (const auto& path : object.second) {
        //     std::cout << "  Path: " << path << std::endl;
        // }
        std::vector<cv::Mat> descriptors = createBOWRepresentation(object.second);

        cv::Ptr<cv::ml::SVM> svm = trainOneClassSVM(descriptors, object.first);
        svmTrains[object.first] = svm;

    }

    
    flattenedTestImages = flattenTestImages(testImages);
    for (const auto& object : flattenedTestImages) {
        allTestImages.push_back(object.second);
        allTestImagesNames.push_back(object.first);
    }
    

    // Compute BOW descriptors for each image
    std::vector<cv::Mat> descriptors = createBOWRepresentation(allTestImages);
    computeMetrics(descriptors, svmTrains, object_names, allTestImagesNames, rgb_image);


    std::cout << "\t ... Depth data part started ... \n";

    paths = collectImagePaths(depth_image);

    std::map<std::string, cv::Ptr<cv::ml::SVM>> svmTrainsDepth;

    std::vector<std::string> object_names_depth;

    for (const auto& object : trainImagesDepth) {
        std::cout << "Training object : " << object.first << std::endl;
        object_names_depth.push_back(object.first);

        std::vector<cv::Mat> descriptors = createBOWRepresentation(object.second);

        cv::Ptr<cv::ml::SVM> svm = trainOneClassSVM(descriptors, object.first);
        svmTrainsDepth[object.first] = svm;

    }

    flattenedTestImagesDepth = flattenTestImages(testImagesDepth);
    for (const auto& object : flattenedTestImagesDepth) {
        allTestImagesDepth.push_back(object.second);
        allTestImagesNamesDepth.push_back(object.first);
    }

    // Compute BOW descriptors for each image
    std::vector<cv::Mat> descriptorsDepth = createBOWRepresentation(allTestImagesDepth);
    computeMetrics(descriptorsDepth, svmTrainsDepth, object_names_depth, allTestImagesNamesDepth, depth_image);

}



void Project4::plotPRCurve(const std::vector<double>& precisions, const std::vector<double>& recalls,
                 const std::vector<double>& thresholds, const std::string& className, Data_Type data_type) {
    plt::figure();
    plt::plot(recalls, precisions, "b.-");  // Blue line with dots
    plt::title("PR Curve for " + className);
    plt::xlabel("Recall");
    plt::ylabel("Precision");

    // Annotating the curve with threshold values
    for (size_t i = 0; i < thresholds.size(); ++i) {
        plt::annotate("τ=" + std::to_string(thresholds[i]), recalls[i], precisions[i]);
    }
    std::string img_name;
    if (data_type == rgb_image){
        img_name = "RGB_" + className + ".png";
    }else if (data_type == depth_image){
        img_name = "Depth_" + className + ".png";
    }

    plt::save(img_name);
    plt::show();
}

std::vector<std::pair<std::string, cv::Mat>> Project4::flattenTestImages(const std::map<std::string, std::vector<cv::Mat>>& mappedImages){
    // Iterate over each entry in the map
    std::vector<std::pair<std::string, cv::Mat>> flattenedImages;
    for (const auto& entry : mappedImages) {
        const std::string& objectName = entry.first;
        const std::vector<cv::Mat>& images = entry.second;

        // For each image associated with the current object name, add a pair to the vector
        for (const cv::Mat& img : images) {
            flattenedImages.push_back(std::make_pair(objectName, img));
        }
    }
    return flattenedImages;
}
