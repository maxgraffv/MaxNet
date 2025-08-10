#include "DataLoader.h"

#include <opencv2/opencv.hpp>
#include <iostream>

namespace fs = std::filesystem;

void DataLoader::loadFromFolder(const std::string& root,
                                 std::vector<Tensor>& images,
                                 std::vector<Tensor>& labels,
                                 const std::vector<std::string>& class_names,
                                 int target_size) {

    for (size_t class_idx = 0; class_idx < class_names.size(); ++class_idx) {
        std::string class_folder = root + "/" + class_names[class_idx];
        if (!fs::exists(class_folder)) {
            std::cerr << "Brak folderu: " << class_folder << std::endl;
            continue;
        }

        for (fs::directory_iterator it(class_folder), end; it != end; ++it) {
            if (fs::is_regular_file(*it)) {
                cv::Mat img = cv::imread(it->path().string(), cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "Nie udało się wczytać obrazu: " << it->path() << std::endl;
                    continue;
                }

                cv::resize(img, img, cv::Size(target_size, target_size));
                img.convertTo(img, CV_32FC3, 1.0 / 255.0);

                Tensor tensor({3, target_size, target_size});
                for (size_t c = 0; c < 3; ++c) {
                    for (size_t y = 0; y < static_cast<size_t>(target_size); ++y) {
                        for (size_t x = 0; x < static_cast<size_t>(target_size); ++x) {
                            tensor.at({c, y, x}) = img.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x))[c];
                        }
                    }
                }

                images.push_back(tensor);
                labels.push_back(Tensor({1}, {static_cast<float>(class_idx)}));
            }
        }
    }
}
