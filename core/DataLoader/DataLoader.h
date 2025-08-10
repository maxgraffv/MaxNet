#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "Tensor.h"
#include <vector>
#include <string>
#include <filesystem>

class DataLoader {
public:
    static void loadFromFolder(const std::string& root,
                                std::vector<Tensor>& images,
                                std::vector<Tensor>& labels,
                                const std::vector<std::string>& class_names,
                                int target_size = 64);
};

#endif