cmake_minimum_required(VERSION 3.13)

add_library(NeuralNet STATIC NeuralNet.cpp NeuralNet.h)
target_include_directories(NeuralNet PUBLIC .)

target_link_libraries(NeuralNet PUBLIC Tensor)