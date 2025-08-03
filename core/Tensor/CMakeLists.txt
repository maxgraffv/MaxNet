cmake_minimum_required(VERSION 3.16)

add_library(Tensor STATIC Tensor.cpp Tensor.h)
target_include_directories(Tensor PUBLIC .)