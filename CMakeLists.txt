cmake_minimum_required(VERSION 3.16)
project(MaxNet LANGUAGES CXX C)

set( CMAKE_CXX_STANDARD 11)
set( CMAKE_C_STANDARD 99)


add_executable(MaxNet Main.cpp)

add_subdirectory(core/Tensor)
add_subdirectory(core/NeuralNet)
add_subdirectory(core/DataLoader)



target_link_libraries(MaxNet PUBLIC
                        Tensor
                        NeuralNet    
                        DataLoader
                        )