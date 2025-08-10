cmake_minimum_required(VERSION 3.16)

# Nazwa biblioteki (statyczna)
add_library(DataLoader STATIC
    DataLoader.cpp
)

# Nagłówki
target_include_directories(DataLoader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../Tensor
)

# OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(DataLoader PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DataLoader PUBLIC ${OpenCV_LIBS})

# Zależność od Tensor
target_link_libraries(DataLoader PUBLIC Tensor)
