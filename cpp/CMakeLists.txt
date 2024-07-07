project(test)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/utils_transform.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/tps_transform.cpp 
                                ${CMAKE_CURRENT_SOURCE_DIR}/tps_upsample.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/upsample.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/grid_sample.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/include/opencv4"
        PUBLIC "/home/wangbo/libs/mylibs/onnxruntime-linux-x64-1.12.0/include"
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
target_link_libraries(${PROJECT_NAME}
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgcodecs.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_highgui.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgproc.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_core.so.4.8.0"
        "/home/wangbo/libs/mylibs/onnxruntime-linux-x64-1.12.0/lib/libonnxruntime.so.1.12.0"
        )