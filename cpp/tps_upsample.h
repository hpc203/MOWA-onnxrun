# ifndef TPSUPSAMPLE
# define TPSUPSAMPLE
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void transformer_(std::vector<float>& output, cv::Mat source, cv::Mat target, const int out_height, const int out_width, bool transpose=true);

#endif