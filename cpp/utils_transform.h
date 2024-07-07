# ifndef UTILSTRANSFORM
# define UTILSTRANSFORM
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


cv::Mat get_rigid_mesh(const int height, const int width, const int grid_w, const int grid_h);
cv::Mat get_norm_mesh(cv::Mat mesh, const int height, const int width);
void transform_tps_fea(std::vector<float>& output, float* offset, float down_size, float* input_tensor, const int grid_w, const int grid_h, const int dim, const int h, const int w);
void upsample_tps(std::vector<float>& output, float* offset, const int grid_w, const int grid_h, const int out_h, const int out_w);
void resample_image_xy(float* output, float* feature, std::vector<int> fea_shape, float* flow, std::vector<int> fl_shape);
void resample_image_xy_vector(std::vector<float>& output, std::vector<float> feature, std::vector<int> fea_shape, std::vector<float> flow, std::vector<int> fl_shape);
cv::Mat convert4dtoimage(std::vector<float> blob, std::vector<int> shape);
cv::Mat draw_mesh_on_warp(const cv::Mat warp, const cv::Mat mesh_np, const int grid_h, const int grid_w);
void split2xy(std::vector<float> flow, std::vector<int> f_shape, cv::Mat& flow_x, cv::Mat& flow_y);
cv::Mat flow_mesh(cv::Mat predflow_x, cv::Mat predflow_y, cv::Mat srcimg, const int ori_height, const int ori_width);

#endif