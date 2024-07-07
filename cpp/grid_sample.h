# ifndef GRIDSAMPLE
# define GRIDSAMPLE
#include <opencv2/core.hpp>

void GridSamplerBilinear(float* output, float* input, std::vector<int> inp_shape, float* grid, std::vector<int> g_shape, int padding_mode, bool align_corners);
void GridSamplerBilinear_vector(std::vector<float>& output, std::vector<float> input, std::vector<int> inp_shape, std::vector<float> grid, std::vector<int> g_shape, int padding_mode, bool align_corners);

#endif