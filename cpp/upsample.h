# ifndef UPSAMPLE
# define UPSAMPLE
#include <iostream>
#include <vector>

void interpolate(float* inp, std::vector<int> inp_shape, float* out, int outHeight, int outWidth, std::string interpolation, bool alignCorners, bool halfPixelCenters=false);

#endif