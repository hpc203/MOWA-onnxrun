#include<cmath>
#include "upsample.h"

using namespace std;


void interpolate(float* inp, vector<int> inp_shape, float* out, int outHeight, int outWidth, string interpolation, bool alignCorners, bool halfPixelCenters)
{
    const int N = inp_shape[0];
	const int C = inp_shape[1];
	const int inpHeight = inp_shape[2];
	const int inpWidth = inp_shape[3];

    float scaleWidth, scaleHeight;
    if (alignCorners && outHeight > 1)
            scaleHeight = static_cast<float>(inpHeight - 1) / (outHeight - 1);
        else
            scaleHeight = static_cast<float>(inpHeight) / outHeight;

        if (alignCorners && outWidth > 1)
            scaleWidth = static_cast<float>(inpWidth - 1) / (outWidth - 1);
        else
            scaleWidth = static_cast<float>(inpWidth) / outWidth;
    
    if (interpolation == "nearest")
    {
        const int inpSpatialSize = inpHeight * inpWidth;
        const int outSpatialSize = outHeight * outWidth;
        const int numPlanes = N * C;

        float heightOffset = 0.0f;
        float widthOffset = 0.0f;

        if (halfPixelCenters)
        {
            heightOffset = 0.5f * scaleHeight;
            widthOffset = 0.5f * scaleWidth;
        }

        for (int y = 0; y < outHeight; ++y)
        {
            float input_y = y * scaleHeight + heightOffset;
            int y0 = halfPixelCenters ? std::floor(input_y) : lroundf(input_y);
            y0 = std::min(y0, inpHeight - 1);

            const float* inpData_row = inp+y0*inpWidth;

            for (int x = 0; x < outWidth; ++x)
            {
                float input_x = x * scaleWidth + widthOffset;
                int x0 = halfPixelCenters ? std::floor(input_x) : lroundf(input_x);
                x0 = std::min(x0, inpWidth - 1);

                float* outData = out+y*outWidth+x;
                const float* inpData_row_c = inpData_row;

                for (int c = 0; c < numPlanes; ++c)
                {
                    *outData = inpData_row_c[x0];

                    inpData_row_c += inpSpatialSize;
                    outData += outSpatialSize;
                }
            }
        }
    }
    else if (interpolation == "bilinear")
    {
        const int inpSpatialSize = inpHeight * inpWidth;
        const int outSpatialSize = outHeight * outWidth;
        const int numPlanes = N * C;
        
        for (int y = 0; y < outHeight; ++y)
        {
            float input_y = y * scaleHeight;
            int y0 = static_cast<int>(input_y);
            const float* inpData_row0 = inp+y0*inpWidth;
            const int row1 = std::min(y0 + 1, inpHeight - 1);
            const float* inpData_row1 = inp+row1*inpWidth;
            for (int x = 0; x < outWidth; ++x)
            {
                float input_x = x * scaleWidth;
                int x0 = static_cast<int>(input_x);
                int x1 = std::min(x0 + 1, inpWidth - 1);

                float* outData = out+y*outWidth+x;
                const float* inpData_row0_c = inpData_row0;
                const float* inpData_row1_c = inpData_row1;
                for (int c = 0; c < numPlanes; ++c)
                {
                    *outData = inpData_row0_c[x0] +
                        (input_y - y0) * (inpData_row1_c[x0] - inpData_row0_c[x0]) +
                        (input_x - x0) * (inpData_row0_c[x1] - inpData_row0_c[x0] +
                        (input_y - y0) * (inpData_row1_c[x1] - inpData_row0_c[x1] - inpData_row1_c[x0] + inpData_row0_c[x0]));

                    inpData_row0_c += inpSpatialSize;
                    inpData_row1_c += inpSpatialSize;
                    outData += outSpatialSize;
                }
            }
        }
    }
    else
    {
        cout<<"Unknown interpolation: "<<interpolation<<endl;
    }
}