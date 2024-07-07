#include<iostream>
# include "grid_sample.h"

using namespace cv;
using namespace std;

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )

float SAFE_GET(float* input, int x, int y, int n, int c, int H, int W, int chans)
{
	 if(x >= 0 && x < W && y >=0 && y < H)
	 {
		return input[n*H*W*chans + c*H*W + y*W + x];
	 }
	 else
	 {
		return 0;
	 }
}

float SAFE_GET_vector(vector<float> input, int x, int y, int n, int c, int H, int W, int chans)
{
	 if(x >= 0 && x < W && y >=0 && y < H)
	 {
		return input[n*H*W*chans + c*H*W + y*W + x];
	 }
	 else
	 {
		return 0;
	 }
}

#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))


///c++数组和 vector访问执行性能比较 https://blog.csdn.net/h799710/article/details/107544792
void GridSamplerBilinear(float* output, float* input, vector<int> inp_shape, float* grid, vector<int> g_shape, int padding_mode, bool align_corners)
{
	int N = inp_shape[0];
	int C = inp_shape[1];
	int IH = inp_shape[2];
	int IW = inp_shape[3];
	int H = g_shape[1];
	int W = g_shape[2];
	// vector<int> out_shape = {N, C, H, W};

	int n, i, j, ic, ind;
	for (n = 0; n < N; ++n) 
	{
		for (i = 0; i < H; ++i) 
		{
			for (j = 0; j < W; ++j) 
			{
				// get the corresponding input x, y co-ordinates from grid
				ind = i*W*2+j*2;
				float ix = grid[ind];  ////batchsize=1
				float iy = grid[ind+1];  ////batchsize=1

				// normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
				if (align_corners) 
				{
					ix = ((ix + 1) / 2) * (IW-1);
					iy = ((iy + 1) / 2) * (IH-1);
				} 
				else 
				{
					ix = ((ix + 1) * IW - 1) / 2;
					iy = ((iy + 1) * IH - 1) / 2;
				}

				// get NE, NW, SE, SW pixel values from (x, y)
				int ix_nw = floor(ix);
				int iy_nw = floor(iy);
				int ix_ne = ix_nw + 1;
				int iy_ne = iy_nw;
				int ix_sw = ix_nw;
				int iy_sw = iy_nw + 1;
				int ix_se = ix_nw + 1;
				int iy_se = iy_nw + 1;

				// get surfaces to each neighbor:
				float nw = (ix_se - ix)    * (iy_se - iy);
				float ne = (ix    - ix_sw) * (iy_sw - iy);
				float sw = (ix_ne - ix)    * (iy    - iy_ne);
				float se = (ix    - ix_nw) * (iy    - iy_nw);

				if (padding_mode==1)
				{
					// clip coordinates to image borders
					CLIP_COORDINATES(ix_nw, ix_nw, IW);
					CLIP_COORDINATES(iy_nw, iy_nw, IH);
					CLIP_COORDINATES(ix_ne, ix_ne, IW);
					CLIP_COORDINATES(iy_ne, iy_ne, IH);
					CLIP_COORDINATES(ix_sw, ix_sw, IW);
					CLIP_COORDINATES(iy_sw, iy_sw, IH);
					CLIP_COORDINATES(ix_se, ix_se, IW);
					CLIP_COORDINATES(iy_se, iy_se, IH);
				}

                // calculate bilinear weighted pixel value and set output pixel
                for (ic = 0; ic < C; ++ic) 
                {
					//   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
					// + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
					float nw_val = SAFE_GET(input, ix_nw, iy_nw, n, ic, IH, IW, C);
					float ne_val = SAFE_GET(input, ix_ne, iy_ne, n, ic, IH, IW, C);
					float sw_val = SAFE_GET(input, ix_sw, iy_sw, n, ic, IH, IW, C);
					float se_val = SAFE_GET(input, ix_se, iy_se, n, ic, IH, IW, C);
					float out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
					ind = n*C*H*W + ic*H*W + i*W + j;
					output[ind] = out_val;
                }
			}
		}
	}
}

void GridSamplerBilinear_vector(vector<float>& output, vector<float> input, vector<int> inp_shape, vector<float> grid, vector<int> g_shape, int padding_mode, bool align_corners)
{
	int N = inp_shape[0];
	int C = inp_shape[1];
	int IH = inp_shape[2];
	int IW = inp_shape[3];
	int H = g_shape[1];
	int W = g_shape[2];
	// vector<int> out_shape = {N, C, H, W};
	// output.resize(N*C*H*W);

	int n, i, j, ic, ind;
	for (n = 0; n < N; ++n) 
	{
		for (i = 0; i < H; ++i) 
		{
			for (j = 0; j < W; ++j) 
			{
				// get the corresponding input x, y co-ordinates from grid
				ind = i*W*2+j*2;
				float ix = grid[ind];  ////batchsize=1
				float iy = grid[ind+1];  ////batchsize=1

				// normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
				if (align_corners) 
				{
					ix = ((ix + 1) / 2) * (IW-1);
					iy = ((iy + 1) / 2) * (IH-1);
				} 
				else 
				{
					ix = ((ix + 1) * IW - 1) / 2;
					iy = ((iy + 1) * IH - 1) / 2;
				}

				// get NE, NW, SE, SW pixel values from (x, y)
				int ix_nw = floor(ix);
				int iy_nw = floor(iy);
				int ix_ne = ix_nw + 1;
				int iy_ne = iy_nw;
				int ix_sw = ix_nw;
				int iy_sw = iy_nw + 1;
				int ix_se = ix_nw + 1;
				int iy_se = iy_nw + 1;

				// get surfaces to each neighbor:
				float nw = (ix_se - ix)    * (iy_se - iy);
				float ne = (ix    - ix_sw) * (iy_sw - iy);
				float sw = (ix_ne - ix)    * (iy    - iy_ne);
				float se = (ix    - ix_nw) * (iy    - iy_nw);

				if (padding_mode==1)
				{
					// clip coordinates to image borders
					CLIP_COORDINATES(ix_nw, ix_nw, IW);
					CLIP_COORDINATES(iy_nw, iy_nw, IH);
					CLIP_COORDINATES(ix_ne, ix_ne, IW);
					CLIP_COORDINATES(iy_ne, iy_ne, IH);
					CLIP_COORDINATES(ix_sw, ix_sw, IW);
					CLIP_COORDINATES(iy_sw, iy_sw, IH);
					CLIP_COORDINATES(ix_se, ix_se, IW);
					CLIP_COORDINATES(iy_se, iy_se, IH);
				}

                // calculate bilinear weighted pixel value and set output pixel
                for (ic = 0; ic < C; ++ic) 
                {
					//   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
					// + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
					float nw_val = SAFE_GET_vector(input, ix_nw, iy_nw, n, ic, IH, IW, C);
					float ne_val = SAFE_GET_vector(input, ix_ne, iy_ne, n, ic, IH, IW, C);
					float sw_val = SAFE_GET_vector(input, ix_sw, iy_sw, n, ic, IH, IW, C);
					float se_val = SAFE_GET_vector(input, ix_se, iy_se, n, ic, IH, IW, C);
					float out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
					ind = n*C*H*W + ic*H*W + i*W + j;
					output[ind] = out_val;
                }
			}
		}
	}
}
