#include "tps_transform.h"

using namespace cv;
using namespace std;

void _interpolate(vector<float>& output, float* im, const int dim, Mat xy_flat, const int out_height, const int out_width);

void transformer(vector<float>& output, float* U, const int dim, Mat source, Mat target, const int out_height, const int out_width)
{
    const int num_point = source.size[1];
    Mat W(num_point + 3, num_point + 3, CV_32FC1);
    Mat tp(num_point + 3, 2, CV_32FC1);
    
	for (int i = 0; i < num_point; i++)
	{
        const float x = source.ptr<float>(0, i)[0];
        const float y = source.ptr<float>(0, i)[1];
        W.at<float>(i, 0) = 1;
        W.at<float>(i, 1) = x;
        W.at<float>(i, 2) = y;

        W.at<float>(num_point, 3 + i) = 1;
        W.at<float>(num_point + 1, 3 + i) = x;
        W.at<float>(num_point + 2, 3 + i) = y;

        tp.at<float>(i, 0) = target.ptr<float>(0, i)[0];
		tp.at<float>(i, 1) = target.ptr<float>(0, i)[1];
    }
    for (int i = 0; i < num_point; i++)
	{
		for (int j = 0; j < num_point; j++)
		{
			const float d2_ij = powf(W.at<float>(i, 0) - W.at<float>(j, 0), 2.0) + powf(W.at<float>(i, 1) - W.at<float>(j, 1), 2.0) + powf(W.at<float>(i, 2) - W.at<float>(j, 2), 2.0);
			W.at<float>(i, 3 + j) = d2_ij * logf(d2_ij + 1e-6);
		}
	}
    for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			W.at<float>(num_point + i, j) = 0;
		}

        tp.at<float>(num_point + i, 0) = 0;
		tp.at<float>(num_point + i, 1) = 0;
	}
	Mat W_inv = W.inv();

    Mat T = W_inv * tp;  ////舍弃batchsize
    T = T.t();   

    const float interval_x = 2.0 / (out_width - 1);
	const float interval_y = 2.0 / (out_height - 1);
	const int grid_width = out_height * out_width;
    Mat grid(num_point + 3, grid_width, CV_32FC1); 
    for (int i = 0; i < out_height; i++)
	{
		for (int j = 0; j < out_width; j++)
		{
			const float x = -1.0 + j * interval_x;
			const float y = -1.0 + i * interval_y;
			const int col_ind = i * out_width + j;
			grid.at<float>(0, col_ind) = 1;
			grid.at<float>(1, col_ind) = x;
			grid.at<float>(2, col_ind) = y;
		}
	}
    for (int i = 0; i < num_point; i++)
	{
		const float x = source.ptr<float>(0, i)[0];
        const float y = source.ptr<float>(0, i)[1];
        for (int j = 0; j < grid_width; j++)
		{
			const float d2_ij = powf(x - grid.at<float>(1, j), 2.0) + powf(y - grid.at<float>(2, j), 2.0);
			grid.at<float>(3 + i, j) = d2_ij * logf(d2_ij + 1e-6);
		}
	}

    Mat T_g = T * grid;
    
    _interpolate(output, U, dim, T_g, out_height, out_width);
}


void _interpolate(vector<float>& output, float* im, const int dim, Mat xy_flat, const int out_height, const int out_width)
{
	const int max_x = out_width - 1;
	const int max_y = out_height - 1;
	const float height_f = float(out_height);
	const float width_f = float(out_width);
	const int area = out_height * out_width;
    output.clear();
    output.resize(1*area*dim);
    int ind = 0;
	for (int col_ind = 0; col_ind < area; col_ind++)
	{
        float x = (xy_flat.at<float>(0, col_ind) + 1.0)*width_f*0.5;
        float y = (xy_flat.at<float>(1, col_ind) + 1.0)*height_f*0.5;

        int x0 = int(x);
        int x1 = x0 + 1;
        int y0 = int(y);
        int y1 = y0 + 1;
        x0 = std::min(std::max(x0, 0), max_x);
        x1 = std::min(std::max(x1, 0), max_x);
        y0 = std::min(std::max(y0, 0), max_y);
        y1 = std::min(std::max(y1, 0), max_y);

        int base_y0 = y0 * out_width;
        int base_y1 = y1 * out_width;
        int idx_a = base_y0 + x0;
        int idx_b = base_y1 + x0;
        int idx_c = base_y0 + x1;
        int idx_d = base_y1 + x1;

        float x0_f = float(x0);
        float x1_f = float(x1);
        float y0_f = float(y0);
        float y1_f = float(y1);
        float wa = (x1_f - x) * (y1_f - y);
        float wb = (x1_f - x) * (y - y0_f);
        float wc = (x - x0_f) * (y1_f - y);
        float wd = (x - x0_f) * (y - y0_f);

        for(int c=0; c<dim; c++)
        {
            float pix = wa * im[idx_a*dim+c] + wb * im[idx_b*dim+c] + wc * im[idx_c*dim+c] + wd * im[idx_d*dim+c];
            output[ind] = pix;
            ind++;
        }
	}
}