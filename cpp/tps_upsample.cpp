#include "tps_upsample.h"

using namespace cv;
using namespace std;

void transformer_(vector<float>& output, Mat source, Mat target, const int out_height, const int out_width, bool transpose)
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
			W.at<float>(i, 3 + j) = d2_ij * logf(d2_ij + 1e-9);
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
			grid.at<float>(3 + i, j) = d2_ij * logf(d2_ij + 1e-9);
		}
	}

    Mat T_g = T * grid;

    const int area = out_height * out_width;
	output.clear();
	output.resize(1 * 2 * area);  ////不考虑batchsize，始终等于1
	////以vector方式跟以指向数组的指针形式访问,耗时没差别
	if(transpose)
	{
		for(int col_ind=0;col_ind<area;col_ind++)
		{
			output[col_ind*2] = (T_g.at<float>(0, col_ind) - grid.at<float>(1, col_ind)) * (float)out_width * 0.5;    ///按照flow形状是{1, out_height*out_width, 2}来赋值的
			output[col_ind*2+1] = (T_g.at<float>(1, col_ind) - grid.at<float>(2, col_ind)) * (float)out_height * 0.5;  ///按照flow形状是{1, out_height*out_width, 2}来赋值的
		}
	}
	else
	{
		for(int col_ind=0;col_ind<area;col_ind++)
		{
			output[col_ind] = (T_g.at<float>(0, col_ind) - grid.at<float>(1, col_ind)) * out_width * 0.5;    ///按照flow形状是{1, 2, out_height, out_width}来赋值的
			output[area+col_ind] = (T_g.at<float>(1, col_ind) - grid.at<float>(2, col_ind)) * out_height * 0.5;  ///按照flow形状是{1, 2, out_height, out_width}来赋值的
		}
	}
}