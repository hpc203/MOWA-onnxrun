#include "utils_transform.h"
#include "tps_transform.h"
#include "tps_upsample.h"
#include "grid_sample.h"

using namespace cv;
using namespace std;


Mat get_rigid_mesh(const int height, const int width, const int grid_w, const int grid_h)
{
    float interval_x = (float)width / grid_w;
    float interval_y = (float)height / grid_h;
    const int h = grid_h + 1;
    const int w = grid_w + 1;
    int shape[4] = {1, h, w, 2};
    Mat rigid_mesh = cv::Mat::zeros(4, shape, CV_32FC1); 
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            rigid_mesh.ptr<float>(0, i, j)[0] = (float)j * interval_x;
            rigid_mesh.ptr<float>(0, i, j)[1] = (float)i * interval_y;
        }
    }
    return rigid_mesh;
}

Mat get_norm_mesh(Mat mesh, const int height, const int width)
{
    const int h = mesh.size[1];
    const int w = mesh.size[2];
    int shape[3] = {1, h*w, 2};
    Mat norm_mesh = cv::Mat::zeros(3, shape, CV_32FC1); 
    int row_ind = 0;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            norm_mesh.ptr<float>(0, row_ind)[0] = mesh.ptr<float>(0, i, j)[0] * 2.0 / float(width) - 1.0;
            norm_mesh.ptr<float>(0, row_ind)[1] = mesh.ptr<float>(0, i, j)[1] * 2.0 / float(height) - 1.0;
            row_ind++;
        }
    }
    return norm_mesh;
}

void transform_tps_fea(vector<float>& output, float* offset, float down_size, float* input_tensor, const int grid_w, const int grid_h, const int dim, const int h, const int w)
{
    ////input_tensor的形状是(1, h*w, dim)
    Mat rigid_mesh = get_rigid_mesh(h, w, grid_w, grid_h);
    Mat ori_mesh = rigid_mesh.clone();
    int row_ind = 0;
    for (int i = 0; i < ori_mesh.size[1]; i++)
    {
        for (int j = 0; j < ori_mesh.size[2]; j++)
        {
            float clamped_x = rigid_mesh.ptr<float>(0, i, j)[0] + offset[row_ind*2]/down_size;
            clamped_x = std::max(0.0f, std::min(clamped_x, float(h - 1)));
            float clamped_y = rigid_mesh.ptr<float>(0, i, j)[1] + offset[row_ind*2+1]/down_size;
            clamped_y = std::max(0.0f, std::min(clamped_y, float(w - 1)));

            ori_mesh.ptr<float>(0, i, j)[0] = clamped_x;
            ori_mesh.ptr<float>(0, i, j)[1] = clamped_y;
            row_ind++;
        }
    }

    Mat norm_rigid_mesh = get_norm_mesh(rigid_mesh, h, w);
    Mat norm_ori_mesh = get_norm_mesh(ori_mesh, h, w);
    rigid_mesh.release();
    ori_mesh.release();

    auto start_time_process = std::chrono::high_resolution_clock::now();
    transformer(output, input_tensor, dim, norm_rigid_mesh, norm_ori_mesh, h, w);
    auto end_time_model_process = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_model_process = end_time_model_process - start_time_process;
    cout<<"transformer waste time "<<to_string(diff_model_process.count())<<" s"<<endl;
    
    norm_rigid_mesh.release();
    norm_ori_mesh.release();
}

void upsample_tps(vector<float>& output, float* offset, const int grid_w, const int grid_h, const int out_h, const int out_w)
{
    if(grid_w+1 == out_w)
    {
        const int length = (grid_h+1)*(grid_w+1)*2;
        output.clear();
        output.resize(length);
        memcpy(output.data(), offset, length * sizeof(float));
    }
    else
    {
        Mat rigid_mesh = get_rigid_mesh(out_h, out_w, grid_w, grid_h);
        Mat ori_mesh = rigid_mesh.clone();
        int row_ind = 0;
        for (int i = 0; i < ori_mesh.size[1]; i++)
        {
            for (int j = 0; j < ori_mesh.size[2]; j++)
            {
                ori_mesh.ptr<float>(0, i, j)[0] = rigid_mesh.ptr<float>(0, i, j)[0] + offset[row_ind*2];
                ori_mesh.ptr<float>(0, i, j)[1] = rigid_mesh.ptr<float>(0, i, j)[1] + offset[row_ind*2+1];
                row_ind++;
            }
        }

        Mat norm_rigid_mesh = get_norm_mesh(rigid_mesh, out_h, out_w);
        Mat norm_ori_mesh = get_norm_mesh(ori_mesh, out_h, out_w);
        rigid_mesh.release();
        ori_mesh.release();

        transformer_(output, norm_rigid_mesh, norm_ori_mesh, out_h, out_w, true);
        
        norm_rigid_mesh.release();
        norm_ori_mesh.release();
    }
}

//// fea_shape=(1, 3, h, w) fl_shape=(1, 2, h, w)
void resample_image_xy(float* output, float* feature, vector<int> fea_shape, float* flow, vector<int> fl_shape)
{
    const int h = fea_shape[2];
    const int w = fea_shape[3];
    float y0 = ((float)h - 1) / 2.0;
    float x0 = ((float)w - 1) / 2.0;
    vector<int> g_shape = {1, h, w, 2};
    float* grid = new float[1*h*w*2];   ////shape=(1, h, w, 2)

    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            grid[i*w*2+j*2] = (j + flow[i*w+j] - x0) / x0;           ///x坐标
            grid[i*w*2+j*2+1] = (i + flow[h*w + i*w+j] - y0) / y0;   ///y坐标
        }  
    }

    auto start_time_process = std::chrono::high_resolution_clock::now();
    GridSamplerBilinear(output, feature, fea_shape, grid, g_shape, 0, false);
    auto end_time_model_process = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff_model_process = end_time_model_process - start_time_process;
    cout<<"GridSamplerBilinear waste time "<<to_string(diff_model_process.count())<<" s , vector length = "<<1*3*h*w<<endl;

    delete [] grid;
    grid = nullptr;
}

void resample_image_xy_vector(vector<float>& output, vector<float> feature, vector<int> fea_shape, vector<float> flow, vector<int> fl_shape)
{
    const int h = fea_shape[2];
    const int w = fea_shape[3];
    float y0 = ((float)h - 1) / 2.0;
    float x0 = ((float)w - 1) / 2.0;
    vector<float> grid(flow.size());   ////shape=(1, h, w, 2)
    vector<int> g_shape = {1, h, w, 2};

    auto a = std::chrono::high_resolution_clock::now();
    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            grid[i*w*2+j*2] = (j + flow[i*w+j] - x0) / x0;           ///x坐标
            grid[i*w*2+j*2+1] = (i + flow[h*w + i*w+j] - y0) / y0;   ///y坐标
        }  
    }
    auto b = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> c = b - a;
    cout<<"resample_image_xy_vector 2for waste time "<<to_string(c.count())<<" s , vector length = "<<grid.size()<<endl;

    auto start_time_process = std::chrono::high_resolution_clock::now();
    GridSamplerBilinear_vector(output, feature, fea_shape, grid, g_shape, 0, false);
    auto end_time_model_process = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff_model_process = end_time_model_process - start_time_process;
    cout<<"GridSamplerBilinear_vector waste time "<<to_string(diff_model_process.count())<<" s , vector length = "<<1*3*h*w<<endl;
}

Mat convert4dtoimage(vector<float> blob, vector<int> shape)
{
	const int H = shape[2];
	const int W = shape[3];
	const int area = H * W;
	const float* pdata = blob.data();
	Mat output(H, W, CV_32FC3);
	for(int i=0;i<H;i++)
	{
		for(int j=0;j<W;j++)
		{
			const int idx = i*W+j;
			float pix_r = pdata[idx] * 255.0;
			float pix_g = pdata[area + idx] * 255.0;
			float pix_b = pdata[2*area + idx] * 255.0;
			output.at<Vec3f>(i, j) = Vec3f(pix_r, pix_g, pix_b);
		}
	}
	output.convertTo(output, CV_8UC3);
	return output;
}

Mat draw_mesh_on_warp(const Mat warp, const Mat mesh_np, const int grid_h, const int grid_w)
{
	const int mesh_h = mesh_np.size[1];
    const int mesh_w = mesh_np.size[2];
    Mat f_local_x(mesh_h, mesh_w, CV_32FC1);
    Mat f_local_y(mesh_h, mesh_w, CV_32FC1);
    for(int i = 0; i < mesh_h; i++)
    {
        for (int j = 0; j < mesh_w; j++)
        {
            f_local_x.at<float>(i, j) = mesh_np.ptr<float>(0, i, j)[0];
            f_local_y.at<float>(i, j) = mesh_np.ptr<float>(0, i, j)[1];
        }
    }

    const int height = warp.rows;
	const int width = warp.cols;

	double minValue_x, maxValue_x;    // 最大值，最小值
	cv::Point  minIdx_x, maxIdx_x;    // 最小值坐标，最大值坐标     
	cv::minMaxLoc(f_local_x, &minValue_x, &maxValue_x, &minIdx_x, &maxIdx_x);
	const int min_w = int(std::min(minValue_x, 0.0));
	const int max_w = int(std::max(maxValue_x, double(width)));

	double minValue_y, maxValue_y;    // 最大值，最小值
	cv::Point  minIdx_y, maxIdx_y;    // 最小值坐标，最大值坐标     
	cv::minMaxLoc(f_local_y, &minValue_y, &maxValue_y, &minIdx_y, &maxIdx_y);
	const int min_h = int(std::min(minValue_y, 0.0));
	const int max_h = int(std::max(maxValue_y, double(height)));
	
	const int cw = max_w - min_w;
	const int ch = max_h - min_h;
	const int pad_top = 0 - min_h + 5;
	const int pad_bottom = ch + 10 - (pad_top + height);
	const int pad_left = 0 - min_w + 5;
	const int pad_right = cw + 10 - (pad_left + width);
	Mat pic;
	copyMakeBorder(warp, pic, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, Scalar(255,255,255));
	pic.convertTo(pic, CV_8UC3);
	for (int i = 0; i < (grid_h + 1); i++)
	{
		for (int j = 0; j < (grid_w + 1); j++)
		{
			if (j == grid_w && i == grid_h) continue;
			else if (j == grid_w)
			{
				line(pic, Point(int(f_local_x.at<float>(i, j) - min_w + 5), int(f_local_y.at<float>(i, j) - min_h + 5)), Point(int(f_local_x.at<float>(i + 1, j) - min_w + 5), int(f_local_y.at<float>(i + 1, j) - min_h + 5)), Scalar(0, 255, 0), 2);
			}
			else if (i == grid_h)
			{
				line(pic, Point(int(f_local_x.at<float>(i, j) - min_w + 5), int(f_local_y.at<float>(i, j) - min_h + 5)), Point(int(f_local_x.at<float>(i, j + 1) - min_w + 5), int(f_local_y.at<float>(i, j + 1) - min_h + 5)), Scalar(0, 255, 0), 2);
			}
			else
			{
				line(pic, Point(int(f_local_x.at<float>(i, j) - min_w + 5), int(f_local_y.at<float>(i, j) - min_h + 5)), Point(int(f_local_x.at<float>(i + 1, j) - min_w + 5), int(f_local_y.at<float>(i + 1, j) - min_h + 5)), Scalar(0, 255, 0), 2);
				line(pic, Point(int(f_local_x.at<float>(i, j) - min_w + 5), int(f_local_y.at<float>(i, j) - min_h + 5)), Point(int(f_local_x.at<float>(i, j + 1) - min_w + 5), int(f_local_y.at<float>(i, j + 1) - min_h + 5)), Scalar(0, 255, 0), 2);
			}
		}
	}
    f_local_x.release();
    f_local_y.release();
	return pic;
}

void split2xy(vector<float> flow, vector<int> f_shape, Mat& flow_x, Mat& flow_y)
{
	const int h = f_shape[2];
	const int w = f_shape[3];
	flow_x.create(h, w, CV_32FC1);
	flow_y.create(h, w, CV_32FC1);
    const int area = h*w;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			flow_x.at<float>(i, j) = flow[i*w+j];
			flow_y.at<float>(i, j) = flow[area+i*w+j];
		}
	}
}

Mat flow_mesh(Mat predflow_x, Mat predflow_y, Mat srcimg, const int ori_height, const int ori_width)
{
	Mat mesh_x(ori_height, ori_width, CV_32FC1);
	Mat mesh_y(ori_height, ori_width, CV_32FC1);
	for (int i = 0; i < ori_height; i++)
	{
		for (int j = 0; j < ori_width; j++)
		{
			mesh_x.at<float>(i, j) = predflow_x.at<float>(i, j) + j;
			mesh_y.at<float>(i, j) = predflow_y.at<float>(i, j) + i;
		}
	}
	Mat pred_out;
	cv::remap(srcimg, pred_out, mesh_x, mesh_y, INTER_LINEAR);
	pred_out.convertTo(pred_out, CV_8UC3);
	return pred_out;
}
