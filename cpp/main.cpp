#define _CRT_SECURE_NO_WARNINGS
#include<map>
#include<string>
#include<iostream>
#include <string>
#include <math.h>
#include "utils_transform.h"
#include "tps_upsample.h"
#include "upsample.h"
// #include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>


using namespace cv;
using namespace std;
using namespace Ort;


class MOWA
{
public:
	MOWA();
	vector<Mat> detect(const Mat srcimg, const Mat mask_img, const bool resize_flow);
private:
	vector<float> input1_tensor;
	vector<float> input2_tensor;
	vector<float> mask_tensor;
	vector<float> heads_input_tensor;
	
	const int tps_points[4] = {10, 12, 14, 16};
	const int head_num = 4;
	const int down_size = 16;
	const int embed_dim = 32;
	const int mini_size = 16;
	
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "MOWA: Multiple-in-One Image Warping Model");
	SessionOptions sessionOptions = SessionOptions();

	Ort::Session *encoder_ort_session = nullptr;
	vector<char *> encoder_input_names;
	vector<char *> encoder_output_names;
	int encoder_input_height;
	int encoder_input_width;

	Ort::Session *heads0_ort_session = nullptr;
	vector<char *> heads_input_names;
	vector<char *> heads_output_names;
	int heads_input_height;
	int heads_input_width;
	int heads0_output_len;    
	Ort::Session *heads1_ort_session = nullptr;    ////4个heads模块,输入形状是一样的，输出相撞不一样
	Ort::Session *heads2_ort_session = nullptr;
	Ort::Session *heads3_ort_session = nullptr;

	Ort::Session *decoder_ort_session = nullptr;
	vector<char *> decoder_input_names;
	vector<char *> decoder_output_names;
	
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::RunOptions runOptions;

	std::vector<int64_t> encoder_input_tensor_shape;
	std::vector<int64_t> encoder_input_mask_shape;
	std::vector<int64_t> heads_input_tensor_shape;

	std::vector<vector<int64_t>> decoder_inputs_tensor_shape;
	std::vector<int64_t> decoder_inputs_tensor_len;
	std::vector<int> decoder_flow_out_shape;
	int decoder_flow_out_len;
};

MOWA::MOWA()
{
	/// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

	string model_path = "/home/wangbo/mowa/weights/encoder.onnx";
    /// std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
    /// ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    encoder_ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法

    size_t numInputNodes = encoder_ort_session->GetInputCount();
    size_t numOutputNodes = encoder_ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
        encoder_input_names.push_back(encoder_ort_session->GetInputName(i, allocator));      ///低版本onnxruntime的接口函数
        // AllocatedStringPtr input_name_Ptr = encoder_ort_session->GetInputNameAllocated(i, allocator);  /// 高版本onnxruntime的接口函数
        // encoder_input_names.push_back(input_name_Ptr.get()); /// 高版本onnxruntime的接口函数
    }
	
	Ort::TypeInfo input_type_info = encoder_ort_session->GetInputTypeInfo(0);
	auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	auto input_dims = input_tensor_info.GetShape();
	this->encoder_input_height = input_dims[2];
    this->encoder_input_width = input_dims[3];
	this->encoder_input_tensor_shape = { 1, 3, this->encoder_input_height, this->encoder_input_width };
	this->encoder_input_mask_shape = { 1, 1, this->encoder_input_height, this->encoder_input_width };

    for (int i = 0; i < numOutputNodes; i++)
    {
        encoder_output_names.push_back(encoder_ort_session->GetOutputName(i, allocator));  ///低版本onnxruntime的接口函数
        // AllocatedStringPtr output_name_Ptr= encoder_ort_session->GetInputNameAllocated(i, allocator);
        // encoder_output_names.push_back(output_name_Ptr.get()); /// 高版本onnxruntime的接口函数
    }
	
    model_path = "/home/wangbo/mowa/weights/tps_regression_heads0.onnx";
	heads0_ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法, windows写法参照上面的注释
    numInputNodes = heads0_ort_session->GetInputCount();
    numOutputNodes = heads0_ort_session->GetOutputCount();
    for (int i = 0; i < numInputNodes; i++)
    {
        heads_input_names.push_back(heads0_ort_session->GetInputName(i, allocator));      ///低版本onnxruntime的接口函数
        // AllocatedStringPtr input_name_Ptr = heads0_ort_session->GetInputNameAllocated(i, allocator);  /// 高版本onnxruntime的接口函数
        // heads_input_names.push_back(input_name_Ptr.get()); /// 高版本onnxruntime的接口函数
    }
	
	input_type_info = heads0_ort_session->GetInputTypeInfo(0);
	auto heads_input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	input_dims = heads_input_tensor_info.GetShape();
	this->heads_input_height = input_dims[1];
    this->heads_input_width = input_dims[2];
	this->heads_input_tensor_shape = { 1, this->heads_input_height, this->heads_input_width };
	for (int i = 0; i < numOutputNodes; i++)
    {
        heads_output_names.push_back(heads0_ort_session->GetOutputName(i, allocator));  ///低版本onnxruntime的接口函数
        // AllocatedStringPtr output_name_Ptr= heads0_ort_session->GetInputNameAllocated(i, allocator);
        // heads_output_names.push_back(output_name_Ptr.get()); /// 高版本onnxruntime的接口函数
		input_type_info = heads0_ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		input_dims = input_tensor_info.GetShape();
		heads0_output_len = 1 * input_dims[1] * input_dims[2];
    }
	Ort::TypeInfo output_type_info = heads0_ort_session->GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto out_dims = output_tensor_info.GetShape();
	this->heads0_output_len = 1 * out_dims[1] * out_dims[2];
	
	model_path = "/home/wangbo/mowa/weights/tps_regression_heads1.onnx";
	heads1_ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法, windows写法参照上面的注释
	model_path = "/home/wangbo/mowa/weights/tps_regression_heads2.onnx";
	heads2_ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法, windows写法参照上面的注释
	model_path = "/home/wangbo/mowa/weights/tps_regression_heads3.onnx";
	heads3_ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法, windows写法参照上面的注释

	model_path = "/home/wangbo/mowa/weights/point_classification_decoder.onnx";
	decoder_ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法, windows写法参照上面的注释
	numInputNodes = decoder_ort_session->GetInputCount();
    numOutputNodes = decoder_ort_session->GetOutputCount();
    for (int i = 0; i < numInputNodes; i++)
    {
        decoder_input_names.push_back(decoder_ort_session->GetInputName(i, allocator));      ///低版本onnxruntime的接口函数
        // AllocatedStringPtr input_name_Ptr = decoder_ort_session->GetInputNameAllocated(i, allocator);  /// 高版本onnxruntime的接口函数
        // decoder_input_names.push_back(input_name_Ptr.get()); /// 高版本onnxruntime的接口函数
		
		input_type_info = decoder_ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		input_dims = input_tensor_info.GetShape();
		this->decoder_inputs_tensor_shape.push_back({ 1, input_dims[1], input_dims[2] });
		this->decoder_inputs_tensor_len.push_back(1 * input_dims[1] * input_dims[2]);
    }
	for (int i = 0; i < numOutputNodes; i++)
    {
        decoder_output_names.push_back(decoder_ort_session->GetOutputName(i, allocator));  ///低版本onnxruntime的接口函数
        // AllocatedStringPtr output_name_Ptr= decoder_ort_session->GetInputNameAllocated(i, allocator);
        // decoder_output_names.push_back(output_name_Ptr.get()); /// 高版本onnxruntime的接口函数
    }
	output_type_info = decoder_ort_session->GetOutputTypeInfo(0);
	auto output_tensor_info_ = output_type_info.GetTensorTypeAndShapeInfo();
	out_dims = output_tensor_info_.GetShape();
	this->decoder_flow_out_shape = { 1, (int)out_dims[1], (int)out_dims[2], (int)out_dims[3] };
	this->decoder_flow_out_len = 1 * this->decoder_flow_out_shape[1] *this->decoder_flow_out_shape[2] * this->decoder_flow_out_shape[3];
}

vector<Mat> MOWA::detect(const Mat srcimg, const Mat mask_img, const bool resize_flow)
{
	Mat input1;
	srcimg.convertTo(input1, CV_32FC3, 1.0 / 255.0);
	vector<cv::Mat> bgrChannels(3);
	split(input1, bgrChannels);
	const int img_h = srcimg.rows;
	const int img_w = srcimg.cols;
	int image_area = img_h * img_w;
	this->input1_tensor.clear();
	this->input1_tensor.resize(1 * 3 * image_area);   ////不考虑batchsize, batchsize始终等于1
	size_t single_chn_size = image_area * sizeof(float);
	memcpy(this->input1_tensor.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input1_tensor.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input1_tensor.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
	input1.release();

	Mat input2;
	resize(srcimg, input2, Size(this->encoder_input_width, this->encoder_input_height), INTER_LINEAR);
	Mat img = input2.clone();
	input2.convertTo(input2, CV_32FC3, 1.0 / 255.0);
	bgrChannels.clear();
	bgrChannels.resize(3);
	split(input2, bgrChannels);
	image_area = input2.rows * input2.cols;
	this->input2_tensor.clear();
	this->input2_tensor.resize(1 * 3 * image_area);
	single_chn_size = image_area * sizeof(float);
	memcpy(this->input2_tensor.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input2_tensor.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input2_tensor.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
	input2.release();
	
	
	Mat mask;  ////mask_img是单通道灰度图
	resize(mask_img, mask, Size(this->encoder_input_width, this->encoder_input_height), INTER_LINEAR);
	mask.convertTo(mask, CV_32FC1, 1.0 / 255.0);    
	image_area = mask.rows * mask.cols;
	this->mask_tensor.clear();
	this->mask_tensor.resize(1 * 1 * image_area);
	single_chn_size = image_area * sizeof(float);
	memcpy(this->mask_tensor.data(), (float *)mask.data, single_chn_size);
	mask.release();

	std::vector<Ort::Value> encoder_inputTensors;
	encoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, input2_tensor.data(), input2_tensor.size(), this->encoder_input_tensor_shape.data(), this->encoder_input_tensor_shape.size())));
	encoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, mask_tensor.data(), mask_tensor.size(), this->encoder_input_mask_shape.data(), this->encoder_input_mask_shape.size())));

	vector<Value> encoder_ort_outputs = this->encoder_ort_session->Run(runOptions, this->encoder_input_names.data(), encoder_inputTensors.data(), encoder_inputTensors.size(), this->encoder_output_names.data(), this->encoder_output_names.size());
	float *conv0 = encoder_ort_outputs[0].GetTensorMutableData<float>();
	float *conv1 = encoder_ort_outputs[1].GetTensorMutableData<float>();
	float *conv2 = encoder_ort_outputs[2].GetTensorMutableData<float>();
	float *conv3 = encoder_ort_outputs[3].GetTensorMutableData<float>();
	float *conv4 = encoder_ort_outputs[4].GetTensorMutableData<float>();
	float *conv_fea = encoder_ort_outputs[4].GetTensorMutableData<float>();

	image_area = this->heads_input_height * this->heads_input_width;
	this->heads_input_tensor.clear();
	this->heads_input_tensor.resize(1*image_area);
	single_chn_size = image_area * sizeof(float);
	memcpy(this->heads_input_tensor.data(), conv_fea, single_chn_size);

	vector<vector<float>> offset;
	int i = 0;
	Value heads0_input_tensor = Value::CreateTensor<float>(memory_info_handler, this->heads_input_tensor.data(), this->heads_input_tensor.size(), this->heads_input_tensor_shape.data(), this->heads_input_tensor_shape.size());
	vector<Value> heads0_ort_outputs = this->heads0_ort_session->Run(runOptions, this->heads_input_names.data(), &heads0_input_tensor, this->heads_input_names.size(), this->heads_output_names.data(), this->heads_output_names.size());
	float* pre = heads0_ort_outputs[0].GetTensorMutableData<float>();
	vector<float> tps(heads0_output_len);
	memcpy(tps.data(), pre, heads0_output_len*sizeof(float));
	offset.push_back(tps);
	vector<float> warp;
	transform_tps_fea(warp, pre, (float)this->down_size, conv_fea, this->tps_points[i]-1, this->tps_points[i]-1, this->embed_dim*16, this->mini_size, this->mini_size);
	memcpy(this->heads_input_tensor.data(), warp.data(), single_chn_size);
	vector<float> tps_up;
	upsample_tps(tps_up, pre, this->tps_points[i]-1, this->tps_points[i]-1, this->tps_points[i+1], this->tps_points[i+1]);

	i = 1;
	conv_fea = this->heads_input_tensor.data();
	Value heads1_input_tensor = Value::CreateTensor<float>(memory_info_handler, this->heads_input_tensor.data(), this->heads_input_tensor.size(), this->heads_input_tensor_shape.data(), this->heads_input_tensor_shape.size());
	vector<Value> heads1_ort_outputs = this->heads1_ort_session->Run(runOptions, this->heads_input_names.data(), &heads1_input_tensor, this->heads_input_names.size(), this->heads_output_names.data(), this->heads_output_names.size());
	pre = heads1_ort_outputs[0].GetTensorMutableData<float>();
	transform_tps_fea(warp, pre, (float)this->down_size, conv_fea, this->tps_points[i]-1, this->tps_points[i]-1, this->embed_dim*16, this->mini_size, this->mini_size);
	memcpy(this->heads_input_tensor.data(), warp.data(), single_chn_size);
	tps.resize(tps_up.size());
	for(int n=0;n<tps_up.size();n++)
	{
		tps[n] = pre[n] + tps_up[n];
	}
	offset.push_back(tps);
	upsample_tps(tps_up, tps.data(), this->tps_points[i]-1, this->tps_points[i]-1, this->tps_points[i+1], this->tps_points[i+1]);

	i = 2;
	conv_fea = this->heads_input_tensor.data();
	Value heads2_input_tensor = Value::CreateTensor<float>(memory_info_handler, this->heads_input_tensor.data(), this->heads_input_tensor.size(), this->heads_input_tensor_shape.data(), this->heads_input_tensor_shape.size());
	vector<Value> heads2_ort_outputs = this->heads2_ort_session->Run(runOptions, this->heads_input_names.data(), &heads2_input_tensor, this->heads_input_names.size(), this->heads_output_names.data(), this->heads_output_names.size());
	pre = heads2_ort_outputs[0].GetTensorMutableData<float>();
	transform_tps_fea(warp, pre, (float)this->down_size, conv_fea, this->tps_points[i]-1, this->tps_points[i]-1, this->embed_dim*16, this->mini_size, this->mini_size);
	memcpy(this->heads_input_tensor.data(), warp.data(), single_chn_size);
	tps.resize(tps_up.size());
	for(int n=0;n<tps_up.size();n++)
	{
		tps[n] = pre[n] + tps_up[n];
	}
	offset.push_back(tps);
	upsample_tps(tps_up, tps.data(), this->tps_points[i]-1, this->tps_points[i]-1, this->tps_points[i+1], this->tps_points[i+1]);
	
	i = 3;
	conv_fea = this->heads_input_tensor.data();
	Value heads3_input_tensor = Value::CreateTensor<float>(memory_info_handler, this->heads_input_tensor.data(), this->heads_input_tensor.size(), this->heads_input_tensor_shape.data(), this->heads_input_tensor_shape.size());
	vector<Value> heads3_ort_outputs = this->heads3_ort_session->Run(runOptions, this->heads_input_names.data(), &heads3_input_tensor, this->heads_input_names.size(), this->heads_output_names.data(), this->heads_output_names.size());
	pre = heads3_ort_outputs[0].GetTensorMutableData<float>();
	transform_tps_fea(warp, pre, (float)this->down_size, conv_fea, this->tps_points[i]-1, this->tps_points[i]-1, this->embed_dim*16, this->mini_size, this->mini_size);
	memcpy(this->heads_input_tensor.data(), warp.data(), single_chn_size);
	tps.resize(tps_up.size());
	for(int n=0;n<tps_up.size();n++)
	{
		tps[n] = pre[n] + tps_up[n];
	}
	offset.push_back(tps);

	std::vector<Ort::Value> decoder_inputTensors;
	i = 0;
	decoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, tps.data(), this->decoder_inputs_tensor_len[i], this->decoder_inputs_tensor_shape[i].data(), this->decoder_inputs_tensor_shape[i].size())));
	i = 1;
	decoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, conv4, this->decoder_inputs_tensor_len[i], this->decoder_inputs_tensor_shape[i].data(), this->decoder_inputs_tensor_shape[i].size())));
	i = 2;
	decoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, warp.data(), this->decoder_inputs_tensor_len[i], this->decoder_inputs_tensor_shape[i].data(), this->decoder_inputs_tensor_shape[i].size())));
	i = 3;
	decoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, conv3, this->decoder_inputs_tensor_len[i], this->decoder_inputs_tensor_shape[i].data(), this->decoder_inputs_tensor_shape[i].size())));
	i = 4;
	decoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, conv2, this->decoder_inputs_tensor_len[i], this->decoder_inputs_tensor_shape[i].data(), this->decoder_inputs_tensor_shape[i].size())));
	i = 5;
	decoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, conv1, this->decoder_inputs_tensor_len[i], this->decoder_inputs_tensor_shape[i].data(), this->decoder_inputs_tensor_shape[i].size())));
	i = 6;
	decoder_inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, conv0, this->decoder_inputs_tensor_len[i], this->decoder_inputs_tensor_shape[i].data(), this->decoder_inputs_tensor_shape[i].size())));

	vector<Value> decoder_ort_outputs = this->decoder_ort_session->Run(runOptions, this->decoder_input_names.data(), decoder_inputTensors.data(), decoder_inputTensors.size(), this->decoder_output_names.data(), this->decoder_output_names.size());
	float* flow = decoder_ort_outputs[0].GetTensorMutableData<float>();
	float* point_cls = decoder_ort_outputs[1].GetTensorMutableData<float>();

	conv0 = nullptr;   ///置空, 防止野指针
	conv1 = nullptr;
	conv2 = nullptr;
	conv3 = nullptr;
	conv4 = nullptr;
	conv_fea = nullptr;
	pre = nullptr;
	
	vector<float> tps2flow;
	vector<float> output_tps(1*3*img_h*img_w);
	vector<int> out_tps_shape = {1, 3, img_h, img_w};
	///float* output_tps = new float[3*img_h*img_w];  ///也可以使用new的方式分配内存
	Mat ori_mesh;
	for(int n=0;n<this->head_num;n++)
	{
		Mat rigid_mesh = get_rigid_mesh(this->encoder_input_height, this->encoder_input_width, this->tps_points[n]-1, this->tps_points[n]-1);
		ori_mesh = rigid_mesh.clone();
		int row_ind = 0;
		for (i = 0; i < ori_mesh.size[1]; i++)
		{
			for (int j = 0; j < ori_mesh.size[2]; j++)
			{
				float clamped_x = rigid_mesh.ptr<float>(0, i, j)[0] + offset[n][row_ind*2];
				clamped_x = std::max(0.0f, std::min(clamped_x, float(this->encoder_input_height - 1)));
				float clamped_y = rigid_mesh.ptr<float>(0, i, j)[1] + offset[n][row_ind*2+1];
				clamped_y = std::max(0.0f, std::min(clamped_y, float(this->encoder_input_width - 1)));

				ori_mesh.ptr<float>(0, i, j)[0] = clamped_x;
				ori_mesh.ptr<float>(0, i, j)[1] = clamped_y;
				row_ind++;
			}
		}

		Mat norm_rigid_mesh = get_norm_mesh(rigid_mesh, this->encoder_input_height, this->encoder_input_width);
    	Mat norm_ori_mesh = get_norm_mesh(ori_mesh, this->encoder_input_height, this->encoder_input_width);
		auto start_time_process = std::chrono::high_resolution_clock::now();
		transformer_(tps2flow, norm_rigid_mesh, norm_ori_mesh, img_h, img_w, false);
		auto end_time_model_process = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff_model_process = end_time_model_process - start_time_process;
		cout<<"transformer_ waste time "<<to_string(diff_model_process.count())<<" s"<<endl;
		resample_image_xy(output_tps.data(), this->input1_tensor.data(), {1, 3, img_h, img_w}, tps2flow.data(), {1, 2, img_h, img_w}); /// vector.data()返回指向数组的指针
		///resample_image_xy_vector(output_tps, this->input1_tensor, {1, 3, img_h, img_w}, tps2flow, {1, 2, img_h, img_w});
	}

	vector<float> flow_(this->decoder_flow_out_len);
	memcpy(flow_.data(), flow, this->decoder_flow_out_len*sizeof(float));
	vector<int> flow_shape(this->decoder_flow_out_shape);
	if(resize_flow)
	{
		flow_.resize(this->decoder_flow_out_shape[0]*this->decoder_flow_out_shape[1]*img_h*img_w);
		flow_shape[2] = img_h;
		flow_shape[3] = img_w;
		interpolate(flow, this->decoder_flow_out_shape, flow_.data(), img_h, img_w, "bilinear", true, false);
		const float scale_H = (float)img_h / this->encoder_input_height;
		const float scale_W = (float)img_w / this->encoder_input_width;
		const int image_area = img_h * img_w;
		float* pdata = flow_.data();    ///指向数组的指针比直接访问vector更高效
		for(i=0;i<img_h;i++)
		{
			for(int j=0;j<img_w;j++)
			{
				pdata[i*img_w+j] *= scale_W;
				pdata[image_area + i*img_w+j] *= scale_H;
			}
		}
		
	}
	vector<float> final_flow(flow_.size());
	auto a = std::chrono::high_resolution_clock::now();
	for(i=0;i<flow_.size();i++)
	{
		final_flow[i] = flow_[i] + tps2flow[i];
	}
	auto b = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> c = b - a;
    cout<<"final_flow = flow + tps_flow waste time "<<to_string(c.count())<<" s , vector length = "<<final_flow.size()<<endl;

	vector<float> output_flow(output_tps.size());
	resample_image_xy(output_flow.data(), output_tps.data(), out_tps_shape, flow_.data(), flow_shape);

	Mat warp_tps_np = convert4dtoimage(output_tps, out_tps_shape);
	Mat warp_flow_np = convert4dtoimage(output_flow, out_tps_shape);
	Mat input_with_mesh = draw_mesh_on_warp(img, ori_mesh, this->tps_points[3]-1, this->tps_points[3]-1);
	vector<Mat> results;
	results.emplace_back(warp_tps_np);
	results.emplace_back(input_with_mesh);
	results.emplace_back(warp_flow_np);
	
	// delete [] output_tps;  ////用完了记得释放内存并且置空
	// output_tps = nullptr;
	return results;
}



int main()
{
	std::map<int, string> task_dict = {{0, "Stitched Image"}, {1, "Rectified Wide-Angle Image"}, {2, "Unrolling Shutter Image"}, {3, "Rotated Image"}, {4, "Fisheye Image"}};
	const int task_id = 1;
	cout<<"任务类型: "<<task_dict[task_id]<<endl;
	MOWA mynet;
    
	string imgname = "24.jpg";
	string imgpath = "/home/wangbo/mowa/Datasets/" + task_dict[task_id] + "/input/";
	string maskpath = "/home/wangbo/mowa/Datasets/" + task_dict[task_id] + "/mask/";
	imgpath.append(imgname);
	maskpath.append(imgname);

	Mat srcimg = imread(imgpath);
	Mat mask_img = imread(maskpath, IMREAD_GRAYSCALE);
	if(srcimg.empty() || mask_img.empty())
	{
		cout<<"opencv读取图片为空, 请检查输入图片和mask的路径"<<endl;
		return -1;
	}
	vector<Mat> results = mynet.detect(srcimg, mask_img, true);
	
	imwrite("source.jpg", srcimg);
	imwrite("mesh.jpg", results[0]);
	imwrite("grid_mesh.jpg", results[1]);
	imwrite("flow.jpg", results[2]);

	return 0;
}