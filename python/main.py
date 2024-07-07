import cv2
import numpy as np
import onnxruntime
import os, shutil
import matplotlib.pyplot as plt
from utils_transform import transform_tps_fea, upsample_tps, get_rigid_mesh, get_norm_mesh, resample_image_xy, draw_mesh_on_warp
import numpy_tps_upsample
import numpy_upsample

class MOWA():
    def __init__(self):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.encoder = onnxruntime.InferenceSession('weights/encoder.onnx', so)
        _, _, self.encoder_input_height, self.encoder_input_width = self.encoder.get_inputs()[0].shape
        self.encoder_input_names = [input.name for input in self.encoder.get_inputs()]
        self.encoder_output_names = [out.name for out in self.encoder.get_outputs()]

        self.tps_regression_heads0 = onnxruntime.InferenceSession('weights/tps_regression_heads0.onnx', so)
        _, self.heads_input_height, self.heads_input_width = self.tps_regression_heads0.get_inputs()[0].shape
        self.heads_input_name = self.tps_regression_heads0.get_inputs()[0].name   ##### 4个heads模型的输入节点名称和形状都是一样的, 输出节点名称也是一样的
        self.tps_regression_heads1 = onnxruntime.InferenceSession('weights/tps_regression_heads1.onnx', so)
        self.tps_regression_heads2 = onnxruntime.InferenceSession('weights/tps_regression_heads2.onnx', so)
        self.tps_regression_heads3 = onnxruntime.InferenceSession('weights/tps_regression_heads3.onnx', so)

        self.decoder = onnxruntime.InferenceSession('weights/point_classification_decoder.onnx', so)
        self.decoder_inputs_hxw = [(input.shape[1], input.shape[2]) for input in self.decoder.get_inputs()]
        self.decoder_input_names = [input.name for input in self.decoder.get_inputs()]
        self.decoder_output_names = [out.name for out in self.decoder.get_outputs()]

        self.tps_points = [10, 12, 14, 16]
        self.head_num = 4
        self.down_size = 16
        self.embed_dim = 32
        self.mini_size = 16
    
    def detect(self, srcimg, mask_img, resize_flow=False):
        input1 = srcimg.astype(np.float32) / 255.0
        input1_tensor = np.transpose(input1, [2, 0, 1])[np.newaxis, :]  ###也可以用np.expand_dims
        del input1
        
        input2 = cv2.resize(srcimg, (self.encoder_input_width, self.encoder_input_height))
        input2 = input2.astype(np.float32) / 255.0
        input2_tensor = np.transpose(input2, [2, 0, 1])[np.newaxis, :]
        del input2

        mask = cv2.resize(mask_img, (self.encoder_input_width, self.encoder_input_height))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(np.float32) / 255.0
        mask_tensor = np.transpose(mask, [2, 0, 1])[np.newaxis, :]
        del mask

        conv0, conv1, conv2, conv3, conv4 = self.encoder.run(self.encoder_output_names, {self.encoder_input_names[0]:input2_tensor, self.encoder_input_names[1]:mask_tensor})

        tps = []
        tps_up = []
        fea = [conv4]
        
        i = 0
        conv_fea = fea[-1]
        pre = self.tps_regression_heads0.run(None, {self.heads_input_name:conv_fea})[0]
        warp = transform_tps_fea(pre/self.down_size, conv_fea, self.tps_points[i]-1, self.tps_points[i]-1, self.embed_dim*16, self.mini_size, self.mini_size)
        fea.append(warp)
        tps.append(pre)
        tps_up.append(upsample_tps(tps[i], self.tps_points[i]-1, self.tps_points[i]-1, self.tps_points[i+1], self.tps_points[i+1]))

        i = 1
        conv_fea = fea[-1]
        pre = self.tps_regression_heads1.run(None, {self.heads_input_name:conv_fea})[0]
        warp = transform_tps_fea(pre/self.down_size, conv_fea, self.tps_points[i]-1, self.tps_points[i]-1, self.embed_dim*16, self.mini_size, self.mini_size)
        fea.append(warp)
        tps.append(pre + tps_up[-1])
        tps_up.append(upsample_tps(tps[i], self.tps_points[i]-1, self.tps_points[i]-1, self.tps_points[i+1], self.tps_points[i+1]))
        
        i = 2
        conv_fea = fea[-1]
        pre = self.tps_regression_heads2.run(None, {self.heads_input_name:conv_fea})[0]
        warp = transform_tps_fea(pre/self.down_size, conv_fea, self.tps_points[i]-1, self.tps_points[i]-1, self.embed_dim*16, self.mini_size, self.mini_size)
        fea.append(warp)
        tps.append(pre + tps_up[-1])
        tps_up.append(upsample_tps(tps[i], self.tps_points[i]-1, self.tps_points[i]-1, self.tps_points[i+1], self.tps_points[i+1]))

        i = 3
        conv_fea = fea[-1]
        pre = self.tps_regression_heads3.run(None, {self.heads_input_name:conv_fea})[0]
        warp = transform_tps_fea(pre/self.down_size, conv_fea, self.tps_points[i]-1, self.tps_points[i]-1, self.embed_dim*16, self.mini_size, self.mini_size)
        fea.append(warp)
        tps.append(pre + tps_up[-1])

        decoder_input_dict = {self.decoder_input_names[0]:tps[-1], self.decoder_input_names[1]:conv4}
        decoder_input_dict.update({self.decoder_input_names[2]:fea[-1], self.decoder_input_names[3]:conv3})
        decoder_input_dict.update({self.decoder_input_names[4]:conv2, self.decoder_input_names[5]:conv1, self.decoder_input_names[6]:conv0})
        flow, point_cls = self.decoder.run(self.decoder_output_names, decoder_input_dict)
        
        ####后处理
        offset = tps
        batch_size, _, img_h, img_w = input1_tensor.shape
        batch_size, _, input_size, input_size = input2_tensor.shape
        head_num = len(offset)     ###其实就等于self.head_num
        norm_rigid_mesh_list = []  ####append元素之后并没有用被使用到
        norm_ori_mesh_list = []    ####append元素之后并没有用被使用到
        output_tps_list = []       ###其实只有列表里的最后一个元素有被使用到
        ori_mesh_list = []         ###其实只有列表里的最后一个元素有被使用到
        tps2flow_list = []         ###其实只有列表里的最后一个元素有被使用到
        for i in range(head_num):
            mesh_motion = offset[i].reshape((-1, self.tps_points[i], self.tps_points[i], 2))
            rigid_mesh = get_rigid_mesh(batch_size, input_size, input_size, self.tps_points[i]-1, self.tps_points[i]-1)
            ori_mesh = rigid_mesh + mesh_motion
            clamped_x = np.clip(ori_mesh[..., 0], 0, input_size - 1)
            clamped_y = np.clip(ori_mesh[..., 1], 0, input_size - 1)
            ori_mesh = np.stack((clamped_x, clamped_y), axis=-1)
            
            norm_rigid_mesh = get_norm_mesh(rigid_mesh, input_size, input_size)
            norm_ori_mesh = get_norm_mesh(ori_mesh, input_size, input_size)
            tps2flow = numpy_tps_upsample.transformer(norm_rigid_mesh, norm_ori_mesh, (img_h, img_w))
            output_tps = resample_image_xy(input1_tensor, tps2flow)
            norm_rigid_mesh_list.append(norm_rigid_mesh)
            norm_ori_mesh_list.append(norm_ori_mesh)
            output_tps_list.append(output_tps)
            ori_mesh_list.append(ori_mesh)
            tps2flow_list.append(tps2flow)
        
        tps_flow = tps2flow_list[-1]
        if(resize_flow):
            flow = numpy_upsample.interpolate(flow, size=(img_h, img_w), mode="bilinear", align_corners=True)
            scale_H, scale_W = img_h / input_size, img_w / input_size
            flow[:, 0, :, :] *= scale_W
            flow[:, 1, :, :] *= scale_H
        
        final_flow = flow + tps_flow
        output_flow = resample_image_xy(output_tps_list[-1], flow)
        out_dict = {}
        out_dict.update(warp_tps=output_tps_list, warp_flow=output_flow, mesh=ori_mesh_list,
                        flow1=flow, flow2=tps_flow, flow3=final_flow, point_cls=point_cls)
        
        warp_tps, warp_flow, mesh, flow1, flow2, flow3, point_cls = [out_dict[key] for key in ['warp_tps', 'warp_flow', 'mesh', 'flow1', 'flow2', 'flow3', 'point_cls']]
        input_np2 = np.transpose(input2_tensor[0]*255.0, (1,2,0)).astype(np.uint8)
        warp_flow_np = np.transpose(warp_flow[0]*255.0, (1,2,0)).astype(np.uint8)
        warp_tps_np = np.transpose(warp_tps[-1][0]*255.0, (1,2,0)).astype(np.uint8)
        mesh_np = mesh[-1][0]
        input_with_mesh = draw_mesh_on_warp(input_np2, mesh_np, self.tps_points[-1]-1, self.tps_points[-1]-1)
        return warp_tps_np, input_with_mesh, warp_flow_np
    
if __name__=='__main__':
    task_dict = {0:"Stitched Image", 1:"Rectified Wide-Angle Image", 2:"Unrolling Shutter Image", 3:"Rotated Image", 4:"Fisheye Image"}
    
    task_id = 1
    print("任务类型:", task_dict[task_id])
    imgname = '24.jpg'
    imgpath = os.path.join('/home/wangbo/mowa/Datasets', task_dict[task_id], 'input', imgname)
    maskpath = os.path.join('/home/wangbo/mowa/Datasets', task_dict[task_id], 'mask', imgname)
    
    myNet = MOWA()
    srcimg = cv2.imread(imgpath)
    ori_height, ori_width, _ = srcimg.shape 
    mask_img = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)  ###单通道灰度图
    warp_tps_np, input_with_mesh, warp_flow_np = myNet.detect(srcimg, mask_img, resize_flow=True)

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('source image', color='red')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(warp_tps_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('mesh', color='red')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(input_with_mesh, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('grid_mesh', color='red')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(warp_flow_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('flow', color='red')

    fig=plt.gcf()
    fig.set_facecolor('black')   ###黑色背景
    
    plt.suptitle(f'{task_dict[task_id]}', fontsize=15, color='blue')
    # plt.show()
    plt.savefig(f'{task_dict[task_id]}.jpg', dpi=500, bbox_inches='tight') ###保存高清图

    
    # savepath = os.path.join(os.getcwd(), 'results', task_dict[task_id])
    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # os.makedirs(savepath)
    # cv2.imwrite(os.path.join(savepath, "mesh.jpg"), warp_tps_np)
    # cv2.imwrite(os.path.join(savepath, "grid_mesh.jpg"), input_with_mesh)
    # cv2.imwrite(os.path.join(savepath, "flow.jpg"), warp_flow_np)
    # cv2.imwrite(os.path.join(savepath, "source.jpg"), srcimg)
