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

        mask = np.ones((1, 1, 256, 256), dtype=np.uint8) * 255  ###是一个定值，没别要在每次输入图片时去生成
        mask = mask.astype(np.float32) / 255.0
        self.mask_tensor = mask

    
    def detect(self, srcimg, resize_flow=False):
        ori_height, ori_width, _ = srcimg.shape
        
        input_ = cv2.resize(srcimg, (self.encoder_input_width, self.encoder_input_height))
        input_ = input_.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_, [2, 0, 1])[np.newaxis, :]   ###也可以用np.expand_dims
        del input_

        conv0, conv1, conv2, conv3, conv4 = self.encoder.run(self.encoder_output_names, {self.encoder_input_names[0]:input_tensor, self.encoder_input_names[1]:self.mask_tensor})

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
        batch_size, _, img_h, img_w = input_tensor.shape
        batch_size, _, input_size, input_size = input_tensor.shape
        head_num = len(offset)
        norm_rigid_mesh_list = []
        norm_ori_mesh_list = []
        output_tps_list = []
        ori_mesh_list = []
        tps2flow_list = []
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
            output_tps = resample_image_xy(input_tensor, tps2flow)
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
        
        pred = out_dict['flow3'].squeeze(axis=0)
        pflow = np.transpose(pred, (1, 2, 0))
        predflow_x, predflow_y = pflow[:, :, 0], pflow[:, :, 1]

        scale_x = ori_width / predflow_x.shape[1]
        scale_y = ori_height / predflow_x.shape[0]
        predflow_x = cv2.resize(predflow_x, (ori_width, ori_height)) * scale_x
        predflow_y = cv2.resize(predflow_y, (ori_width, ori_height)) * scale_y

        # Get the [predicted image]"""
        ys, xs = np.mgrid[:ori_height, :ori_width]
        mesh_x = predflow_x.astype(np.float32) + xs.astype(np.float32)
        mesh_y = predflow_y.astype(np.float32) + ys.astype(np.float32)
        pred_out = cv2.remap(srcimg, mesh_x, mesh_y, cv2.INTER_LINEAR).astype(np.uint8)
        return pred_out
    
if __name__=='__main__':
    imgpath = '/home/wangbo/mowa/Datasets/Portrait Photo/0001_n21.jpg'
    
    myNet = MOWA()
    srcimg = cv2.imread(imgpath)
    ori_height, ori_width, _ = srcimg.shape 

    pred_out = myNet.detect(srcimg, resize_flow=False)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('source image', color='red')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(pred_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Portrait Photo', color='red')

    plt.suptitle('Portrait Correction', fontsize=15, color='blue')
    # plt.show()
    plt.savefig(f'Portrait Correction.jpg', dpi=500, bbox_inches='tight') ###保存高清图


    # savepath = os.path.join(os.getcwd(), 'results', 'Portrait Correction')
    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # os.makedirs(savepath)
    # cv2.imwrite(os.path.join(savepath, "source_pred.jpg"), np.hstack((srcimg, np.zeros((ori_height,20,3), dtype=np.uint8)+255, pred_out)))