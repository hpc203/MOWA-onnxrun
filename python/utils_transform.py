import numpy as np
import cv2
import numpy_tps_transform
import numpy_tps_upsample
import numpy_grid_sample


def get_rigid_mesh(batch_size, height, width, grid_w, grid_h): 
    ww = np.matmul(np.ones([grid_h+1, 1]), np.expand_dims(np.linspace(0., float(width), grid_w+1), 0))
    hh = np.matmul(np.expand_dims(np.linspace(0.0, float(height), grid_h+1), 1), np.ones([1, grid_w+1]))
    
    ori_pt = np.concatenate((np.expand_dims(ww, 2), np.expand_dims(hh,2)), axis=2)[np.newaxis, :]  ###batchsize=1
    ori_pt = np.tile(ori_pt, (batch_size, 1, 1, 1))
    return ori_pt

def get_norm_mesh(mesh, height, width):
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = np.stack([mesh_w, mesh_h], axis=3) 
    
    return norm_mesh.reshape((1, -1, 2))

def transform_tps_fea(offset, input_tensor, grid_w, grid_h, dim, h, w):
    input_tensor = np.transpose(input_tensor, (0,2,1)).reshape((-1, dim, h, w))
    batch_size, _, img_h, img_w = input_tensor.shape    ###img_h和img_w就是输入参数里的h和w
    
    mesh_motion = offset.reshape(-1, grid_h+1, grid_w+1, 2)
    
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w, grid_w, grid_h)
    ori_mesh = rigid_mesh + mesh_motion
    
    clamped_x = np.clip(ori_mesh[..., 0], 0, img_h - 1)
    clamped_y = np.clip(ori_mesh[..., 1], 0, img_w - 1)
    ori_mesh = np.stack((clamped_x, clamped_y), axis=-1)

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_ori_mesh = get_norm_mesh(ori_mesh, img_h, img_w)
    
    output_tps = numpy_tps_transform.transformer(input_tensor, norm_rigid_mesh, norm_ori_mesh, (img_h, img_w))
    output_tps = np.transpose(output_tps.reshape((-1, dim, h*w)), (0,2,1))
        
    return output_tps.astype(np.float32)

def upsample_tps(offset, grid_w, grid_h, out_h, out_w):
    if(grid_w+1 == out_w):
        return offset
    
    else:
        batch_size, *_ = offset.shape
        mesh_motion = offset.reshape((-1, grid_h+1, grid_w+1, 2))
        
        rigid_mesh = get_rigid_mesh(batch_size, out_h, out_w, grid_w, grid_h)
        ori_mesh = rigid_mesh + mesh_motion

        norm_rigid_mesh = get_norm_mesh(rigid_mesh, out_h, out_w)
        norm_ori_mesh = get_norm_mesh(ori_mesh, out_h, out_w)
        
        up_points = numpy_tps_upsample.transformer(norm_rigid_mesh, norm_ori_mesh, (out_h, out_w))
        out = np.transpose(up_points, (0, 2, 3, 1)).reshape((-1, out_h*out_w, 2))
            
        return out.astype(np.float32)

def get_coordinate_xy(shape, det_uv):
    b, _, h, w = shape
    uv_d = np.zeros([h, w, 2], np.float32)

    for j in range(0, h):
        for i in range(0, w):
            uv_d[j, i, 0] = i
            uv_d[j, i, 1] = j

    uv_d = np.expand_dims(uv_d.swapaxes(2, 1).swapaxes(1, 0), 0)
    '''
    (1) np.repeat() 用来重复数组元素，重复的元素放在原元素的临近位置。
    (2) torch.repeat()和np.tile()函数类似，是将整个数组进行复制而非数组元素重复指定的次数。
    '''
    uv_d = np.tile(uv_d, (b, 1, 1, 1))
    # uv_d = uv_d.repeat(b, 1, 1, 1)
    det_uv = uv_d + det_uv
    return det_uv

def uniform_xy(shape, uv):
    b, _, h, w = shape
    y0 = (h - 1) / 2.
    x0 = (w - 1) / 2.

    nor = uv.copy()
    nor[:, 0, :, :] = (uv[:, 0, :, :] - x0) / x0 
    nor[:, 1, :, :] = (uv[:, 1, :, :] - y0) / y0
    nor = np.transpose(nor, (0, 2, 3, 1))  # b w h 2

    return nor.astype(np.float32)

def resample_image_xy(feature, flow):
    uv = get_coordinate_xy(feature.shape, flow)
    grid = uniform_xy(feature.shape, uv)
    target_image = numpy_grid_sample.grid_sample(feature, grid)
    return target_image

def draw_mesh_on_warp(warp, f_local, grid_h, grid_w):
    height = warp.shape[0]
    width = warp.shape[1]
    
    min_w = np.minimum(np.min(f_local[:,:,0]), 0).astype(np.int32)
    max_w = np.maximum(np.max(f_local[:,:,0]), width).astype(np.int32)
    min_h = np.minimum(np.min(f_local[:,:,1]), 0).astype(np.int32)
    max_h = np.maximum(np.max(f_local[:,:,1]), height).astype(np.int32)
    cw = max_w - min_w
    ch = max_h - min_h
    
    pic = np.ones([ch+10, cw+10, 3], np.int32)*255
    pic[0-min_h+5:0-min_h+height+5, 0-min_w+5:0-min_w+width+5, :] = warp
    
    warp = pic.astype(np.uint8)
    f_local[:,:,0] = f_local[:,:,0] - min_w+5
    f_local[:,:,1] = f_local[:,:,1] - min_h+5
    
    point_color = (0, 255, 0)
    thickness = 2
    lineType = 8
    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
              
    return warp