import numpy as np


def _repeat(x, n_repeats):

    rep = np.ones([n_repeats, ], dtype=np.int32)[np.newaxis, :]
    x = x.astype(np.int32)

    x = np.matmul(x.reshape([-1,1]), rep)
    return x.reshape([-1])

def _interpolate(im, x, y, out_size):

    num_batch, num_channels , height, width = im.shape

    height_f = height
    width_f = width
    out_height, out_width = out_size[0], out_size[1]

    zero = 0
    max_y = height - 1
    max_x = width - 1

    x = (x + 1.0)*(width_f) / 2.0
    y = (y + 1.0) * (height_f) / 2.0

    # sampling
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)
    dim2 = np.array(width)
    dim1 = np.array(width * height)

    base = _repeat(np.arange(0,num_batch) * dim1, out_height * out_width)  ####print(np.unique(base))打印显示全是0的矩阵
    
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # channels dim
    im = np.transpose(im, (0,2,3,1))
    im_flat = im.reshape([-1, num_channels]).astype(np.float32)
    

    idx_a = idx_a[:, np.newaxis].astype(np.int32)
    idx_a = np.tile(idx_a, (1, num_channels))
    Ia = np.take_along_axis(im_flat, idx_a, axis=0)

    idx_b = idx_b[:, np.newaxis].astype(np.int32)
    idx_b = np.tile(idx_b, (1, num_channels))
    Ib = np.take_along_axis(im_flat, idx_b, axis=0)

    idx_c = idx_c[:, np.newaxis].astype(np.int32)
    idx_c = np.tile(idx_c, (1, num_channels))
    Ic = np.take_along_axis(im_flat, idx_c, axis=0)

    idx_d = idx_d[:, np.newaxis].astype(np.int32)
    idx_d = np.tile(idx_d, (1, num_channels))
    Id = np.take_along_axis(im_flat, idx_d, axis=0)

    x0_f = x0.astype(np.float32)
    x1_f = x1.astype(np.float32)
    y0_f = y0.astype(np.float32)
    y1_f = y1.astype(np.float32)

    wa = np.expand_dims(((x1_f - x) * (y1_f - y)), axis=1)
    wb = np.expand_dims(((x1_f - x) * (y - y0_f)), axis=1)
    wc = np.expand_dims(((x - x0_f) * (y1_f - y)), axis=1)
    wd = np.expand_dims(((x - x0_f) * (y - y0_f)), axis=1)
    output = wa*Ia+wb*Ib+wc*Ic+wd*Id

    return output

def _meshgrid(height, width, source):

    x_t = np.matmul(np.ones([height, 1]), np.expand_dims(np.linspace(-1.0, 1.0, width), axis=0))
    y_t = np.matmul(np.expand_dims(np.linspace(-1.0, 1.0, height), axis=1), np.ones([1, width]))
    
    x_t_flat = x_t.reshape([1, 1, -1])
    y_t_flat = y_t.reshape([1, 1, -1])

    num_batch = source.shape[0]
    px = np.expand_dims(source[:,:,0], axis=2)
    py = np.expand_dims(source[:,:,1], axis=2)
    
    d2 = np.square(x_t_flat - px) + np.square(y_t_flat - py)
    r = d2 * np.log(d2 + 1e-6)
    
    ones = np.ones_like(x_t_flat)
    
    grid = np.concatenate((ones, x_t_flat, y_t_flat, r), axis=1)
    
    return grid

def _transform(T, source, input_dim, out_size):
    num_batch, num_channels, height, width = input_dim.shape

    out_height, out_width = out_size[0], out_size[1]
    grid = _meshgrid(out_height, out_width, source)
    
    # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
    # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
    T_g = np.matmul(T, grid)
    x_s = T_g[:,0,:]
    y_s = T_g[:,1,:]
    x_s_flat = x_s.reshape([-1])
    y_s_flat = y_s.reshape([-1])
    
    input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat,out_size)
    
    output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])

    output = np.transpose(output, (0,3,1,2))
    return output


def _solve_system(source, target):
    num_batch  = source.shape[0]
    num_point  = source.shape[1]
    
    ones = np.ones((num_batch, num_point, 1), dtype=np.float32)
    p = np.concatenate([ones, source], axis=2)
    
    p_1 = p.reshape([num_batch, -1, 1, 3])
    p_2 = p.reshape([num_batch, 1, -1, 3])
    d2 = np.sum(np.square(p_1-p_2), axis=3)

    r = d2 * np.log(d2 + 1e-6)
    
    zeros = np.zeros((num_batch, 3, 3), dtype=np.float32)
    W_0 = np.concatenate((p, r), axis=2)
    W_1 = np.concatenate((zeros, np.transpose(p, (0,2,1))), axis=2)
    W = np.concatenate((W_0, W_1), axis=1)
    
    W_inv = np.linalg.inv(W.astype(np.float32))
    zeros2 = np.zeros((num_batch, 3, 2))
    tp = np.concatenate((target, zeros2), axis=1)
    T = np.matmul(W_inv, tp.astype(np.float32))
    T = np.transpose(T, (0, 2, 1))

    return T.astype(np.float32)

def transformer(U, source, target, out_size):
    """
    Thin Plate Spline Spatial Transformer Layer
    TPS control points are arranged in arbitrary positions given by `source`.
    U : float Tensor [num_batch, height, width, num_channels].
        Input Tensor.
    source : float Tensor [num_batch, num_point, 2]
        The source position of the control points.
    target : float Tensor [num_batch, num_point, 2]
        The target position of the control points.
    out_size: tuple of two integers [height, width]
    The size of the output of the network (height, width)
    """
    
    T = _solve_system(source, target)
    output = _transform(T, source, U, out_size)

    return output