import numpy as np


def _meshgrid(height, width, source):

    x_t = np.matmul(np.ones([height, 1]), np.expand_dims(np.linspace(-1.0, 1.0, width), axis=0))
    y_t = np.matmul(np.expand_dims(np.linspace(-1.0, 1.0, height), axis=1), np.ones([1, width]))
    
    x_t_flat = x_t.reshape([1, 1, -1])
    y_t_flat = y_t.reshape([1, 1, -1])

    num_batch = source.shape[0]
    px = np.expand_dims(source[:,:,0], axis=2)
    py = np.expand_dims(source[:,:,1], axis=2)
    
    d2 = np.square(x_t_flat - px) + np.square(y_t_flat - py)
    r = d2 * np.log(d2 + 1e-9)
    
    ones = np.ones_like(x_t_flat)
    
    grid = np.concatenate((ones, x_t_flat, y_t_flat, r), axis=1)
    
    return grid

def _transform(T, source, out_size):
    num_batch, *_ = T.shape

    out_height, out_width = out_size[0], out_size[1]
    grid = _meshgrid(out_height, out_width, source)
    
    # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
    # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
    T_g = np.matmul(T, grid)
    x_s = T_g[:,0,:]
    y_s = T_g[:,1,:]
    flow_x = (x_s - grid[:,1,:])*(out_width/2)
    flow_y = (y_s - grid[:,2,:])*(out_height/2)

    flow = np.stack([flow_x, flow_y], axis=1)
    flow = flow.reshape([num_batch, 2, out_height, out_width])
    return flow

def _solve_system(source, target):
    num_batch  = source.shape[0]
    num_point  = source.shape[1]
    
    np.set_printoptions(precision=8)
    
    ones = np.ones((num_batch, num_point, 1), dtype=np.float32)
    p = np.concatenate([ones, source], axis=2)
    
    p_1 = p.reshape([num_batch, -1, 1, 3])
    p_2 = p.reshape([num_batch, 1, -1, 3])
    d2 = np.sum(np.square(p_1-p_2), axis=3)

    r = d2 * np.log(d2 + 1e-9)
    
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

def transformer(source, target, out_size):
    """
    Thin Plate Spline Spatial Transformer Layer
    convert the TPS deformation into optical flows 
    TPS control points are arranged in arbitrary positions given by `source`.
    source : float Tensor [num_batch, num_point, 2] 
        The source position of the control points.
    target : float Tensor [num_batch, num_point, 2]
        The target position of the control points.
    out_size: tuple of two integers [height, width]
    The size of the output of the network (height, width)
    """
  
    T = _solve_system(source, target)
    output = _transform(T, source, out_size)

    return output