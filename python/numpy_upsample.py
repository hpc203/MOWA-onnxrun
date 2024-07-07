import numpy as np


def interpolate(input, size, mode="nearest_neighbor", align_corners=False):
    """
    param input: (N, C, H, W)
    param size: (out_h, out_w)
    param mode: [nearest_neighbor, bilinear]
    param align_corners: [True, False]
    """
    N, C, H, W = input.shape
    out_h, out_w = size
    scales = (out_h/H, out_w/W)

    output = np.zeros((N, C, out_h, out_w), dtype=np.float32)

    if mode == "nearest_neighbor":
        for n in np.arange(N):
            for c in np.arange(C):
                for oh in np.arange(out_h):
                    for ow in np.arange(out_w):
                        ih = oh // scales[0]
                        iw = ow // scales[1]
                        output[n, c, oh, ow] = input[n, c, ih, iw]
    elif mode == "bilinear":
        if align_corners == False:
            hs_p = np.array([(i + 0.5) / out_h * H - 0.5 for i in range(out_h)], dtype=np.float32)
            ws_p = np.array([(i + 0.5) / out_w * W - 0.5 for i in range(out_w)], dtype=np.float32)
        else:
            stride_h = (H - 1) / (out_h - 1)
            stride_w = (W - 1) / (out_w - 1)
            hs_p = np.array([i * stride_h for i in range(out_h)], dtype=np.float32)
            ws_p = np.array([i * stride_w for i in range(out_w)], dtype=np.float32)
        hs_p = np.clip(hs_p, 0, H - 1)
        ws_p = np.clip(ws_p, 0, W - 1)

        """找出每个投影点在原图纵轴方向的近邻点坐标对"""
        # ih_0的取值范围是0 ~(H - 2), 因为ih_1 = ih_0 + 1
        hs_0 = np.clip(np.floor(hs_p), 0, H - 2).astype(np.int32)
        """找出每个投影点在原图横轴方向的近邻点坐标对"""
        # iw_0的取值范围是0 ~(W - 2), 因为iw_1 = iw_0 + 1
        ws_0 = np.clip(np.floor(ws_p), 0, W - 2).astype(np.int32)

        """
        计算目标图各个点的像素值
        """
        us = hs_p - hs_0
        vs = ws_p - ws_0
        _1_us = 1 - us
        _1_vs = 1 - vs
        for n in np.arange(N):
            for c in np.arange(C):
                for oh in np.arange(out_h):
                    ih_0, ih_1 = hs_0[oh], hs_0[oh] + 1 # 原图的坐标
                    for ow in np.arange(out_w):
                        iw_0, iw_1 = ws_0[ow], ws_0[ow] + 1 # 原图的坐标
                        output[n, c, oh, ow] = input[n, c, ih_0, iw_0] * _1_us[oh] * _1_vs[ow] \
                                             + input[n, c, ih_0, iw_1] * _1_us[oh] * vs[ow] \
                                             + input[n, c, ih_1, iw_0] * us[oh] * _1_vs[ow] \
                                             + input[n, c, ih_1, iw_1] * us[oh] * vs[ow]
    return output