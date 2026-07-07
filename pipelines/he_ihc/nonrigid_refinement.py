"""Instance-optimization nonrigid registration (pure PyTorch).

Pure geometric optimization of a per-pixel displacement field
via Adam on a multi-resolution Gaussian pyramid.
"""
import numpy as np
import cv2
import torch as tc
import torch.nn.functional as F


def ncc_local(src, tgt, win_size=7):
    """Local normalized cross-correlation cost (sliding window).
    Negative value — minimization means maximizing correlation."""
    pad = win_size // 2
    sum_filt = tc.ones((1, 1, win_size, win_size), device=src.device, dtype=src.dtype)
    src2 = src ** 2
    tgt2 = tgt ** 2
    st = src * tgt
    s_sum = F.conv2d(src, sum_filt, padding=pad)
    t_sum = F.conv2d(tgt, sum_filt, padding=pad)
    s2_sum = F.conv2d(src2, sum_filt, padding=pad)
    t2_sum = F.conv2d(tgt2, sum_filt, padding=pad)
    st_sum = F.conv2d(st, sum_filt, padding=pad)
    n = win_size * win_size
    u_s, u_t = s_sum / n, t_sum / n
    cross = st_sum - u_t * s_sum - u_s * t_sum + u_s * u_t * n
    s_var = s2_sum - 2 * u_s * s_sum + u_s ** 2 * n
    t_var = t2_sum - 2 * u_t * t_sum + u_t ** 2 * n
    ncc = cross * cross / (s_var * t_var + 1e-5)
    return -ncc.mean()


def diffusion_relative(df):
    """Relative diffusion regularizer — size-invariant smoothness prior."""
    H, W = df.shape[1], df.shape[2]
    dx = ((df[:, 1:, :, :] - df[:, :-1, :, :]) * H) ** 2
    dy = ((df[:, :, 1:, :] - df[:, :, :-1, :]) * W) ** 2
    return (dx.mean() + dy.mean()) / 2


def warp(src, df):
    """Warp source with displacement field df (normalized grid coords)."""
    H, W = src.shape[-2], src.shape[-1]
    yy, xx = tc.meshgrid(
        tc.linspace(-1, 1, H, device=src.device, dtype=src.dtype),
        tc.linspace(-1, 1, W, device=src.device, dtype=src.dtype),
        indexing='ij')
    grid = tc.stack([xx, yy], dim=-1).unsqueeze(0) + df
    return F.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=False)


def preprocess(rgb_np, target_max=2048, device='cuda'):
    """Gray + flip intensity + CLAHE + resample to target_max."""
    if rgb_np.ndim == 3:
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_np
    H, W = gray.shape
    s = target_max / max(H, W)
    if s < 1.0:
        gray = cv2.resize(gray, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    g = 255 - gray
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = g.astype(np.float32) / 255.0
    return tc.from_numpy(g).unsqueeze(0).unsqueeze(0).to(device), s


def nonrigid_register(src, tgt, num_levels=7, used_levels=7,
                      iterations=(100,) * 7,
                      lrs=(0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025),
                      alphas=(1.5,) * 7, win_size=7, device='cuda', echo=False):
    """Multi-resolution Adam optimization of a free displacement field."""
    src_pyr = [src]; tgt_pyr = [tgt]
    for _ in range(num_levels - 1):
        src_pyr.append(F.avg_pool2d(src_pyr[-1], 2))
        tgt_pyr.append(F.avg_pool2d(tgt_pyr[-1], 2))
    src_pyr = src_pyr[::-1]; tgt_pyr = tgt_pyr[::-1]

    H0, W0 = src_pyr[0].shape[-2], src_pyr[0].shape[-1]
    df = tc.zeros((1, H0, W0, 2), device=device, requires_grad=True)

    for lvl in range(used_levels):
        s_lvl = src_pyr[lvl]; t_lvl = tgt_pyr[lvl]
        H, W = s_lvl.shape[-2], s_lvl.shape[-1]
        if (df.shape[1], df.shape[2]) != (H, W):
            df_data = df.detach().permute(0, 3, 1, 2)
            df_data = F.interpolate(df_data, size=(H, W), mode='bicubic', align_corners=False)
            df = df_data.permute(0, 2, 3, 1).contiguous().requires_grad_(True)
        opt = tc.optim.Adam([df], lr=lrs[lvl])
        for it in range(iterations[lvl]):
            opt.zero_grad()
            warped = warp(s_lvl, df)
            cost = ncc_local(warped, t_lvl, win_size=win_size)
            reg = diffusion_relative(df)
            (cost + alphas[lvl] * reg).backward()
            opt.step()
    return df.detach()
