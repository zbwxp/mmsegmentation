import torch

def compute_locations_per_level(h, w):
    shifts_x = torch.arange(
        0, 1, step=1 / w,
        dtype=torch.float32, device='cuda'
    )
    shifts_y = torch.arange(
        0, 1, step=1 / h,
        dtype=torch.float32, device='cuda'
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    locations = torch.stack((shift_x, shift_y), dim=0)
    return locations