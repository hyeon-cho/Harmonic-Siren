from utils.discretization import batchwise_bit_decomposition as rgb2bit
import torch 
import torch.nn.functional as F

def calculate_ber(pred_tensor, gt_tensor):
    batch_size, num_channels, height, width = pred_tensor.size()

    bit_pred = rgb2bit(pred_tensor, bit=8)
    bit_gt   = rgb2bit(gt_tensor, bit=8)

    ber_lst = [0]*8
    for bit_idx in range(8):
        incorrect_pixels = torch.sum(torch.abs(bit_pred[:, bit_idx, ...] - bit_gt[:, bit_idx, ...])) 
        total_bits = batch_size * num_channels * height * width 
        ber = incorrect_pixels / total_bits 
        ber_lst[bit_idx] = ber.item()

    incorrect_pixels = torch.sum(torch.abs(bit_pred - bit_gt))  
    total_bits = batch_size * num_channels * height * width * 8 
    ber = incorrect_pixels / total_bits 

    assert sum(ber_lst) / 8 - ber.item() < 1e-5

    return ber.item(), ber_lst

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

from skimage.metrics import structural_similarity as ssim
def calc_ssim(sr, hr):
    sr = sr.permute(0, 2, 3, 1).cpu().numpy()
    hr = hr.permute(0, 2, 3, 1).cpu().numpy()
    ssim_val = ssim(hr.squeeze(0), sr.squeeze(0), channel_axis=-1, data_range=1)
    return ssim_val


_lpips_model = None
def _get_lpips(device):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    return _lpips_model

def calc_lpips(sr, hr):
    """LPIPS distance. sr, hr in [0,1], shape (N,3,H,W). Returns float."""
    device = sr.device if sr.is_cuda else hr.device
    if not sr.is_cuda and not hr.is_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _get_lpips(device)
    with torch.no_grad():
        x = (sr.to(device) * 2 - 1).clamp(-1, 1)
        y = (hr.to(device) * 2 - 1).clamp(-1, 1)
        d = model(x, y)
    return d.mean().item()

def compute_gradient_magnitude_rgb(patch):
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32, device=patch.device).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [0,  0,  0],
                                   [1,  2,  1]], dtype=torch.float32, device=patch.device).unsqueeze(0).unsqueeze(0)
    
    gradient_magnitudes = []
    
    for c in range(patch.shape[1]):
        channel = patch[:, c:c+1, :, :]  # Shape: (1, 1, H, W)
        
        grad_x = F.conv2d(channel, sobel_kernel_x, padding=1)
        grad_y = F.conv2d(channel, sobel_kernel_y, padding=1)
        
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        avg_grad_mag = torch.mean(grad_mag)
        gradient_magnitudes.append(avg_grad_mag)
    
    average_gradient_magnitude = torch.mean(torch.stack(gradient_magnitudes))
    
    return average_gradient_magnitude.item()
