import torch 


def binary_to_gray_1003(binary_code):
    """
    Convert binary code to Gray code.
    
    Args:
        binary_code (torch.Tensor): Input binary code tensor with shape (batch, bit=8, channel=3, height, width).
        
    Returns:
        torch.Tensor: Gray code tensor with the same shape as binary_code.
    """
    # Initialize the Gray code tensor with the same shape as the binary code
    gray_code = torch.zeros_like(binary_code)
    
    # Set the MSB of the Gray code to be the same as the MSB of the binary code
    gray_code[:, 7, :, :, :] = binary_code[:, 7, :, :, :]
    # Compute the Gray code bits
    for i in range(7):
        gray_code[:, i, :, :, :] = binary_code[:, i + 1, :, :, :].to(torch.int8) ^ binary_code[:, i, :, :, :].to(torch.int8)
    
    return gray_code.float()

def gray_to_binary_1003(gray_code):
    """
    Convert Gray code back to binary code.
    
    Args:
        gray_code (torch.Tensor): Input Gray code tensor with shape (batch, bit=8, channel=3, height, width).
        
    Returns:
        torch.Tensor: Binary code tensor with the same shape as gray_code.
    """
    # Initialize the binary code tensor with the same shape as the Gray code
    binary_code = torch.zeros_like(gray_code)
    
    # Set the MSB of the binary code to be the same as the MSB of the Gray code
    binary_code[:, 7, :, :, :] = gray_code[:, 7, :, :, :]
    
    # Compute the binary code bits from MSB-1 down to LSB
    for i in range(6, -1, -1):
        binary_code[:, i, :, :, :] = binary_code[:, i + 1, :, :, :].to(torch.int8) ^ gray_code[:, i, :, :, :].to(torch.int8)
    
    return binary_code.float()

