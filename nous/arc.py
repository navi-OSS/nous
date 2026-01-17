"""
Nous ARC Module (System 5).
Provides the `SoftGrid` primitive for differentiable 2D manipulation.
"""
import torch
import torch.nn.functional as F

class SoftGrid:
    """
    A differentiable 2D Grid wrapper.
    Wrapper around a [H, W, C] or [H, W] tensor.
    """
    def __init__(self, data):
        """
        Args:
            data: list of lists, or torch.Tensor. 
                  Shape should be [H, W] (indices) or [H, W, C] (embeddings).
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Ensure float for differentiability
        if data.dtype == torch.long or data.dtype == torch.int:
            data = data.float()
            
        self.data = data
        
    @property
    def shape(self):
        return self.data.shape
        
    @property
    def height(self):
        return self.data.shape[0]
        
    @property
    def width(self):
        return self.data.shape[1]

    def crop(self, x, y, h, w):
        """
        Crop a subgrid.
        Args:
            x, y: Top-left corner (0-indexed).
            h, w: Height and width of crop.
        """
        # Note: Differentiable cropping usually requires spatial transformer networks
        # if x, y are continuous. Here we assume integer indices for simplicity
        # or rely on Soft Indexing if we build it deep.
        # For System 5 V1, we use native slicing (discrete x, y).
        
        # Ensure indices are integers
        ix = int(x) if isinstance(x, (float, int)) else x.item()
        iy = int(y) if isinstance(y, (float, int)) else y.item()
        ih = int(h) if isinstance(h, (float, int)) else h.item()
        iw = int(w) if isinstance(w, (float, int)) else w.item()
        
        # Slice
        new_data = self.data[iy:iy+ih, ix:ix+iw]
        return SoftGrid(new_data)
        
    def paste(self, other, x, y):
        """
        Paste another grid onto this one at (x, y).
        Returns a NEW SoftGrid (functional style).
        """
        ix = int(x)
        iy = int(y)
        oh = other.height
        ow = other.width
        
        # Clone to avoid in-place mutation issues with gradients?
        # Actually in-place is fine if carefully managed, but functional is safer for Nous.
        new_data = self.data.clone()
        new_data[iy:iy+oh, ix:ix+ow] = other.data
        return SoftGrid(new_data)
        
    def resize(self, h, w):
        """Resize using bilinear interpolation."""
        # Expects [N, C, H, W] for functional.interpolate
        # Self data is [H, W] or [H, W, C].
        # Let's standardize on [H, W, C] internally for interpolation?
        # Or just unsafe squeeze/unsqueeze.
        
        tensor = self.data
        if tensor.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            target_shape = (h, w)
            out = F.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
            out = out.squeeze(0).squeeze(0)
        else:
            # [H, W, C] -> [1, C, H, W] (Permute)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            target_shape = (h, w)
            out = F.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
            # Back to [H, W, C]
            out = out.squeeze(0).permute(1, 2, 0)
            
        return SoftGrid(out)
        
    def map_pixels(self, func):
        """
        Apply a function to every pixel.
        func: lambda pixel_val -> new_val
        """
        # This is effectively just func(self.data) if func is vectorized
        # But explicitly supports elementwise map if needed.
        return SoftGrid(func(self.data))
        
    def __repr__(self):
        return f"SoftGrid(shape={tuple(self.data.shape)})"
