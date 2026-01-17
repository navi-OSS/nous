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
        Differentiable Crop using Spatial Transformer Network (STN).
        Allows gradients to flow through x, y, h, w coordinates.
        """
        # Ensure Inputs are Tensors
        if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.float32, device=self.data.device)
        if not torch.is_tensor(y): y = torch.tensor(y, dtype=torch.float32, device=self.data.device)
        if not torch.is_tensor(h): h = torch.tensor(h, dtype=torch.float32, device=self.data.device)
        if not torch.is_tensor(w): w = torch.tensor(w, dtype=torch.float32, device=self.data.device)
        
        H, W = self.height, self.width
        
        # Affine Matrix Construction
        # Center in NCHW range [-1, 1]
        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_norm = (cx / W) * 2.0 - 1.0
        cy_norm = (cy / H) * 2.0 - 1.0
        
        sx = w / W
        sy = h / H
        
        theta = torch.zeros(1, 2, 3, device=self.data.device)
        theta[0, 0, 0] = sx
        theta[0, 1, 1] = sy
        # Translation: 0*sx + tx = cx_norm => tx = cx_norm
        theta[0, 0, 2] = (x + w/2.0) / W * 2.0 - 1.0
        theta[0, 1, 2] = (y + h/2.0) / H * 2.0 - 1.0

        out_h = int(h.item())
        out_w = int(w.item())
        
        img = self._to_4d()
        grid = F.affine_grid(theta, [1, img.shape[1], out_h, out_w], align_corners=False)
        sampled = F.grid_sample(img, grid, align_corners=False)
        return SoftGrid(self._from_4d(sampled))
        
    def paste(self, other, x, y):
        # Todo: Make differentiable using soft scatter?
        # For now, back to discrete assign but warn about grads.
        ix = int(x)
        iy = int(y)
        oh = other.height
        ow = other.width
        new_data = self.data.clone()
        new_data[iy:iy+oh, ix:ix+ow] = other.data
        return SoftGrid(new_data)

    def resize(self, h, w):
        """Resize using bilinear interpolation (Scale)."""
        tensor = self.data
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            out = F.interpolate(tensor, size=(h, w), mode='bilinear', align_corners=False)
            out = out.squeeze(0).squeeze(0)
        else:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            out = F.interpolate(tensor, size=(h, w), mode='bilinear', align_corners=False)
            out = out.squeeze(0).permute(1, 2, 0)
        return SoftGrid(out)
        
    def scale(self, h, w):
        """Alias for resize."""
        return self.resize(h, w)

    def canvas_resize(self, h, w, anchor_y=0.5, anchor_x=0.5, fill_value=0.0):
        """Change grid dimensions via Padding/Cropping (No distortion)."""
        old_h, old_w = self.height, self.width
        if self.data.dim() == 2:
            new_shape = (int(h), int(w))
        else:
            new_shape = (int(h), int(w), self.data.shape[2])
        canvas = torch.full(new_shape, fill_value, dtype=self.data.dtype, device=self.data.device)
        
        start_y = int((h - old_h) * anchor_y)
        start_x = int((w - old_w) * anchor_x)
        
        src_y1, src_x1 = max(0, -start_y), max(0, -start_x)
        src_y2, src_x2 = min(old_h, h - start_y), min(old_w, w - start_x)
        dst_y1, dst_x1 = max(0, start_y), max(0, start_x)
        dst_y2, dst_x2 = dst_y1 + (src_y2 - src_y1), dst_x1 + (src_x2 - src_x1)
        
        if (dst_y2 > dst_y1) and (dst_x2 > dst_x1):
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = self.data[src_y1:src_y2, src_x1:src_x2]
        return SoftGrid(canvas)

    def rotate(self, degrees):
        """Rotate grid by degrees."""
        if isinstance(degrees, (int, float)) and degrees % 90 == 0:
            k = int(degrees // 90)
            return SoftGrid(torch.rot90(self.data, k, dims=[0, 1] if self.data.dim()==2 else [0, 1]))
        
        angle = -float(degrees) * 3.14159 / 180.0
        theta = torch.tensor([
            [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0],
            [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0]
        ], device=self.data.device).unsqueeze(0)
        
        img = self._to_4d()
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        rotated = F.grid_sample(img, grid, align_corners=False)
        return SoftGrid(self._from_4d(rotated))

    def flip(self, axis):
        return SoftGrid(torch.flip(self.data, dims=[axis]))

    def replace_color(self, old_color, new_color, temperature=10.0):
        dist = torch.abs(self.data - old_color)
        mask = torch.exp(-temperature * (dist**2))
        out = mask * new_color + (1.0 - mask) * self.data
        return SoftGrid(out)
        
    def find(self, template):
        if isinstance(template, SoftGrid): template = template.data
        img = self._to_4d()
        if template.dim() == 2: template = template.unsqueeze(0).unsqueeze(0)
        res = F.conv2d(img, template, padding='same')
        return SoftGrid(self._from_4d(res))
        
    def map_pixels(self, func):
        return SoftGrid(func(self.data))

    def _to_4d(self):
        d = self.data
        if d.dim() == 2: return d.unsqueeze(0).unsqueeze(0)
        if d.dim() == 3: return d.permute(2, 0, 1).unsqueeze(0)
        return d

    def _from_4d(self, t):
        t = t.squeeze(0)
        if self.data.dim() == 2: return t.squeeze(0)
        return t.permute(1, 2, 0)
    
    def __repr__(self):
        return f"SoftGrid(shape={tuple(self.data.shape)})"
