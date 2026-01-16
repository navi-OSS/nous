"""
Nous Symbolic Geometry.
Uses multivariate polynomials and differentiable operations to represent 
and reason about geometric primitives.
"""
import torch
import torch.nn as nn
import math
from .engine import MultivariatePolynomial

class SymbolicGeometry(nn.Module):
    """
    Geometry module for Nous. 
    Represents shapes as differentiable objects.
    """
    def __init__(self):
        super().__init__()
        self.mv = MultivariatePolynomial(num_vars=2)
        
    def create_circle(self, center, radius):
        """
        Creates a circle as an implicit polynomial: (x-xc)^2 + (y-yc)^2 - r^2 = 0
        Returns coefficients for MultivariatePolynomial (3x3 for quadratic in 2D)
        """
        xc, yc = center
        # (x^2 - 2*xc*x + xc^2) + (y^2 - 2*yc*y + yc^2) - r^2
        # Coeffs indices: [x_pow, y_pow]
        coeffs = torch.zeros(3, 3, dtype=torch.float32)
        coeffs[2, 0] = 1.0          # x^2
        coeffs[1, 0] = -2 * xc      # x
        coeffs[0, 2] = 1.0          # y^2
        coeffs[0, 1] = -2 * yc      # y
        coeffs[0, 0] = xc**2 + yc**2 - radius**2 # constant
        return coeffs

    def create_line(self, p1, p2):
        """
        Creates a line through p1 and p2: (y1-y2)x + (x2-x1)y + (x1y2 - x2y1) = 0
        """
        x1, y1 = p1
        x2, y2 = p2
        coeffs = torch.zeros(2, 2, dtype=torch.float32)
        coeffs[1, 0] = y1 - y2      # x
        coeffs[0, 1] = x2 - x1      # y
        coeffs[0, 0] = x1*y2 - x2*y1 # constant
        return coeffs

    def distance_point_to_circle(self, point, circle_params):
        """
        Differentiable distance from a point to a circle boundary.
        circle_params is (xc, yc, r)
        """
        xc, yc, r = circle_params
        xp, yp = point
        dist_to_center = torch.sqrt((xp - xc)**2 + (yp - yc)**2 + 1e-8)
        return torch.abs(dist_to_center - r)

    def intersection_circles(self, c1_params, c2_params):
        """
        Finds intersection points of two circles.
        Returns points (x,y) if they exist.
        Uses the algebraic reduction to a radical.
        """
        x1, y1, r1 = [torch.tensor(p) if not torch.is_tensor(p) else p for p in c1_params]
        x2, y2, r2 = [torch.tensor(p) if not torch.is_tensor(p) else p for p in c2_params]
        
        d2 = (x2 - x1)**2 + (y2 - y1)**2
        d = torch.sqrt(d2 + 1e-8)
        
        # Conditions for intersection
        # We use soft logic for differentiability if needed, but here we use hard checks
        if d > r1 + r2 or d < torch.abs(r1 - r2) or d < 1e-8:
            return None # No intersection or identical
            
        # Distance from c1 to the line joining intersections
        a = (r1**2 - r2**2 + d2) / (2 * d)
        h = torch.sqrt(torch.clamp(r1**2 - a**2, min=1e-10))
        
        x0 = x1 + a * (x2 - x1) / d
        y0 = y1 + a * (y2 - y1) / d
        
        rx = -(y2 - y1) * (h / d)
        ry = (x2 - x1) * (h / d)
        
        return [(x0 + rx, y0 + ry), (x0 - rx, y0 - ry)]


    def area_circle(self, radius):
        return math.pi * radius**2

    def is_inside_circle(self, point, center, radius):
        """
        Differentiable 'insideness' score. Negative if inside.
        """
        xc, yc = center
        xp, yp = point
        return (xp - xc)**2 + (yp - yc)**2 - radius**2

# --- GEOMETRY DEMO ---

def demo_geometry():
    print("=== Nous Symbolic Geometry Demo ===")
    geo = SymbolicGeometry()
    
    # 1. Circle Representation
    c1_center = torch.tensor([0.0, 0.0])
    c1_radius = torch.tensor(5.0)
    coeffs = geo.create_circle(c1_center, c1_radius)
    print(f"\nCircle at (0,0) radius 5, Implicit Polynomial coefficients (3x3):\n{coeffs}")

    # 2. Distance Calculation (Differentiable)
    # Use a point (6,8) outside the radius 5 circle
    p = torch.tensor([6.0, 8.0], requires_grad=True)
    dist = geo.distance_point_to_circle(p, (0.0, 0.0, 5.0))
    print(f"\nDistance from (6,8) to circle boundary: {dist.item():.4f}")
    
    # Gradient of distance w.r.t point (Normal vector at point)
    dist.backward()
    print(f"Normal vector (gradient) at (6,8): {p.grad.tolist()}")


    # 3. Intersections
    c2_params = (4.0, 0.0, 3.0) # Circle at (4,0) radius 3
    intersections = geo.intersection_circles((0.0, 0.0, 5.0), c2_params)
    print(f"\nIntersections of Circle(0,0,5) and Circle(4,0,3):")
    if intersections:
        for i, pt in enumerate(intersections):
            print(f"  Point {i+1}: ({pt[0].item():.2f}, {pt[1].item():.2f})")

    # 4. Area
    print(f"\nArea of circle r=5: {geo.area_circle(5.0):.2f}")

if __name__ == "__main__":
    demo_geometry()
