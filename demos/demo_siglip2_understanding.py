import torch
import torch.nn.functional as F
from nous.workspace import NousWorkspace
from torchvision.utils import save_image
import PIL.Image as Image
import numpy as np

def create_red_circle_image(size=224):
    """Creates a simple red circle on a white background."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    center = size // 2
    radius = size // 4
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    img[mask] = [255, 0, 0] # Red
    return Image.fromarray(img)

def demo_siglip2_understanding():
    print("=== Nous V7.1: SigLIP 2 Multimodal Understanding ===")
    
    ws = NousWorkspace(hard_logic=False)
    
    # 1. THE DATA: A Red Circle
    print("Generating test image (Red Circle)...")
    img = create_red_circle_image()
    img.save("test_circle.png")
    
    # 2. THE REASONING: Zero-Shot Classification
    # We will ask the engine to tell us what it sees.
    prompts = ["a red circle", "a blue square", "a green triangle", "a photo of a cat"]
    
    print("\n[Phase 1] Embedding text prompts...")
    # Wrap in a loop in the interpreter or just run individual
    scores = {}
    ws.save('img_data', img)
    
    understanding_code = """
mem, meta = soft_load_image(img_data)
# Pool patches for global understanding
weights = torch.ones(mem.num_slots) / float(mem.num_slots)
img_vec = mem.read(weights)

results = []
for p in prompts:
    t_vec = soft_embed_text(p)
    # Cosine Similarity
    score = soft_dot(img_vec, t_vec) / (soft_norm(img_vec) * soft_norm(t_vec) + 1e-6)
    results.append(score)
return results
"""
    
    ws.save('prompts', prompts)
    results = ws.run(understanding_code)
    
    # Handle symbolic
    def to_val(node):
        if hasattr(node, "to_taylor"):
            return node.to_taylor(0.0, 1, ws.model.hilbert)[0]
        return node

    print("\n[Phase 2] Semantic Match Results:")
    for i, p in enumerate(prompts):
        score = to_val(results[i])
        print(f"  Prompt: '{p:18}' | Confidence: {score.item():.4f}")

    best_idx = torch.argmax(torch.tensor([to_val(r).item() for r in results])).item()
    print(f"\n✓ WINNER: '{prompts[best_idx]}'")
    
    if prompts[best_idx] == "a red circle":
        print("Success: SigLIP 2 correctly identified the visual concept!")
    else:
        print("Failure: The model did not correctly identify the circle.")

if __name__ == "__main__":
    demo_siglip2_understanding()
