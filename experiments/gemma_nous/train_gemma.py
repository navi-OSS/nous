import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Add root to path to import nous modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from interpreter import NeuralInterpreter
from symbolic import ExprVar
from engine import NousModel
from model import GemmaProgrammer

def train_gemma():
    print("=== Gemma-Nous Experiment ===")
    print("Goal: Train Gemma to synthesize 'f(x) = x^2 + 2x'")
    
    # 1. Setup Model
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    
    # Fallback to a small model if Gemma is not accessible
    model_name = "google/gemma-3-270m-it" 
    # Use dummy if not found handled in model.py, but tokenizer?
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        print("Fallback Tokenizer (GPT2)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
    programmer = GemmaProgrammer(model_name=model_name).to(device)
    
    # Freeze LLM? For 270M we can fine-tune.
    # But let's freeze first for speed/stability and just train projection?
    # Or LoRA.
    # Let's train full because it's tiny (or random projection if dummy).
    optimizer = torch.optim.Adam(programmer.parameters(), lr=1e-4)
    
    # 2. Setup Nous Environment
    nous_model = NousModel()
    interpreter = NeuralInterpreter(nous_model)
    
    cpu_code = """
# Setup Memory
mem = [x, 0.0, 0.0, 0.0, 0.0]

for i in range(3):
    op_probs = ops[i]
    a1_probs = arg1[i]
    a2_probs = arg2[i]
    
    val1 = soft_index(a1_probs, mem)
    val2 = soft_index(a2_probs, mem)
    
    res_add = val1 + val2
    res_mul = val1 * val2
    
    candidates = [res_add, res_mul]
    result = soft_switch(op_probs, candidates)
    
    target_idx = 2 + i
    mem[target_idx] = result

return mem[4]
"""

    print("\nTraining...")
    prompt = "Write a program to calculate x^2 + 2x using registers."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for epoch in range(20):
        optimizer.zero_grad()
        
        # Annealing
        t = max(0.05, 2.0 - (epoch / 15.0) * 1.95)
        
        # 1. Gemma Forward
        ops_w, arg1_w, arg2_w = programmer(inputs.input_ids, inputs.attention_mask, temp=t)
        
        # Ops are [1, 3, 2]. We squeeze batch dim since it's 1 prompt.
        ops_step = [ops_w[0, i] for i in range(3)]
        arg1_step = [arg1_w[0, i] for i in range(3)]
        arg2_step = [arg2_w[0, i] for i in range(3)]
        
        # 2. Data Batch
        x_in = torch.randn(20, requires_grad=True).to(device)
        z_target = x_in**2 + 2*x_in
        
        exec_inputs = {
            'x': x_in,
            'ops': ops_step,
            'arg1': arg1_step,
            'arg2': arg2_step
        }
        
        res_graph = interpreter.execute(cpu_code, exec_inputs)
        
        loss = torch.mean((res_graph - z_target)**2)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch} (Temp {t:.2f}): Loss {loss.item():.4f}")

    # Decode
    print("\nLearned Algorithm via Gemma:")
    ops_w, arg1_w, arg2_w = programmer(inputs.input_ids, inputs.attention_mask, temp=0.01)
    
    op_names = ["Add", "Mul"]
    mem_names = ["x", "0", "r2", "r3", "r4"]
    
    for i in range(3):
        op_idx = torch.argmax(ops_w[0, i]).item()
        a1_idx = torch.argmax(arg1_w[0, i]).item()
        a2_idx = torch.argmax(arg2_w[0, i]).item()
        
        target = mem_names[2+i]
        op = op_names[op_idx]
        src1 = mem_names[a1_idx]
        src2 = mem_names[a2_idx]
        
        print(f"Step {i}: {target} = {op}({src1}, {src2})")

if __name__ == "__main__":
    train_gemma()
