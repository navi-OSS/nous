import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class GemmaProgrammer(nn.Module):
    def __init__(self, model_name="google/gemma-3-270m-it", 
                 steps=3, num_ops=2, num_regs=5):
        super().__init__()
        self.steps = steps
        
        print(f"Loading LLM: {model_name}")
        try:
            self.llm = AutoModel.from_pretrained(model_name)
            hidden_size = self.llm.config.hidden_size
        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}")
            print("Fallback to random projection for testing or tiny config")
            config = AutoConfig.from_pretrained('gpt2') # Fallback config
            config.vocab_size = 1000
            config.n_layer = 2
            config.n_head = 2
            config.n_embd = 64
            self.llm = AutoModel.from_config(config)
            hidden_size = 64
        
        # Projectors for Register Machine
        # We need (Steps * (Num_Ops + 2*Num_Regs)) params.
        
        # Strategy:
        # Prompt: "Solve x^2 + 2x"
        # Output: Last hidden state -> MLP -> [Steps, Features]
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, steps * (num_ops + 2*num_regs))
        )
        
        self.num_ops = num_ops
        self.num_regs = num_regs
        
    def forward(self, input_ids, attention_mask=None, temp=1.0):
        # 1. Run LLM
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Get representation (Last token? CLS? Average?)
        # For simplicity: Average pooling or Last Token.
        # Gemma is causal, so Last Token represents sequence.
        
        last_hidden = outputs.last_hidden_state[:, -1, :] # [Batch, Hidden]
        
        # 3. Project to Control Signals
        flat_params = self.proj(last_hidden) # [Batch, Steps * (...)]
        
        # Reshape
        # [Batch, Steps, Ops+Arg1+Arg2]
        batch_size = input_ids.shape[0]
        control = flat_params.view(batch_size, self.steps, -1)
        
        # Split into Ops/Arg1/Arg2
        # Sizes: num_ops, num_regs, num_regs
        op_logits = control[:, :, :self.num_ops]
        arg1_logits = control[:, :, self.num_ops : self.num_ops+self.num_regs]
        arg2_logits = control[:, :, self.num_ops+self.num_regs :]
        
        # Softmax
        ops = torch.softmax(op_logits / temp, dim=-1)
        arg1 = torch.softmax(arg1_logits / temp, dim=-1)
        arg2 = torch.softmax(arg2_logits / temp, dim=-1)
        
        return ops, arg1, arg2
