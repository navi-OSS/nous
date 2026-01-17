"""
Nous V8.3: Infinite Text Memory Demo

Demonstrates the use of `TextMemory` to bridge the gap between 
neural embeddings and symbolic text strings.
"""
from nous.memory import TextMemory
import torch

def demo():
    # 1. Initialize Memory
    # Using 4D embeddings for simplicity
    mem = TextMemory(embedding_dim=4)
    
    print("=== 1. Storing Knowledge ===")
    
    knowledge_base = [
        ("Paris is the capital of France.", [1.0, 0.0, 0.0, 0.0]),
        ("Berlin is the capital of Germany.", [0.9, 0.1, 0.0, 0.0]),
        ("Python is a programming language.", [0.0, 0.0, 1.0, 0.0]),
        ("Rust is a systems language.", [0.0, 0.0, 0.9, 0.1]),
        ("The sky is blue.", [0.0, 1.0, 0.0, 0.0]),
    ]
    
    for text, vec_list in knowledge_base:
        vec = torch.tensor(vec_list, dtype=torch.float32)
        mem.add(text, vec)
        print(f"Stored: '{text}'")
        
    print(f"\nTotal Memory Items: {len(mem)}")
    
    # 2. Retrieval
    print("\n=== 2. Soft Retrieval ===")
    
    # Query: Something like "France"
    query_france = torch.tensor([1.0, 0.05, 0.0, 0.0])
    results = mem.retrieve(query_france, k=2)
    
    print(f"Query: [1.0, 0.05, ...] (Expected: France/Germany)")
    for text, score in results:
        print(f"  Result: '{text}' (Score: {score:.4f})")
        
    # Query: Something like "Programming"
    query_prog = torch.tensor([0.0, 0.00, 1.0, 0.01])
    results = mem.retrieve(query_prog, k=2)
    
    print(f"\nQuery: [0.0, ..., 1.0, 0.01] (Expected: Python/Rust)")
    for text, score in results:
        print(f"  Result: '{text}' (Score: {score:.4f})")
        
    # Query: Unlimited context test
    print("\n=== 3. Scale Test (Adding 1000 items) ===")
    for i in range(1000):
        mem.add(f"Junk item {i}", torch.randn(4))
        
    print(f"Memory Size: {len(mem)} items")
    results = mem.retrieve(query_france, k=1)
    print(f"Retrieval checks {len(mem)} items...")
    print(f"  Top Result: '{results[0][0]}'")

if __name__ == "__main__":
    demo()
