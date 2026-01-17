"""
Nous V8.3: RAG Agent Demo

Demonstrates an autonomous agent that has "Internet Access" (TextMemory)
to answer questions by querying its own knowledge base from within the DSL.
"""
from nous.workspace import NousWorkspace
from nous.memory import TextMemory
from nous.engine import NousModel
from nous.interpreter import NeuralInterpreter
import torch

def demo():
    print("=== Neural RAG Agent ===")
    
    # 1. Setup Brain (TextMemory)
    brain = TextMemory(embedding_dim=4)
    # Seed knowledge
    brain.add("The capital of France is Paris.", torch.tensor([1.0, 0.0, 0.0, 0.0]))
    brain.add("The capital of Japan is Tokyo.", torch.tensor([0.0, 1.0, 0.0, 0.0]))
    brain.add("Water boils at 100 degrees Celsius.", torch.tensor([0.0, 0.0, 1.0, 0.0]))
    
    # 2. Setup Workspace with Brain
    # Note: Workspace abstracts model/interpreter, we need to inject memory
    model = NousModel()
    interpreter = NeuralInterpreter(model, text_memory=brain)
    
    import textwrap
    code = textwrap.dedent("""
    # 1. Agent formulates a query vector (simulated thought)
    # "I want to know about France" -> [1.0, 0.1, 0.0, 0.0]
    thought_vector = torch.tensor([1.0, 0.1, 0.0, 0.0])
    
    # 2. Agent performs RAG lookup
    # memory_retrieve is now a native persistent tool
    results = memory_retrieve(thought_vector, k=1)
    
    # 3. Agent processes result
    fact = results[0][0] # Get text string
    score = results[0][1]
    
    return [fact, score]
    """)
    
    print("\n[Agent] Thinking...")
    result = interpreter.execute(code, {})
    
    print(f"[Agent] Retrieved Fact: '{result[0]}'")
    score = result[1].value if hasattr(result[1], 'value') else result[1]
    print(f"[Agent] Confidence: {score:.4f}")
    
    assert "Paris" in result[0]
    print("\nSUCCESS: Agent autonomously retrieved external knowledge.")

if __name__ == "__main__":
    demo()
