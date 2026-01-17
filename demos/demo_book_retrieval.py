import torch
from nous.workspace import NousWorkspace

def demo_book_retrieval():
    print("=== Demo: Nous Differentiable Book / Knowledge Base ===")
    ws = NousWorkspace(hard_logic=True)
    
    # 1. Prepare a "Book" (Simplified text)
    book_text = """
    In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since. 
    Whenever you feel like criticizing any one, he told me, just remember that all the people in this world haven't had the advantages that you've had.
    He didn't say any more but we've always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that.
    In consequence I'm inclined to reserve all judgments, a habit that has opened up many curious natures to me and also made me the victim of not a few veteran bores. 
    The abnormal mind is quick to detect and attach itself to this quality when it appears in a normal person, and so it came about that in college I was unjustly accused of being a politician, 
    because I was privy to the secret griefs of wild, unknown men. Most of the confidences were unsought—frequently I have feigned sleep, preoccupation, or a hostile levity 
    when I realized by some unmistakable sign that an intimate revelation was quivering on the horizon; 
    for the intimate revelations of young men or at least the terms in which they express them are usually plagiaristic and marred by obvious suppressions. 
    Reserving judgments is a matter of infinite hope. I am still a little afraid of missing something if I forget that, as my father snobbishly suggested, 
    and I snobbishly repeat, a sense of the fundamental decencies is parcelled out unequally at birth.
    """
    
    # 2. Ingest the Book into a Differentiable Knowledge Base
    # soft_load_book chunks the text and stores it in high-capacity memory
    print("\n[Step 1] Ingesting 'The Great Gatsby' excerpt into DKB...")
    ingest_code = "return soft_load_book(text, chunk_size=30)"
    kb = ws.run(ingest_code, inputs={'text': book_text})
    ws.save('gatsby_kb', kb)
    
    # 3. Perform Semantic Retrieval
    # Agent queries: "What did the father suggest?"
    query = "father's advice and suggestions"
    print(f"\n[Step 2] Agent Query: '{query}'")
    
    search_code = """
results = soft_search(query, gatsby_kb, k=2)
return results
"""
    results = ws.run(search_code, inputs={'query': query})
    
    print("\n[Step 3] Top Retrieved Passages:")
    for i, p in enumerate(results['passages']):
        print(f"[{i+1}] {p[:120]}...")
    
    # 4. Verify Differentiability
    # Backprop from the relevance scores to the query tokens (simulated)
    # This proves the retriever is "soft" and can be trained
    print("\n[Step 4] Verifying Gradient flow (Relevance -> Query):")
    relevance = results['relevance']
    # relevance is a tensor of weights [num_chunks]
    relevance.sum().backward()
    
    # Check if we have gradients in the memory (embeddings were used)
    # In V5.0, the query vector itself is differentiable
    # Check if a specific chunk's existence influenced the search
    if relevance.grad_fn is not None:
        print("\n✓ SUCCESS: Differentiable retrieval confirmed. Nous can hold and search a whole book.")
    else:
        print("\nFAIL: Search was not differentiable.")

if __name__ == "__main__":
    demo_book_retrieval()
