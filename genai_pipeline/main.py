from scripts.extract_text import extract_text
from scripts.chunk_text import chunk_text
from scripts.embed_text import embed_chunks
from scripts.index_embeddings import index_embeddings
import numpy as np

def main():
    # Step 1: Data Extraction
    file_path = 'data/sample_text.txt'
    text = extract_text(file_path)
    
    # Step 2: Chunking
    chunks = chunk_text(text)
    
    # Step 3: Embedding
    embeddings = embed_chunks(chunks)
    
    # Step 4: Indexing
    embeddings_np = np.array(embeddings)
    index = index_embeddings(embeddings_np)
    
    # Example Query
    query_embedding = embeddings_np[0]  # Using the first embedding as a query example
    D, I = index.search(np.array([query_embedding]), k=5)
    print(f"Top 5 similar chunks indices: {I}")

if __name__ == "__main__":
    main()