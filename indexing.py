import os
import pickle
import time
import math
from collections import defaultdict
from preprocess import preprocess_text
from parser import parse_image_metadata

def build_inverted_index(documents):
    """Builds optimized index for image search"""
    print("\nBuilding search index...")
    start_time = time.time()
    
    # Data structures
    inverted_index = defaultdict(dict)
    doc_vectors = {}
    doc_lengths = {}
    term_idf = {}
    filetype_index = defaultdict(set)
    
    # First pass: Document frequencies
    print("Calculating document frequencies...")
    doc_freqs = defaultdict(int)
    for doc_id, doc in documents.items():
        # Weighted term extraction (5x title, 3x text)
        title_terms = preprocess_text(doc["title"]) * 5
        text_terms = preprocess_text(doc["text"]) * 1
        
        all_terms = title_terms + text_terms
        all_terms.append(f"filetype:{doc['filetype']}")
        
        # Update frequencies
        for term in set(all_terms):
            doc_freqs[term] += 1
        
        filetype_index[doc['filetype']].add(doc_id)
    
    # Compute IDF with smoothing
    print("Computing IDF weights...")
    N = len(documents)
    term_idf = {
        term: math.log((N - df + 0.5) / (df + 0.5) + 1)
        for term, df in doc_freqs.items()
    }
    
    # Second pass: Document vectors
    print("Creating document vectors...")
    for doc_id, doc in documents.items():
        # Recompute terms with same weights
        title_terms = preprocess_text(doc["title"]) * 5
        text_terms = preprocess_text(doc["text"]) * 1
        all_terms = title_terms + text_terms
        
        # TF with sublinear scaling
        tf = {
            term: 1 + math.log(1 + all_terms.count(term))
            for term in set(all_terms)
        }
        
        # Create TF-IDF vector
        vector = {
            term: tf_val * term_idf[term]
            for term, tf_val in tf.items()
            if term in term_idf
        }
        
        # Normalize
        norm = math.sqrt(sum(w**2 for w in vector.values())) or 1
        doc_vectors[doc_id] = {term: weight/norm for term, weight in vector.items()}
        doc_lengths[doc_id] = len(all_terms)
        
        # Update inverted index
        for term in vector:
            inverted_index[term][doc_id] = vector[term]
    
    # Print summary
    print(f"\nIndex built in {time.time()-start_time:.2f}s")
    print(f"• Documents: {N}")
    print(f"• Unique terms: {len(inverted_index)}")
    print(f"• Filetypes: {list(filetype_index.keys())}")  # Changed this line
    
    return {
        'inverted_index': inverted_index,
        'doc_vectors': doc_vectors,
        'term_idf': term_idf,
        'documents': documents,
        'stats': {
            'num_documents': N,
            'num_terms': len(inverted_index),
            'filetypes': {ft: len(docs) for ft, docs in filetype_index.items()}  # Added count of each type
        }
    }

def save_index(index_data, path="image_index.pkl"):
    """Saves index with compression"""
    with open(path, 'wb') as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nIndex saved to {os.path.abspath(path)}")

def load_index(path="image_index.pkl"):
    """Loads saved index"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

if __name__ == "__main__":
    # Step 1: Parse metadata
    print("Loading image metadata...")
    documents = parse_image_metadata("data/image_metadata.json")
    
    if not documents:
        print("Error: No documents parsed. Check your image_metadata.json")
        exit(1)
    
    # Step 2: Build index
    index_data = build_inverted_index(documents)
    
    # Step 3: Save index
    save_index(index_data)
    
    # Verify loading
    print("\nTesting index loading...")
    if load_index():
        print("Index loaded successfully!")
    else:
        print("Error loading index")