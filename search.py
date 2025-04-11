from indexing import load_index
from ranking import VectorSpaceModel, BM25

class SearchEngine:
    def __init__(self, index_path="image_index.pkl"):
        self.index = load_index(index_path)
        if not self.index:
            raise ValueError("Index not found. Run indexing.py first")
        
        self.models = {
            "1": ("Vector Space Model", VectorSpaceModel(
                self.index['documents'],
                self.index['term_idf'],
                self.index['doc_vectors']
            )),
            "2": ("BM25", BM25(self.index['documents']))
        }

    def _remove_duplicates(self, results):
        """Filters duplicate images"""
        seen = set()
        unique = []
        for doc_id, score in results:
            url = self.index['documents'][doc_id]['image_url']
            if url not in seen:
                seen.add(url)
                unique.append((doc_id, score))
        return unique

    def search(self, query, model="1", top_k=10):
        """Executes search with deduplication"""
        try:
            model_name, model = self.models[model]
        except KeyError:
            available = "\n".join(f"{num}: {name}" for num, (name, _) in self.models.items())
            raise ValueError(f"Invalid model. Choose:\n{available}")
        
        # Extract filetype filter
        filetype = None
        if "filetype:" in query.lower():
            filetype = query.split("filetype:")[-1].strip().lower()
            query = query.split("filetype:")[0].strip()
        
        results = model.rank_documents(query)
        results = self._remove_duplicates(results)
        
        if filetype:
            results = [
                (doc_id, score) for doc_id, score in results
                if self.index['documents'][doc_id]['filetype'] == filetype
            ]
        
        return results[:top_k]

def display_results(results, index):
    """Formats search results"""
    if not results:
        print("No matching images found.")
        return
    
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = index['documents'][doc_id]
        print(f"\n#{rank} (Score: {score:.4f})")
        print(f"Title: {doc['title']}")
        if doc['text']:
            print(f"Context: {doc['text'][:100]}...")
        print(f"Type: {doc['filetype'].upper()} | URL: {doc['image_url']}")

if __name__ == "__main__":
    engine = SearchEngine()
    print("🌟 Image Search Engine 🌟")
    print("Available models:")
    print("\n".join(f"{num}: {name}" for num, (name, _) in engine.models.items()))
    
    while True:
        try:
            query = input("\nSearch query (or 'quit'): ").strip()
            if query.lower() == 'quit':
                break
                
            model = input("Choose model (default=1): ").strip() or "1"
            results = engine.search(query, model)
            display_results(results, engine.index)
            
        except Exception as e:
            print(f"Error: {e}")