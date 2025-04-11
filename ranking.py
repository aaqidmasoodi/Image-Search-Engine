import math
from collections import defaultdict
from preprocess import preprocess_text

class VectorSpaceModel:
    def __init__(self, documents, term_idf, doc_vectors):
        self.documents = documents
        self.term_idf = term_idf
        self.doc_vectors = doc_vectors
        self.content_boost = {'diagram':1.5, 'photo':1.3, 'screenshot':1.3}

    def _compute_query_vector(self, query_terms):
        """Creates enhanced query vector"""
        tf = defaultdict(float)
        for term in query_terms:
            tf[term] += self.content_boost.get(term, 1.0)
        
        query_vector = {}
        total = 0
        for term, freq in tf.items():
            if term in self.term_idf:
                weight = (1 + math.log(1 + freq)) * self.term_idf[term]
                query_vector[term] = weight
                total += weight ** 2
        
        if total > 0:
            norm = math.sqrt(total)
            for term in query_vector:
                query_vector[term] /= norm
        
        return query_vector

    def rank_documents(self, query):
        """Ranks documents by cosine similarity"""
        query_terms = preprocess_text(query)
        if not query_terms:
            return []
            
        query_vector = self._compute_query_vector(query_terms)
        results = []
        
        for doc_id, doc_vector in self.doc_vectors.items():
            score = sum(
                query_vector.get(term, 0) * weight
                for term, weight in doc_vector.items()
            )
            if score > 0.01:  # Relevance threshold
                results.append((doc_id, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._precompute_stats()

    def _precompute_stats(self):
        """Precomputes document statistics"""
        self.doc_lengths = {}
        self.inverted_index = defaultdict(dict)
        
        for doc_id, doc in self.documents.items():
            terms = preprocess_text(f"{doc['title']} {doc['text']}")
            self.doc_lengths[doc_id] = len(terms)
            for term in set(terms):
                self.inverted_index[term][doc_id] = terms.count(term)
        
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.documents)
        self.doc_freqs = {term: len(docs) for term, docs in self.inverted_index.items()}

    def rank_documents(self, query):
        """BM25 ranking algorithm"""
        query_terms = preprocess_text(query)
        if not query_terms:
            return []
            
        scores = defaultdict(float)
        for term in query_terms:
            if term not in self.inverted_index:
                continue
                
            idf = math.log((len(self.documents) - self.doc_freqs[term] + 0.5) / 
                  (self.doc_freqs[term] + 0.5) + 1)
            
            for doc_id, tf in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                scores[doc_id] += idf * (numerator / denominator)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)