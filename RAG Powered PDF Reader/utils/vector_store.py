import faiss
import numpy as np 

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunk_text = []

    def add(self, embeddings, chunk):
        vec = np.array(embeddings).astype('float32').reshape(1,-1)
        if self.index is None:
            self.index = faiss.IndexFlatL2(vec.shape[1])
        self.index.add(vec)
        self.chunk_text.append(chunk)

    def search(self, query_embeddings, top_k = 3):
        vec = np.array(query_embeddings).astype('float32').reshape(1,-1)
        distances, indices = self.index.search(vec, top_k)
        return [self.chunk_text[i] for i in indices[0]]
    
    
        