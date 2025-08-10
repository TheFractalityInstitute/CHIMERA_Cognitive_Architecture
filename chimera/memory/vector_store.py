"""
Vector storage and similarity search for thoughts and concepts
Optimized for semantic search and pattern matching
"""

import numpy as np
from typing import List, Tuple, Optional
import faiss  # If available, fallback to numpy
import pickle

class VectorStore:
    """
    Efficient vector storage with similarity search
    Scales from numpy to FAISS for large datasets
    """
    
    def __init__(self, dimension: int = 512, use_faiss: bool = False):
        self.dimension = dimension
        self.use_faiss = use_faiss and self._check_faiss()
        
        if self.use_faiss:
            # FAISS index for efficient similarity search
            self.index = faiss.IndexFlatL2(dimension)
            self.id_map = {}  # Map from index position to ID
        else:
            # Fallback to numpy
            self.vectors = {}
            self.vector_array = None
            self.id_list = []
            
    def _check_faiss(self) -> bool:
        """Check if FAISS is available"""
        try:
            import faiss
            return True
        except ImportError:
            return False
            
    def add_vector(self, id: str, vector: np.ndarray):
        """Add a vector to the store"""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}")
            
        if self.use_faiss:
            idx = self.index.ntotal
            self.index.add(vector.reshape(1, -1).astype('float32'))
            self.id_map[idx] = id
        else:
            self.vectors[id] = vector
            self._rebuild_array()
            
    def search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar vectors"""
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query dimension mismatch")
            
        if self.use_faiss:
            distances, indices = self.index.search(
                query_vector.reshape(1, -1).astype('float32'), k
            )
            results = []
            for i, idx in enumerate(indices[0]):
                if idx in self.id_map:
                    results.append((self.id_map[idx], float(distances[0][i])))
            return results
        else:
            if not self.vectors:
                return []
                
            # Compute distances
            distances = []
            for id, vector in self.vectors.items():
                dist = np.linalg.norm(query_vector - vector)
                distances.append((id, dist))
                
            # Sort and return top k
            distances.sort(key=lambda x: x[1])
            return distances[:k]
            
    def _rebuild_array(self):
        """Rebuild numpy array for efficient computation"""
        if not self.vectors:
            return
            
        self.id_list = list(self.vectors.keys())
        self.vector_array = np.stack([self.vectors[id] for id in self.id_list])
        
    def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Get vector by ID"""
        if self.use_faiss:
            # Need to maintain separate storage for retrieval
            # In production, would use a separate key-value store
            pass
        else:
            return self.vectors.get(id)
            
    def remove_vector(self, id: str):
        """Remove a vector from the store"""
        if self.use_faiss:
            # FAISS doesn't support removal, would need to rebuild
            pass
        else:
            if id in self.vectors:
                del self.vectors[id]
                self._rebuild_array()
                
    def save(self, path: str):
        """Save vector store to disk"""
        with open(path, 'wb') as f:
            if self.use_faiss:
                pickle.dump({
                    'index': faiss.serialize_index(self.index),
                    'id_map': self.id_map,
                    'dimension': self.dimension
                }, f)
            else:
                pickle.dump({
                    'vectors': self.vectors,
                    'dimension': self.dimension
                }, f)
                
    def load(self, path: str):
        """Load vector store from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.dimension = data['dimension']
            
            if self.use_faiss:
                self.index = faiss.deserialize_index(data['index'])
                self.id_map = data['id_map']
            else:
                self.vectors = data['vectors']
                self._rebuild_array()