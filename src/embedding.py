import numpy as np
import random
from typing import List, Tuple

from .graph import AttributedDynamicGraph


class EdgeEmbedding:
    """
    Utility class for embedding edges in a low-dimensional space.
    Simulates DeepWalk for edges in a dynamic graph.
    """
    
    def __init__(self, graph: AttributedDynamicGraph, embedding_dim: int = 8):
        """
        Initialize the edge embedding.
        
        Args:
            graph: The attributed dynamic graph
            embedding_dim: Dimensionality of the embedding vectors
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.edge_embeddings = {}
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize random embeddings for all edges."""
        for edge in self.graph.edges:
            self.edge_embeddings[edge] = np.random.randn(self.embedding_dim)
    
    def _random_walk(self, start_edge: Tuple, walk_length: int = 10) -> List[Tuple]:
        """
        Perform a random walk starting from an edge.
        
        Args:
            start_edge: The starting edge for the walk
            walk_length: Length of the random walk
            
        Returns:
            List of edges in the walk
        """
        walk = [start_edge]
        current_edge = start_edge
        
        for _ in range(walk_length - 1):
            # Get the destination node of the current edge
            dest_node = current_edge[1]
            
            # Get all outgoing edges from the destination node after arrival
            next_edges = self.graph.get_outgoing_edges(dest_node, current_edge[4])
            
            if not next_edges:
                break
                
            # Select the next edge with probability inversely proportional to its length
            weights = [1.0 / edge[2] for edge in next_edges]
            total_weight = sum(weights)
            
            if total_weight == 0:
                break
                
            # Normalize weights
            probs = [w / total_weight for w in weights]
            
            # Sample the next edge
            next_edge = random.choices(next_edges, probs)[0]
            walk.append(next_edge)
            current_edge = next_edge
        
        return walk
    
    def train(self, num_walks: int = 10, walk_length: int = 10) -> None:
        """
        Train the edge embeddings using random walks.
        
        Args:
            num_walks: Number of random walks per edge
            walk_length: Length of each random walk
        """
        # This is a simplified version of DeepWalk
        # In a real implementation, you would use the Skip-gram model
        
        all_walks = []
        
        # Generate random walks
        for _ in range(num_walks):
            for edge in self.graph.edges:
                walk = self._random_walk(edge, walk_length)
                all_walks.append(walk)
        
        # Update embeddings based on co-occurrence in walks
        # Here we use a simplified approach for demonstration
        for walk in all_walks:
            for i, edge1 in enumerate(walk):
                for j, edge2 in enumerate(walk):
                    if i != j:
                        # Update embedding based on co-occurrence
                        # The closer edges are in the walk, the more they influence each other
                        weight = 1.0 / (abs(i - j) + 1)
                        self.edge_embeddings[edge1] += weight * 0.01 * self.edge_embeddings[edge2]
                        self.edge_embeddings[edge2] += weight * 0.01 * self.edge_embeddings[edge1]
        
        # Normalize embeddings
        for edge in self.graph.edges:
            norm = np.linalg.norm(self.edge_embeddings[edge])
            if norm > 0:
                self.edge_embeddings[edge] /= norm
    
    def get_embedding(self, edge: Tuple) -> np.ndarray:
        """Get the embedding vector for an edge."""
        return self.edge_embeddings.get(edge, np.zeros(self.embedding_dim))