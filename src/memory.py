import random
import math
import numpy as np
from typing import List, Dict, Tuple

from .graph import TemporalPath


class ReplayMemory:
    """
    Adaptive replay memory structure for storing and querying states.
    Implements priority mechanism and storage mechanism.
    """
    
    def __init__(self, max_size: int = 1000, max_sample_num: int = 100, 
                 edge_limit: int = 4, similarity_alpha: float = 0.5):
        """
        Initialize the replay memory.
        
        Args:
            max_size: Maximum number of states to store
            max_sample_num: Maximum number of states to sample for query
            edge_limit: Minimum number of edges a state should have to be stored
            similarity_alpha: Weight for balancing attribute and trajectory similarities
        """
        self.memory = []  # List of state information tuples
        self.max_size = max_size
        self.max_sample_num = max_sample_num
        self.edge_limit = edge_limit
        self.similarity_alpha = similarity_alpha
        self.last_accessed = {}  # Map from state to its last access time
        self.access_counter = 0  # Counter for access time
        self.reward_threshold = 0.01  # Threshold for average reward
    
    def store(self, path: TemporalPath, attr_embedding: np.ndarray, 
              traj_embedding: List[np.ndarray], avg_reward: float,
              visit_count: int, priority: float) -> None:
        """
        Store a state in the replay memory.
        
        Args:
            path: The temporal path
            attr_embedding: Attribute embedding of the path
            traj_embedding: Trajectory embedding of the path
            avg_reward: Average reward of the path
            visit_count: Number of times the path has been visited
            priority: Priority value of the path
        """
        # Check if the path has enough edges
        edge_count = path.get_edge_count()
        if edge_count < self.edge_limit:
            return
            
        # Calculate storage probability based on edge count
        storage_prob = min(1.0, edge_count / (2.0 * self.edge_limit))
        
        # Randomly decide whether to store based on probability
        if random.random() > storage_prob:
            return
            
        # Check if the absolute reward is above threshold
        if abs(avg_reward) < self.reward_threshold:
            return
            
        # Store the state information
        state_info = (path, attr_embedding, traj_embedding, avg_reward, visit_count, priority)
        
        # If memory is full, replace the least recently accessed entry
        if len(self.memory) >= self.max_size:
            # Find the least recently accessed entry
            oldest_idx = min(range(len(self.memory)), 
                             key=lambda i: self.last_accessed.get(id(self.memory[i][0]), 0))
            self.memory[oldest_idx] = state_info
        else:
            self.memory.append(state_info)
            
        # Update last access time
        self.access_counter += 1
        self.last_accessed[id(path)] = self.access_counter
    
    def update(self, path: TemporalPath, avg_reward: float, 
               visit_count: int, curr_reward: float) -> None:
        """
        Update the information of a state in memory.
        
        Args:
            path: The temporal path
            avg_reward: New average reward
            visit_count: New visit count
            curr_reward: Current reward in the MDP
        """
        # Find the state in memory
        for i, (stored_path, attr_emb, traj_emb, old_reward, old_count, old_priority) in enumerate(self.memory):
            if self._paths_equal(path, stored_path):
                # Update priority based on reward stability
                new_priority = old_priority
                if old_count > 0:
                    reward_diff = abs(curr_reward - old_reward)
                    stability = max(0, 1 - reward_diff)
                    new_priority = min(1.0, old_priority + 0.1 * stability)
                
                # Update the entry
                self.memory[i] = (stored_path, attr_emb, traj_emb, avg_reward, visit_count, new_priority)
                
                # Update last access time
                self.access_counter += 1
                self.last_accessed[id(stored_path)] = self.access_counter
                return
    
    def query(self, path: TemporalPath, attr_embedding: np.ndarray, 
              traj_embedding: List[np.ndarray], constraints: Dict[str, float]) -> float:
        """
        Query the replay memory for the approximate reward of a state.
        
        Args:
            path: The temporal path
            attr_embedding: Attribute embedding of the path
            traj_embedding: Trajectory embedding of the path
            constraints: Dictionary mapping attribute names to upper bound values
            
        Returns:
            Approximate reward for the state
        """
        if not self.memory:
            return 0.0
            
        # Determine sample number based on path edge count
        edge_count = path.get_edge_count()
        sample_ratio = min(1.0, edge_count / (2.0 * self.edge_limit))
        sample_num = max(1, min(self.max_sample_num, int(sample_ratio * self.max_sample_num)))
        
        # Sample states based on priority
        if len(self.memory) <= sample_num:
            sampled = self.memory
        else:
            # Calculate sampling probabilities based on priority values
            priorities = []
            for _, _, _, _, _, priority in self.memory:
                action_count = len(path.get_actions())
                if action_count > 0:
                    priority_weight = priority * math.log(action_count)
                else:
                    priority_weight = priority
                priorities.append(priority_weight)
                
            # Normalize priorities
            total_priority = sum(priorities)
            if total_priority > 0:
                probs = [p / total_priority for p in priorities]
            else:
                probs = [1.0 / len(self.memory)] * len(self.memory)
                
            # Sample based on probabilities
            indices = random.choices(range(len(self.memory)), probs, k=sample_num)
            sampled = [self.memory[i] for i in indices]
        
        # Find top m similar states
        similarities = []
        for stored_path, stored_attr, stored_traj, reward, visit_count, _ in sampled:
            # Calculate similarity
            attr_similarity = self._cosine_similarity(attr_embedding, stored_attr)
            traj_similarity = self._dtw_similarity(traj_embedding, stored_traj)
            
            # Combine similarities
            similarity = self.similarity_alpha * attr_similarity + (1 - self.similarity_alpha) * traj_similarity
            
            # Add to the list
            similarities.append((similarity, reward, visit_count))
        
        # Sort by similarity (higher is better)
        similarities.sort(reverse=True)
        
        # Take top m similar states
        m = min(5, len(similarities))
        top_m = similarities[:m]
        
        # Calculate weighted average reward
        total_weight = 0.0
        weighted_sum = 0.0
        for similarity, reward, visit_count in top_m:
            # Weight is based on similarity and visit count
            weight = similarity * math.log(1 + visit_count)
            total_weight += weight
            weighted_sum += weight * reward
        
        # Return the approximate reward
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        return dot_product / (norm1 * norm2)
    
    def _dtw_similarity(self, traj1: List[np.ndarray], traj2: List[np.ndarray]) -> float:
        """
        Calculate similarity between two trajectories using Dynamic Time Warping.
        
        This is a simplified version of DTW for demonstration.
        """
        if not traj1 or not traj2:
            return 0.0
            
        # Use Fast-DTW algorithm (simplified for demonstration)
        # In a real implementation, you would use a proper Fast-DTW library
        
        # Create a distance matrix
        n = len(traj1)
        m = len(traj2)
        dtw_matrix = np.zeros((n + 1, m + 1))
        
        # Initialize the matrix with infinity
        for i in range(n + 1):
            for j in range(m + 1):
                dtw_matrix[i, j] = float('inf')
        
        dtw_matrix[0, 0] = 0
        
        # Fill the matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Calculate Euclidean distance between embeddings
                cost = np.linalg.norm(traj1[i-1] - traj2[j-1])
                
                # Update matrix
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # Insertion
                    dtw_matrix[i, j-1],      # Deletion
                    dtw_matrix[i-1, j-1]     # Match
                )
        
        # Convert DTW distance to similarity (inversely proportional)
        dtw_distance = dtw_matrix[n, m]
        similarity = 1.0 / (1.0 + dtw_distance)
        
        return similarity
    
    def _paths_equal(self, path1: TemporalPath, path2: TemporalPath) -> bool:
        """Check if two paths are equal based on their edges."""
        if path1.get_edge_count() != path2.get_edge_count():
            return False
            
        return path1.edges == path2.edges