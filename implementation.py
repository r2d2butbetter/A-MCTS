import numpy as np
import random
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class AttributedDynamicGraph:
    """
    Represents an Attributed Dynamic Graph (ADG) where edges have temporal information
    and multiple attributes.
    """
    
    def __init__(self):
        self.nodes = set()  # Set of nodes in the graph
        self.edges = []  # List of edges (quintuple: u, v, length, dep_time, arr_time)
        self.attributes = {}  # Dictionary mapping edge to attribute values
        self.edge_index = defaultdict(list)  # Index for fast edge lookup from source node
        self.max_time = 0  # Maximum arrival time in the graph
        
    def add_node(self, node_id: int) -> None:
        """Add a node to the graph."""
        self.nodes.add(node_id)
    
    def add_edge(self, u: int, v: int, length: float, dep_time: int, arr_time: int, 
                 attributes: Dict[str, float]) -> None:
        """
        Add an edge to the graph with temporal information and attributes.
        
        Args:
            u: Source node
            v: Destination node
            length: Length of the edge
            dep_time: Departure time from u
            arr_time: Arrival time at v
            attributes: Dictionary of attribute names to values
        """
        self.nodes.add(u)
        self.nodes.add(v)
        
        edge = (u, v, length, dep_time, arr_time)
        self.edges.append(edge)
        self.attributes[edge] = attributes
        self.edge_index[u].append(edge)
        
        self.max_time = max(self.max_time, arr_time)
    
    def get_outgoing_edges(self, node: int, time: int) -> List[Tuple]:
        """
        Get all outgoing edges from a node after a specific time.
        
        Args:
            node: Source node
            time: Current time (only edges with departure time >= time will be returned)
            
        Returns:
            List of edges (quintuples) starting from the node
        """
        return [edge for edge in self.edge_index[node] if edge[3] >= time]
    
    def get_edge_attribute(self, edge: Tuple, attribute: str) -> float:
        """Get the value of a specific attribute for an edge."""
        return self.attributes[edge].get(attribute, 0.0)


class TemporalPath:
    """
    Represents a temporal path in an attributed dynamic graph.
    A temporal path is a sequence of edges that follow temporal constraints.
    """
    
    def __init__(self, graph: AttributedDynamicGraph):
        self._graph = graph
        self._edges = []
        self._source = None
        self._current_node = None
        self._arrival_time = 0
        
    @property
    def edges(self) -> List[Tuple]:
        """Get the list of edges in the path."""
        return self._edges
    
    @property
    def source(self) -> int:
        """Get the source node of the path."""
        return self._source
    
    @source.setter
    def source(self, node: int) -> None:
        """Set the source node of the path."""
        self._source = node
        self._current_node = node
    
    @property
    def current_node(self) -> int:
        """Get the current (last) node in the path."""
        return self._current_node
    
    @property
    def destination(self) -> Optional[int]:
        """Get the destination node if the path has edges."""
        if not self._edges:
            return None
        return self._edges[-1][1]
    
    @property
    def length(self) -> float:
        """Calculate the total length of the path."""
        return sum(edge[2] for edge in self._edges)
    
    @property
    def arrival_time(self) -> int:
        """Get the arrival time at the last node."""
        return self._arrival_time
    
    def add_edge(self, edge: Tuple) -> None:
        """
        Add an edge to the path.
        
        Args:
            edge: A quintuple (u, v, length, dep_time, arr_time)
        """
        # Ensure the edge starts from the current node
        if edge[0] != self._current_node:
            raise ValueError(f"Edge {edge} does not start from current node {self._current_node}")
        
        # Ensure temporal consistency
        if self._edges and edge[3] < self._arrival_time:
            raise ValueError(f"Edge {edge} departs before arrival at previous node")
        
        self._edges.append(edge)
        self._current_node = edge[1]  # Update current node to the destination
        self._arrival_time = edge[4]  # Update arrival time
    
    def calculate_attribute_value(self, attribute: str) -> float:
        """Calculate the aggregated value of an attribute along the path."""
        return sum(self._graph.get_edge_attribute(edge, attribute) for edge in self._edges)
    
    def is_feasible(self, constraints: Dict[str, float], time_interval: Tuple[int, int]) -> bool:
        """
        Check if the path satisfies all constraints.
        
        Args:
            constraints: Dictionary mapping attribute names to upper bound values
            time_interval: Tuple (min_time, max_time) for departure and arrival
            
        Returns:
            True if the path satisfies all constraints, False otherwise
        """
        # Check time constraints
        if not self._edges:
            return False
        
        min_time, max_time = time_interval
        if self._edges[0][3] < min_time or self._arrival_time > max_time:
            return False
        
        # Check attribute constraints
        for attr, upper_bound in constraints.items():
            if self.calculate_attribute_value(attr) > upper_bound:
                return False
                
        return True
    
    def copy(self) -> 'TemporalPath':
        """Create a deep copy of the path."""
        new_path = TemporalPath(self._graph)
        new_path._source = self._source
        new_path._current_node = self._current_node
        new_path._arrival_time = self._arrival_time
        new_path._edges = self._edges.copy()
        return new_path
    
    def get_actions(self) -> List[Tuple]:
        """
        Get all valid actions (outgoing edges) from the current node after arrival.
        
        Returns:
            List of valid edges (actions) that can be taken from the current state
        """
        return self._graph.get_outgoing_edges(self._current_node, self._arrival_time)
    
    def get_edge_count(self) -> int:
        """Get the number of edges in the path."""
        return len(self._edges)
    
    def get_attribute_embedding(self, constraints: Dict[str, float]) -> np.ndarray:
        """
        Get the attribute embedding of the path.
        
        Args:
            constraints: Dictionary mapping attribute names to upper bound values
            
        Returns:
            Numpy array with normalized attribute values
        """
        embedding = []
        for attr, upper_bound in constraints.items():
            value = self.calculate_attribute_value(attr)
            # Normalize by the constraint value
            if upper_bound > 0:
                normalized_value = value / upper_bound
            else:
                normalized_value = value
            embedding.append(normalized_value)
        
        return np.array(embedding)
    
    def __str__(self) -> str:
        """String representation of the path."""
        if not self._edges:
            return f"Empty path from node {self._source}"
        
        path_str = f"Path from {self._source} to {self.destination}: "
        for edge in self._edges:
            path_str += f"({edge[0]} -> {edge[1]}, t={edge[3]}->{edge[4]}) "
        
        return path_str


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


class TreeNode:
    """Node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, path: TemporalPath, parent=None):
        self.path = path
        self.parent = parent
        self.children = {}  # Map from action to child node
        self.visit_count = 0
        self.state_value = 0.0
        self.avg_reward = 0.0
        self.mem_reward = 0.0
        self.priority = 0.0
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been expanded."""
        return len(self.children) == len(self.path.get_actions())
    
    def is_terminal(self) -> bool:
        """Check if the node is terminal (no actions available)."""
        return len(self.path.get_actions()) == 0


class AMCTS:
    """
    Adaptive Monte Carlo Tree Search algorithm for temporal path discovery.
    """
    
    def __init__(self, graph: AttributedDynamicGraph, embedding_dim: int = 8, 
                 max_iterations: int = 1000, replay_memory_size: int = 1000,
                 max_sample_num: int = 100, edge_limit: int = 4,
                 discount_positive: float = 0.95, discount_negative: float = 0.9,
                 exploration_weight: float = 0.45, priority_weight: float = 0.3):
        """
        Initialize the A-MCTS algorithm.
        
        Args:
            graph: The attributed dynamic graph
            embedding_dim: Dimensionality for edge embeddings
            max_iterations: Maximum number of iterations for search
            replay_memory_size: Maximum size of the replay memory
            max_sample_num: Maximum number of states to sample for query
            edge_limit: Minimum number of edges a state should have to be stored
            discount_positive: Discount factor for positive rewards
            discount_negative: Discount factor for negative rewards
            exploration_weight: Weight for exploration in UCT
            priority_weight: Weight for priority in UCT
        """
        self.graph = graph
        self.max_iterations = max_iterations
        self.discount_positive = discount_positive
        self.discount_negative = discount_negative
        self.exploration_weight = exploration_weight
        self.priority_weight = priority_weight
        
        # Initialize edge embeddings
        self.edge_embedder = EdgeEmbedding(graph, embedding_dim)
        self.edge_embedder.train()
        
        # Initialize replay memory
        self.replay_memory = ReplayMemory(
            max_size=replay_memory_size,
            max_sample_num=max_sample_num,
            edge_limit=edge_limit
        )
        
        # Best path found so far
        self.best_path = None
        self.best_path_length = float('inf')
    
    def find_path(self, source: int, destination: int, 
                  constraints: Dict[str, float], 
                  time_interval: Tuple[int, int]) -> Optional[TemporalPath]:
        """
        Find the shortest feasible temporal path from source to destination.
        
        Args:
            source: Source node
            destination: Destination node
            constraints: Dictionary mapping attribute names to upper bound values
            time_interval: Tuple (min_time, max_time) for departure and arrival
            
        Returns:
            The shortest feasible temporal path if found, None otherwise
        """
        # Initialize the root node
        root_path = TemporalPath(self.graph)
        root_path.source = source
        root = TreeNode(root_path)
        
        # Initialize the best path
        self.best_path = None
        self.best_path_length = float('inf')
        
        # Run Monte Carlo Tree Search
        for i in range(self.max_iterations):
            # Selection and expansion
            leaf, state_list = self._select_and_expand(root, constraints)
            
            # Simulation
            reward = self._simulate(leaf, destination, constraints, time_interval)
            
            # Backpropagation
            self._backpropagate(leaf, state_list, reward, constraints)
            
            # Early stopping if no better path can be found
            if i % 100 == 0 and i > 0:
                # Check if progress has been made
                if self.best_path is None:
                    continue
                
                # Check if the search has converged
                # This is a simple heuristic - in practice, you'd use more sophisticated criteria
                if self.best_path.length == self.best_path_length:
                    break
        
        return self.best_path
    
    def _select_and_expand(self, root: TreeNode, constraints: Dict[str, float]) -> Tuple[TreeNode, List[TreeNode]]:
        """
        Select a node to expand using UCT and expand it.
        
        Args:
            root: Root node of the tree
            constraints: Dictionary mapping attribute names to upper bound values
            
        Returns:
            A tuple (leaf node, list of nodes in the path from root to leaf)
        """
        node = root
        state_list = [node]
        
        # Selection: Traverse the tree using UCT until reaching a leaf node
        while not node.is_terminal() and node.is_fully_expanded():
            # Select the best child according to UCT
            node = self._select_best_child(node)
            state_list.append(node)
        
        # Expansion: If the node is not terminal and not fully expanded, expand it
        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node, constraints)
            state_list.append(node)
        
        return node, state_list
    
    def _select_best_child(self, node: TreeNode) -> TreeNode:
        """
        Select the best child node according to UCT formula.
        
        Args:
            node: The parent node
            
        Returns:
            The best child node
        """
        # Get possible actions
        actions = node.path.get_actions()
        
        # UCT scores for each action
        uct_scores = {}
        
        for action in actions:
            if action in node.children:
                child = node.children[action]
                
                # Exploitation term: action-value function
                action_value = self._action_value(node, child)
                
                # Exploration term
                exploration = self.exploration_weight * math.sqrt(
                    math.log(node.visit_count) / (child.visit_count + 1e-5)
                )
                
                # Priority term
                priority = self.priority_weight * (1.0 - child.priority)
                
                # UCT score
                uct_scores[action] = action_value + exploration + priority
            
        # Select the action with the highest UCT score
        best_action = max(uct_scores.items(), key=lambda x: x[1])[0]
        return node.children[best_action]
    
    def _expand(self, node: TreeNode, constraints: Dict[str, float]) -> TreeNode:
        """
        Expand a node by adding a new child.
        
        Args:
            node: The node to expand
            constraints: Dictionary mapping attribute names to upper bound values
            
        Returns:
            The newly created child node
        """
        # Get unexpanded actions
        unexpanded_actions = [
            action for action in node.path.get_actions()
            if action not in node.children
        ]
        
        if not unexpanded_actions:
            return node
        
        # Select a random unexpanded action
        action = random.choice(unexpanded_actions)
        
        # Create a new path by adding the action
        new_path = node.path.copy()
        new_path.add_edge(action)
        
        # Create a new node
        child = TreeNode(new_path, parent=node)
        
        # Initialize state value using attribute values
        child.state_value = self._initial_state_value(new_path, constraints)
        
        # Add the child to the parent
        node.children[action] = child
        
        return child
    
    def _simulate(self, node: TreeNode, destination: int, 
                  constraints: Dict[str, float], 
                  time_interval: Tuple[int, int]) -> float:
        """
        Simulate a random play-out from the node until reaching a terminal state.
        
        Args:
            node: The starting node
            destination: Destination node
            constraints: Dictionary mapping attribute names to upper bound values
            time_interval: Tuple (min_time, max_time) for departure and arrival
            
        Returns:
            The reward of the terminal state
        """
        current_path = node.path.copy()
        
        # Check if the current path is a terminal state
        if current_path.current_node == destination:
            # Check if it's feasible
            if current_path.is_feasible(constraints, time_interval):
                # Found a feasible path
                path_length = current_path.length
                
                # Update the best path if this one is shorter
                if path_length < self.best_path_length:
                    self.best_path = current_path.copy()
                    self.best_path_length = path_length
                
                # Calculate reward based on path length
                # Shorter paths get higher rewards
                if self.best_path is not None:
                    reward = 0.5 * min(1.0, self.best_path_length / path_length)
                else:
                    reward = 0.5
                
                return reward
            else:
                # Infeasible path (violates constraints)
                return 0.0
        
        # Simulate until reaching a terminal state
        while True:
            # Get possible actions
            actions = current_path.get_actions()
            
            if not actions:
                # No more actions, terminal state
                return 0.0
            
            # Select a random action
            action = random.choice(actions)
            
            # Apply the action
            current_path.add_edge(action)
            
            # Check if we reached the destination
            if current_path.current_node == destination:
                # Check if the path is feasible
                if current_path.is_feasible(constraints, time_interval):
                    # Found a feasible path
                    path_length = current_path.length
                    
                    # Update the best path if this one is shorter
                    if path_length < self.best_path_length:
                        self.best_path = current_path.copy()
                        self.best_path_length = path_length
                    
                    # Calculate reward
                    if self.best_path is not None:
                        reward = 0.5 * min(1.0, self.best_path_length / path_length)
                    else:
                        reward = 0.5
                    
                    return reward
                else:
                    # Infeasible path
                    return 0.0
            
            # Check if the path violates any constraint
            for attr, upper_bound in constraints.items():
                if current_path.calculate_attribute_value(attr) > upper_bound:
                    # Path violates constraint
                    return 0.0
            
            # Check if we've exceeded the time interval
            if current_path.arrival_time > time_interval[1]:
                # Path exceeds max time
                return 0.0
    
    def _backpropagate(self, leaf: TreeNode, state_list: List[TreeNode], 
                       reward: float, constraints: Dict[str, float]) -> None:
        """
        Backpropagate the reward through the tree and update statistics.
        
        Args:
            leaf: The leaf node where simulation started
            state_list: List of nodes in the path from root to leaf
            reward: The reward obtained from simulation
            constraints: Dictionary mapping attribute names to upper bound values
        """
        # Update all nodes in the path
        for i, node in enumerate(state_list):
            # Increment visit count
            node.visit_count += 1
            
            # Update average reward
            node.avg_reward = (node.avg_reward * (node.visit_count - 1) + reward) / node.visit_count
            
            # Calculate current reward for the node
            if reward > 0:
                # Positive reward (found a feasible path)
                # Discount based on distance from leaf
                current_reward = self.discount_positive ** (len(state_list) - i - 1) * reward
            else:
                # Negative reward (infeasible path)
                # Use a smaller discount factor for penalties
                penalty = min(1.0, abs(node.state_value))
                current_reward = -self.discount_negative ** (len(state_list) - i - 1) * penalty
            
            # Update state value
            node.state_value = node.state_value + 0.1 * (current_reward - node.state_value)
            
            # Calculate attribute and trajectory embeddings
            attr_embedding = node.path.get_attribute_embedding(constraints)
            traj_embedding = [self.edge_embedder.get_embedding(edge) for edge in node.path.edges]
            
            # Update replay memory
            if i == 0:
                # Root node doesn't need to be in replay memory
                continue
                
            # Update the node's memory reward through query
            node.mem_reward = self.replay_memory.query(
                node.path, attr_embedding, traj_embedding, constraints
            )
            
            # Update or store in replay memory
            self.replay_memory.update(
                node.path, node.avg_reward, node.visit_count, current_reward
            )
            
            if node.visit_count == 1:
                # First visit, store in replay memory
                self.replay_memory.store(
                    node.path, attr_embedding, traj_embedding,
                    node.avg_reward, node.visit_count, node.priority
                )
    
    def _initial_state_value(self, path: TemporalPath, constraints: Dict[str, float]) -> float:
        """
        Calculate the initial state value for a path.
        
        Args:
            path: The temporal path
            constraints: Dictionary mapping attribute names to upper bound values
            
        Returns:
            Initial state value in [-1, 1]
        """
        # Initialize value
        value = 0.0
        
        # Check if path violates any constraint
        for attr, upper_bound in constraints.items():
            attr_value = path.calculate_attribute_value(attr)
            
            if attr_value > upper_bound:
                # Path violates constraint
                return -1.0
            
            # Contribution to value is higher when attribute value is lower compared to constraint
            if upper_bound > 0:
                contribution = 1.0 - attr_value / upper_bound
                value += contribution
        
        # Normalize to [0, 1] range
        if constraints:
            value /= len(constraints)
        
        return value
    
    def _action_value(self, parent: TreeNode, child: TreeNode) -> float:
        """
        Calculate the action value (Q-value) for taking an action from parent to child.
        
        Args:
            parent: The parent node
            child: The child node
            
        Returns:
            Action value
        """
        # Weight for memory reward vs. average reward
        memory_weight = 0.0
        
        if child.visit_count > 0:
            # Calculate based on action count
            action_count = len(child.path.get_actions())
            if action_count > 0:
                memory_weight = min(1.0, math.log(child.visit_count) / math.log(1 + action_count))
        
        # Calculate the weighted average of memory reward and average reward
        child_reward = memory_weight * child.mem_reward + (1 - memory_weight) * child.avg_reward
        
        # Action value is the difference in rewards
        # If the difference is positive, it means the action leads to a better state
        return child_reward - parent.avg_reward


def create_example_graph() -> AttributedDynamicGraph:
    """Create an example attributed dynamic graph for testing."""
    graph = AttributedDynamicGraph()
    
    # Add edges with attributes (cost and time)
    # Format: (source, dest, length, dep_time, arr_time, attributes)
    
    # Simple graph with fewer nodes and edges
    graph.add_edge(1, 2, 2, 0, 2, {"cost": 2, "time": 2})
    graph.add_edge(1, 3, 3, 0, 3, {"cost": 3, "time": 3})
    graph.add_edge(2, 4, 2, 3, 5, {"cost": 2, "time": 2})
    graph.add_edge(3, 4, 2, 4, 6, {"cost": 2, "time": 2})
    
    return graph


def main():
    """Main function to test the A-MCTS algorithm."""
    # Create example graph
    graph = create_example_graph()
    
    # Create A-MCTS instance
    amcts = AMCTS(
        graph, 
        embedding_dim=8, 
        max_iterations=1000,
        replay_memory_size=500,
        max_sample_num=100,
        edge_limit=2,
        discount_positive=0.95,
        discount_negative=0.9,
        exploration_weight=0.45,
        priority_weight=0.3
    )
    
    # Define source, destination, constraints and time interval
    source = 1
    destination = 4  # Changed from 7 to 4 to match our simpler graph
    constraints = {"cost": 10, "time": 12}
    time_interval = (0, 15)
    
    print(f"Finding path from {source} to {destination} with constraints: {constraints}")
    print(f"Time interval: {time_interval}")
    
    # Find path
    path = amcts.find_path(source, destination, constraints, time_interval)
    
    # Print results
    if path is None:
        print("No feasible path found.")
    else:
        print("\nFound path:")
        print(path)
        print(f"Path length: {path.length}")
        print(f"Path cost: {path.calculate_attribute_value('cost')}")
        print(f"Path time: {path.calculate_attribute_value('time')}")
        print(f"Arrival time: {path.arrival_time}")
        
        print("\nPath edges:")
        for i, edge in enumerate(path.edges):
            src, dst, length, dep, arr = edge
            cost = graph.get_edge_attribute(edge, "cost")
            time = graph.get_edge_attribute(edge, "time")
            print(f"Edge {i+1}: {src} -> {dst}, Length: {length}, Time: {dep}->{arr}, Cost: {cost}, Time: {time}")


if __name__ == "__main__":
    main()