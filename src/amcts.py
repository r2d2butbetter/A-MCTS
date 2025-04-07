import math
import random
from typing import List, Dict, Tuple, Optional

import numpy as np

from .graph import AttributedDynamicGraph, TemporalPath
from .embedding import EdgeEmbedding
from .memory import ReplayMemory


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