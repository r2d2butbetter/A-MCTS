import numpy as np
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