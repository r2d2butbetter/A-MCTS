"""
Example usage of the A-MCTS algorithm for finding temporal paths in attributed dynamic graphs.
"""

from src.graph import AttributedDynamicGraph
from src.amcts import AMCTS


def create_example_graph() -> AttributedDynamicGraph:
    """Create an example attributed dynamic graph for testing."""
    graph = AttributedDynamicGraph()
    
    # Add edges with attributes (cost and time)
    # Format: (source, dest, length, dep_time, arr_time, attributes)
    
    graph.add_edge(1, 2, 2, 0, 2,  {"cost": 2, "time": 2})
    graph.add_edge(1, 3, 3, 0, 3,  {"cost": 3, "time": 3})
    graph.add_edge(2, 4, 2, 3, 5,  {"cost": 2, "time": 2})
    graph.add_edge(3, 4, 2, 4, 6,  {"cost": 2, "time": 2})
    graph.add_edge(2, 5, 4, 5, 9,  {"cost": 4, "time": 4})
    graph.add_edge(5, 6, 3, 9, 12, {"cost": 3, "time": 3})
    graph.add_edge(4, 6, 2, 6, 8,  {"cost": 2, "time": 2})
    graph.add_edge(3, 5, 5, 4, 9,  {"cost": 5, "time": 5})
    graph.add_edge(6, 7, 1, 12, 13,{"cost": 1, "time": 1})
    graph.add_edge(5, 7, 2, 10, 12,{"cost": 2, "time": 2})
    
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
    destination = 7
    constraints = {"cost": 6, "time": 15}  # High constraint values to allow more paths
    # Note: Setting high constraints allows the algorithm to explore more paths, 
    # including those with higher costs and times.
    
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