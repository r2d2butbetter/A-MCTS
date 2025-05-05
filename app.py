"""
Flask web application for A-MCTS algorithm visualization.
"""

import json
import networkx as nx
from flask import Flask, render_template, request, jsonify, url_for

from src.graph import AttributedDynamicGraph
from src.amcts import AMCTS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'amcts_visualization_key'

# Create a global instance of the example graph
from example import create_example_graph
# Initialize with a default graph, but allow users to modify it
graph = create_example_graph()

@app.route('/')
def index():
    """Render the main page."""
    # Get all nodes from the graph
    nodes = list(graph.nodes)
    nodes.sort()  # Sort for consistent display
    
    # Get min and max time from the graph
    max_time = graph.max_time
    
    return render_template('index.html', nodes=nodes, max_time=max_time)


@app.route('/find_path', methods=['POST'])
def find_path():
    """Handle path finding requests."""
    data = request.get_json()
    
    # Get parameters from the request
    source = int(data['source'])
    destination = int(data['destination'])
    constraints = {
        "cost": float(data['cost_constraint']),
        "time": float(data['time_constraint'])
    }
    time_interval = (
        int(data['start_time']),
        int(data['end_time'])
    )
    
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
    
    # Find path
    path = amcts.find_path(source, destination, constraints, time_interval)
    
    # Prepare result
    result = {}
    if path is None:
        result['success'] = False
        result['message'] = "No feasible path found."
    else:
        # Extract nodes from path edges
        nodes = []
        if path.edges:
            # Add the source node
            nodes.append(path.edges[0][0])
            # Add all destination nodes
            for edge in path.edges:
                nodes.append(edge[1])
        
        result['success'] = True
        result['path'] = {
            'nodes': nodes,
            'length': path.length,
            'cost': path.calculate_attribute_value('cost'),
            'time': path.calculate_attribute_value('time'),
            'arrival_time': path.arrival_time,
            'edges': []
        }
        
        # Add detailed edge information
        for edge in path.edges:
            src, dst, length, dep, arr = edge
            cost = graph.get_edge_attribute(edge, "cost")
            time = graph.get_edge_attribute(edge, "time")
            result['path']['edges'].append({
                'source': src,
                'target': dst,
                'length': length,
                'departure': dep,
                'arrival': arr,
                'cost': cost,
                'time': time
            })
    
    return jsonify(result)


@app.route('/get_graph', methods=['GET'])
def get_graph():
    """Return the graph structure for visualization."""
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph.nodes:
        G.add_node(node)
    
    # Add edges
    for edge in graph.edges:
        u, v, length, dep_time, arr_time = edge
        attributes = graph.attributes[edge]
        G.add_edge(u, v, length=length, 
                   dep_time=dep_time, 
                   arr_time=arr_time, 
                   cost=attributes['cost'], 
                   time=attributes['time'])
    
    # Convert to serializable format
    nodes = [{'id': n, 'label': str(n)} for n in G.nodes()]
    edges = [{'source': u, 'target': v, 'length': G[u][v]['length'],
              'dep_time': G[u][v]['dep_time'], 'arr_time': G[u][v]['arr_time'],
              'cost': G[u][v]['cost'], 'time': G[u][v]['time']} 
             for u, v in G.edges()]
    
    return jsonify({'nodes': nodes, 'edges': edges})


@app.route('/clear_graph', methods=['POST'])
def clear_graph():
    """Clear the current graph and prepare for a new one."""
    global graph
    # Initialize with a new empty graph
    graph = AttributedDynamicGraph()
    return jsonify({'success': True, 'message': 'Graph cleared successfully'})


@app.route('/add_node', methods=['POST'])
def add_node():
    """Add a new node to the graph."""
    data = request.get_json()
    node_id = int(data['node_id'])
    
    # Check if the node already exists
    if node_id in graph.nodes:
        return jsonify({'success': False, 'message': 'Node already exists'})
    
    # Add the node
    graph.add_node(node_id)
    return jsonify({'success': True, 'message': 'Node added successfully'})


@app.route('/add_edge', methods=['POST'])
def add_edge():
    """Add a new edge to the graph."""
    data = request.get_json()
    
    # Extract edge data
    source = int(data['source'])
    target = int(data['target'])
    length = float(data['length'])
    dep_time = int(data['dep_time'])
    arr_time = int(data['arr_time'])
    cost = float(data['cost'])
    time = float(data['time'])
    
    # Validate that arrival time is after departure time
    if arr_time <= dep_time:
        return jsonify({'success': False, 'message': 'Arrival time must be after departure time'})
    
    # Add the edge
    graph.add_edge(source, target, length, dep_time, arr_time, {"cost": cost, "time": time})
    
    return jsonify({'success': True, 'message': 'Edge added successfully'})


@app.route('/delete_edge', methods=['POST'])
def delete_edge():
    """Delete an edge from the graph."""
    data = request.get_json()
    
    source = int(data['source'])
    target = int(data['target'])
    dep_time = int(data['dep_time'])
    
    # Find and delete the edge
    edge_to_delete = None
    for edge in graph.edges:
        if edge[0] == source and edge[1] == target and edge[3] == dep_time:
            edge_to_delete = edge
            break
    
    if edge_to_delete:
        # Remove the edge from all data structures
        graph.edges.remove(edge_to_delete)
        del graph.attributes[edge_to_delete]
        
        # Update edge_index
        if edge_to_delete in graph.edge_index[source]:
            graph.edge_index[source].remove(edge_to_delete)
        
        # Update max_time if needed
        if edge_to_delete[4] == graph.max_time:
            # Recalculate max_time
            graph.max_time = max([edge[4] for edge in graph.edges]) if graph.edges else 0
            
        return jsonify({'success': True, 'message': 'Edge deleted successfully'})
    else:
        return jsonify({'success': False, 'message': 'Edge not found'})


@app.route('/reset_example_graph', methods=['POST'])
def reset_example_graph():
    """Reset to the default example graph."""
    global graph
    graph = create_example_graph()
    return jsonify({'success': True, 'message': 'Example graph loaded'})


if __name__ == '__main__':
    app.run(debug=False)