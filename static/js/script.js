// JavaScript for A-MCTS Visualization

document.addEventListener('DOMContentLoaded', function() {
    // Graph visualization variables
    let svg = d3.select('#graphVisualization');
    let width = svg.node().parentElement.clientWidth;
    let height = svg.node().parentElement.clientHeight;
    let simulation, nodes, links, nodeElements, linkElements, linkLabels, nodeLabels;
    let transform = d3.zoomIdentity;
    
    // Timeline variables
    let timeline = d3.select('#timeline');
    
    // Modal elements
    const modal = document.getElementById('graphEditorModal');
    const modalOpenBtn = document.getElementById('editGraphBtn');
    const modalCloseBtn = document.querySelector('.close-modal');
    
    // Setup zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            transform = event.transform;
            const graphContainer = svg.select('.graph-container');
            if (graphContainer.node()) {
                graphContainer.attr('transform', transform);
            }
        });
    
    svg.call(zoom);
    
    // Zoom controls
    document.getElementById('zoomIn').addEventListener('click', () => {
        svg.transition().call(zoom.scaleBy, 1.3);
    });
    
    document.getElementById('zoomOut').addEventListener('click', () => {
        svg.transition().call(zoom.scaleBy, 0.7);
    });
    
    document.getElementById('resetZoom').addEventListener('click', () => {
        svg.transition().call(zoom.transform, d3.zoomIdentity);
    });
    
    // View options
    document.getElementById('showEdgeLabels').addEventListener('change', function() {
        const edgeLabels = svg.selectAll('.edge-label');
        edgeLabels.style('visibility', this.checked ? 'visible' : 'hidden');
    });
    
    // Store the current path data for reapplying highlighting
    let currentPathData = null;
    
    document.getElementById('highlightPath').addEventListener('change', function() {
        if (this.checked && currentPathData) {
            // Reapply highlighting using the stored path data
            highlightPath(currentPathData);
        } else {
            // Remove highlighting but keep the currentPathData
            nodeElements.classed('source', false)
                        .classed('target', false)
                        .classed('path-node', false);
            linkElements.classed('path-edge', false);
        }
    });
    
    // Modal open and close
    modalOpenBtn.addEventListener('click', function() {
        modal.style.display = 'block';
    });
    
    modalCloseBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });
    
    // Close modal when clicking outside of it
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Handle form submission
    document.getElementById('pathForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        document.getElementById('loading-indicator').style.display = 'flex';
        
        // Get form data
        const formData = {
            source: document.getElementById('source').value,
            destination: document.getElementById('destination').value,
            cost_constraint: document.getElementById('cost_constraint').value,
            time_constraint: document.getElementById('time_constraint').value,
            start_time: document.getElementById('start_time').value,
            end_time: document.getElementById('end_time').value
        };
        
        // Send request to find path
        fetch('/find_path', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            document.getElementById('loading-indicator').style.display = 'none';
            
            // Display results
            displayResults(data);
            
            // Update graph visualization based on result
            if (data.success) {
                highlightPath(data.path);
                drawTimeline(data.path);
            } else {
                // Reset visualizations when no path is found
                resetGraphHighlighting();
                resetTimeline("No feasible path found between the selected nodes.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('loading-indicator').style.display = 'none';
            alert('An error occurred while processing your request.');
        });
    });
      // Graph Editor Controls
    // Reset to example graph
    document.getElementById('resetExampleBtn').addEventListener('click', function() {
        resetExampleGraph();
    });
    
    // Clear the graph
    document.getElementById('clearGraphBtn').addEventListener('click', function() {
        clearGraph();
    });
    
    // Add a new node
    document.getElementById('addNodeBtn').addEventListener('click', function() {
        const nodeId = parseInt(document.getElementById('newNodeId').value);
        if (isNaN(nodeId) || nodeId < 1) {
            showNotification("Node ID must be a positive integer", "error");
            return;
        }
        
        addNode(nodeId);
    });
    
    // Add a new edge
    document.getElementById('addEdgeBtn').addEventListener('click', function() {
        const source = parseInt(document.getElementById('edgeSource').value);
        const target = parseInt(document.getElementById('edgeTarget').value);
        const length = parseFloat(document.getElementById('edgeLength').value);
        const depTime = parseInt(document.getElementById('edgeDepTime').value);
        const arrTime = parseInt(document.getElementById('edgeArrTime').value);
        const cost = parseFloat(document.getElementById('edgeCost').value);
        const time = parseFloat(document.getElementById('edgeTime').value);
        
        // Validate input
        if (isNaN(source) || isNaN(target) || isNaN(length) || 
            isNaN(depTime) || isNaN(arrTime) || isNaN(cost) || isNaN(time)) {
            showNotification("All fields must be valid numbers", "error");
            return;
        }
        
        if (arrTime <= depTime) {
            showNotification("Arrival time must be after departure time", "error");
            return;
        }
        
        addEdge(source, target, length, depTime, arrTime, cost, time);
    });
    
    // Delete an edge
    document.getElementById('deleteEdgeBtn').addEventListener('click', function() {
        const source = parseInt(document.getElementById('deleteSource').value);
        const target = parseInt(document.getElementById('deleteTarget').value);
        const depTime = parseInt(document.getElementById('deleteDepTime').value);
        
        if (isNaN(source) || isNaN(target) || isNaN(depTime)) {
            showNotification("All fields must be valid numbers", "error");
            return;
        }
        
        deleteEdge(source, target, depTime);
    });
    
    // Close modal after successful operations
    function closeModalAfterSuccess() {
        setTimeout(() => {
            modal.style.display = 'none';
        }, 1000);
    }
    
    // Initial graph loading
    loadGraph();
    
    // Function to load and display the graph
    function loadGraph() {
        fetch('/get_graph')
            .then(response => response.json())
            .then(data => {
                initializeGraph(data);
                // Update the node selection dropdowns
                updateNodeSelections(data.nodes.map(n => n.id));
            })
            .catch(error => {
                console.error('Error loading graph:', error);
                showNotification("Error loading graph", "error");
            });
    }
    
    // Initialize the graph visualization
    function initializeGraph(data) {
        // Clear previous graph
        svg.selectAll('*').remove();
        
        // Create container for zoomable content
        const container = svg.append('g')
            .attr('class', 'graph-container');
            
        // Extract nodes and links from data
        nodes = data.nodes;
        links = data.edges.map(d => ({
            source: d.source,
            target: d.target,
            length: d.length,
            dep_time: d.dep_time,
            arr_time: d.arr_time,
            cost: d.cost,
            time: d.time
        }));
        
        // Create force simulation
        simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links)
                .id(d => d.id)
                .distance(d => d.length * 80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        // Create links
        linkElements = container.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('class', 'link')
            .attr('stroke-width', 2);
        
        // Create link labels
        linkLabels = container.append('g')
            .attr('class', 'link-labels')
            .selectAll('text')
            .data(links)
            .enter().append('text')
            .attr('class', 'edge-label')
            .text(d => `Cost: ${d.cost}, Time: ${d.time}`);
        
        // Create nodes
        nodeElements = container.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('class', 'node')
            .attr('r', 20)
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // Create node labels
        nodeLabels = container.append('g')
            .attr('class', 'node-labels')
            .selectAll('text')
            .data(nodes)
            .enter().append('text')
            .attr('class', 'node-label')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .text(d => d.label);
        
        // Add tooltips to nodes
        nodeElements.each(function(d) {
            tippy(this, {
                content: `Node: ${d.label}`,
                allowHTML: false
            });
        });
        
        // Add tooltips to links
        linkElements.each(function(d) {
            tippy(this, {
                content: `From ${d.source.id} to ${d.target.id}<br>Cost: ${d.cost}<br>Time: ${d.time}<br>Departure: ${d.dep_time}<br>Arrival: ${d.arr_time}`,
                allowHTML: true
            });
        });
        
        // Update the simulation on tick
        simulation.on('tick', () => {
            linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
                
            linkLabels
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2);
                
            nodeElements
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
                
            nodeLabels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }
    
    // Graph Editor Functions    // Reset to example graph
    function resetExampleGraph() {
        fetch('/reset_example_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification("Example graph loaded", "success");
                loadGraph();
                closeModalAfterSuccess();
            } else {
                showNotification("Failed to load example graph", "error");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification("Error loading example graph", "error");
        });
    }
    
    // Clear graph
    function clearGraph() {
        fetch('/clear_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification("Graph cleared successfully", "success");
                loadGraph();
                closeModalAfterSuccess();
            } else {
                showNotification("Failed to clear graph", "error");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification("Error clearing graph", "error");
        });
    }
    
    // Add node
    function addNode(nodeId) {
        fetch('/add_node', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ node_id: nodeId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(`Node ${nodeId} added successfully`, "success");
                loadGraph();
            } else {
                showNotification(data.message, "warning");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification("Error adding node", "error");
        });
    }
    
    // Add edge
    function addEdge(source, target, length, depTime, arrTime, cost, time) {
        fetch('/add_edge', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source: source,
                target: target,
                length: length,
                dep_time: depTime,
                arr_time: arrTime,
                cost: cost,
                time: time
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(`Edge from ${source} to ${target} added successfully`, "success");
                loadGraph();
            } else {
                showNotification(data.message, "error");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification("Error adding edge", "error");
        });
    }
    
    // Delete edge
    function deleteEdge(source, target, depTime) {
        fetch('/delete_edge', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source: source,
                target: target,
                dep_time: depTime
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(`Edge from ${source} to ${target} deleted successfully`, "success");
                loadGraph();
            } else {
                showNotification(data.message, "warning");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification("Error deleting edge", "error");
        });
    }
    
    // Update node selection dropdowns
    function updateNodeSelections(nodeIds) {
        // Sort node IDs for consistent display
        nodeIds.sort((a, b) => a - b);
        
        // Update source and destination dropdowns for path finding
        const sourceSelect = document.getElementById('source');
        const destinationSelect = document.getElementById('destination');
        
        // Clear existing options
        sourceSelect.innerHTML = '';
        destinationSelect.innerHTML = '';
        
        // Add new options
        nodeIds.forEach(nodeId => {
            const sourceOption = document.createElement('option');
            sourceOption.value = nodeId;
            sourceOption.textContent = nodeId;
            sourceSelect.appendChild(sourceOption);
            
            const destOption = document.createElement('option');
            destOption.value = nodeId;
            destOption.textContent = nodeId;
            destinationSelect.appendChild(destOption);
        });
        
        // Update end time to match max time
        fetch('/get_graph')
            .then(response => response.json())
            .then(data => {
                const maxTime = Math.max(...data.edges.map(e => e.arr_time), 0);
                document.getElementById('end_time').value = maxTime;
            });
    }
    
    // Notification system
    function showNotification(message, type = 'default') {
        const notification = document.getElementById('notification');
        const notificationMessage = document.getElementById('notification-message');
        
        // Set message
        notificationMessage.textContent = message;
        
        // Clear previous classes and add type class
        notification.className = 'notification';
        if (type !== 'default') {
            notification.classList.add(type);
        }
        
        // Show notification
        notification.classList.add('show');
        
        // Set timeout to hide notification
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
    
    // Close notification when clicked
    document.querySelector('.notification-close').addEventListener('click', function() {
        document.getElementById('notification').classList.remove('show');
    });
    
    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    // Display results in the results panel
    function displayResults(data) {
        const resultsPanel = document.getElementById('pathResults');
        resultsPanel.innerHTML = '';
        
        if (!data.success) {
            resultsPanel.innerHTML = `<div class="path-not-found">
                <p>${data.message}</p>
            </div>`;
            return;
        }
        
        const path = data.path;
        
        // Create summary element
        const summary = document.createElement('div');
        summary.className = 'path-summary';
        summary.innerHTML = `
            <h3>Path Found: ${path.nodes.join(' → ')}</h3>
            <p><strong>Length:</strong> ${path.length}</p>
            <p><strong>Cost:</strong> ${path.cost}</p>
            <p><strong>Time:</strong> ${path.time}</p>
            <p><strong>Arrival Time:</strong> ${path.arrival_time}</p>
        `;
        
        // Create details element
        const details = document.createElement('div');
        details.className = 'path-details';
        details.innerHTML = '<h4>Path Edges:</h4>';
        
        // Add all edges
        path.edges.forEach((edge, i) => {
            const edgeDiv = document.createElement('div');
            edgeDiv.className = 'edge-info';
            edgeDiv.innerHTML = `
                <p><strong>${i + 1}:</strong> ${edge.source} → ${edge.target} | 
                Cost: ${edge.cost} | Time: ${edge.time} | 
                Departure: ${edge.departure} | Arrival: ${edge.arrival}</p>
            `;
            details.appendChild(edgeDiv);
        });
        
        // Add to results panel
        const resultDiv = document.createElement('div');
        resultDiv.className = 'path-result';
        resultDiv.appendChild(summary);
        resultDiv.appendChild(details);
        resultsPanel.appendChild(resultDiv);
    }
      // Highlight path in the graph
    function highlightPath(path) {
        // Store the current path data for reuse
        currentPathData = path;
        
        // Reset all nodes and links
        nodeElements.classed('source', false)
                    .classed('target', false)
                    .classed('path-node', false);
                    
        linkElements.classed('path-edge', false);
        
        if (!path || !path.edges || path.edges.length === 0) {
            return;
        }
        
        // Check if highlighting is enabled
        const shouldHighlight = document.getElementById('highlightPath').checked;
        if (!shouldHighlight) {
            return;
        }
        
        // Set source and target nodes
        const sourceId = path.edges[0].source;
        const targetId = path.edges[path.edges.length - 1].target;
        
        // Highlight path nodes
        const pathNodeIds = new Set(path.nodes);
        
        nodeElements.classed('path-node', d => pathNodeIds.has(d.id) && d.id !== sourceId && d.id !== targetId)
                    .classed('source', d => d.id === sourceId)
                    .classed('target', d => d.id === targetId);
                    
        // Highlight path edges
        const pathEdgeMap = new Map();
        path.edges.forEach(edge => {
            pathEdgeMap.set(`${edge.source}-${edge.target}`, true);
        });
        
        linkElements.classed('path-edge', d => pathEdgeMap.has(`${d.source.id}-${d.target.id}`));
        
        // Show edge labels by default
        document.getElementById('showEdgeLabels').checked = true;
        document.getElementById('highlightPath').checked = true;
    }
      // Draw timeline visualization
    function drawTimeline(path) {
        const timelineContainer = document.getElementById('timeline');
        timelineContainer.innerHTML = '';
        
        if (!path || !path.edges || path.edges.length === 0) {
            resetTimeline("No path data to display.");
            return;
        }
        
        const startTime = parseInt(path.edges[0].departure);
        const endTime = parseInt(path.arrival_time);
        const timeRange = endTime - startTime;
        
        // Create SVG
        const svgWidth = Math.max(timelineContainer.clientWidth, timeRange * 40);
        const svgHeight = 100;
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };
        
        const svg = d3.select(timelineContainer).append('svg')
            .attr('width', svgWidth)
            .attr('height', svgHeight);
            
        // Create scales
        const xScale = d3.scaleLinear()
            .domain([startTime, endTime])
            .range([padding.left, svgWidth - padding.right]);
            
        const yScale = d3.scalePoint()
            .domain(path.nodes.map(String))
            .range([padding.top, svgHeight - padding.bottom]);
            
        // Create axis
        const xAxis = d3.axisBottom(xScale);
        svg.append('g')
            .attr('transform', `translate(0, ${svgHeight - padding.bottom})`)
            .call(xAxis);
            
        const yAxis = d3.axisLeft(yScale);
        svg.append('g')
            .attr('transform', `translate(${padding.left}, 0)`)
            .call(yAxis);
            
        // Draw edges
        path.edges.forEach(edge => {
            const sourceNode = edge.source.toString();
            const targetNode = edge.target.toString();
            const startX = xScale(edge.departure);
            const endX = xScale(edge.arrival);
            const sourceY = yScale(sourceNode);
            const targetY = yScale(targetNode);
            
            // Draw horizontal line for the time spent on the edge
            svg.append('line')
                .attr('x1', startX)
                .attr('y1', sourceY)
                .attr('x2', endX)
                .attr('y2', sourceY)
                .attr('stroke', 'var(--clr-primary-a30)')
                .attr('stroke-width', 2);
                
            // Draw vertical line to connect to the next node
            svg.append('line')
                .attr('x1', endX)
                .attr('y1', sourceY)
                .attr('x2', endX)
                .attr('y2', targetY)
                .attr('stroke', 'var(--clr-primary-a10)')
                .attr('stroke-width', 1.5)
                .attr('stroke-dasharray', '3,3');
        });
        
        // Draw nodes
        path.nodes.forEach((node, index) => {
            let x = padding.left;
            let y = yScale(node.toString());
            
            // For the first node, use the departure time
            if (index === 0) {
                x = xScale(path.edges[0].departure);
            } 
            // For intermediate nodes, use arrival time of incoming edge
            else if (index < path.edges.length) {
                x = xScale(path.edges[index-1].arrival);
            } 
            // For the last node, use the arrival time of the last edge
            else if (index === path.edges.length) {
                x = xScale(path.edges[index-1].arrival);
            }
            
            svg.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', 5)
                .attr('fill', index === 0 ? 'var(--clr-primary-a20)' : 
                       index === path.nodes.length-1 ? 'var(--clr-success)' : 'var(--clr-primary-a10)')
                .attr('stroke', 'var(--clr-light-a0)')
                .attr('stroke-width', 1);
        });
        
        // Add title
        svg.append('text')
            .attr('x', svgWidth / 2)
            .attr('y', padding.top - 5)
            .attr('text-anchor', 'middle')
            .style('font-size', '10px')
            .style('fill', 'var(--clr-text-secondary)')
            .text(`Time Progression (${startTime} to ${endTime})`);
    }
      // Reset all graph highlighting to show original graph
    function resetGraphHighlighting() {
        // Reset the stored path data
        currentPathData = null;
        
        // Remove all highlighting classes from nodes and edges
        nodeElements.classed('source', false)
                    .classed('target', false)
                    .classed('path-node', false);
                    
        linkElements.classed('path-edge', false);
        
        // Reset edge labels and path highlighting checkboxes to default
        document.getElementById('showEdgeLabels').checked = true;
        document.getElementById('highlightPath').checked = true;
    }
    
    // Reset the timeline to show a message when no path is found
    function resetTimeline(message) {
        const timelineContainer = document.getElementById('timeline');
        timelineContainer.innerHTML = '';
        
        // Create message container
        const messageDiv = document.createElement('div');
        messageDiv.className = 'path-not-found';
        messageDiv.innerHTML = `<p>${message}</p>`;
        
        timelineContainer.appendChild(messageDiv);
    }
    
    // Handle window resize
    window.addEventListener('resize', function() {
        width = svg.node().parentElement.clientWidth;
        height = svg.node().parentElement.clientHeight;
        
        if (simulation) {
            simulation.force('center', d3.forceCenter(width / 2, height / 2))
                     .alpha(0.3)
                     .restart();
        }
    });
});
