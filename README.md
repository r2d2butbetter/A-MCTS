# A-MCTS: Adaptive Monte Carlo Tree Search for Temporal Path Discovery

This repository contains our implementation of the paper "A-MCTS: Adaptive Monte Carlo Tree Search for Temporal Path Discovery".

- [Find the Paper here](https://ieeexplore.ieee.org/document/9580584)

## Overview

A-MCTS is an algorithm for finding optimal paths in attributed dynamic graphs with constraints. This project includes:

- Core A-MCTS algorithm implementation
- Visualization web application for interactive path discovery
- Jupyter notebooks for experimentation and analysis

## Setup Instructions

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or pip

### Getting Started

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd A-MCTS
   ```

2. Install dependencies:
   
   With Conda (recommended):

   ```bash
   conda env create -f environment.yml
   conda activate amcts
   ```
   
   Or with pip:

   ```bash
   pip install -r requirements.txt
   ```

3. For Jupyter notebook exploration:

   ```bash
   python -m ipykernel install --user --name=amcts --display-name="A-MCTS"
   jupyter notebook
   ```

   Or use VS Code and select the "amcts" kernel when opening a notebook.

## Visualization Application

The project includes a web-based visualization tool to interact with the A-MCTS algorithm.

### Running the Visualization

1. Start the Flask application:

   ```bash
   python app.py
   ```

2. Open your browser and navigate to:

   ```
   http://localhost:5000
   ```

### Using the Visualization Interface

1. In the sidebar, enter your path parameters:
   - Source node
   - Destination node
   - Cost constraint
   - Time constraint
   - Time interval

2. Click "Find Path" to compute a path.

3. The results will appear in the right panel:
   - The graph visualization will highlight the found path
   - The timeline will show the temporal progression of the path
   - Detailed path information will be shown in the results panel

### Graph Visualization Features

The graph visualization uses D3.js to create a force-directed layout. You can:

- Zoom in/out using the controls or mouse wheel
- Drag nodes to rearrange them
- Hover over nodes and edges to see more information
- Toggle edge labels and path highlighting using the checkboxes

## Repository Structure

```
A-MCTS/
├── notebooks/
│   ├── 01_introduction.ipynb         # Overview of the project and algorithm
│   ├── 02_graph_representation.ipynb # Working with AttributedDynamicGraph and TemporalPath
│   ├── 03_mcts_algorithm.ipynb       # A-MCTS algorithm implementation and examples
│   └── 04_experiments.ipynb          # Running experiments and visualizing results
│
├── src/
│   ├── __init__.py
│   ├── graph.py                      # AttributedDynamicGraph and TemporalPath classes
│   ├── embedding.py                  # EdgeEmbedding class
│   ├── memory.py                     # ReplayMemory class
│   └── amcts.py                      # TreeNode and AMCTS classes
│
├── static/                           # Web application static files
│   ├── css/
│   └── js/
│
├── templates/                        # Web application templates
│   └── index.html                    # Main visualization interface
│
├── app.py                            # Flask web application
├── example.py                        # Example graph creation and simple run
├── environment.yml                   # Conda environment specification
├── requirements.txt                  # Pip requirements
└── README.md                         # This file

## Troubleshooting

If you encounter any issues:

1. Check the terminal output for error messages
2. Ensure all dependencies are properly installed
3. Verify that your browser is up-to-date (for visualization)

## Git and Jupyter Notebooks

To avoid git conflicts with Jupyter notebooks, we use two complementary tools:

### 1. nbstripout

Strips output cells before committing to reduce file size and conflicts:

```bash
conda activate amcts
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

### 2. nbdime

For improved diffing and merging of notebooks:

```bash
conda activate amcts
nbdime config-git --enable
```

This configures Git to use nbdime's content-aware diffing and merging tools for .ipynb files.

You can use these commands for better notebook collaboration:

- `nbdiff notebook1.ipynb notebook2.ipynb` - Show rich diff between notebooks
- `nbdiff-web notebook1.ipynb notebook2.ipynb` - Web-based visual diff
- `nbmerge notebook_base.ipynb notebook_local.ipynb notebook_remote.ipynb` - Three-way merge

For resolving conflicts during merge, use:

```bash
git mergetool --tool nbdime
```

## Repository Structure

a-mcts
├── notebooks
│   ├── 01_introduction.ipynb         # Overview of the project and algorithm
│   ├── 02_graph_representation.ipynb # Working with AttributedDynamicGraph and TemporalPath
│   ├── 03_mcts_algorithm.ipynb       # A-MCTS algorithm implementation and examples
│   └── 04_experiments.ipynb          # Running experiments and visualizing results
│
├── src
│   ├── __init__.py
│   ├── graph.py                      # AttributedDynamicGraph and TemporalPath classes
│   ├── embedding.py                  # EdgeEmbedding class
│   ├── memory.py                     # ReplayMemory class
│   └── amcts.py                      # TreeNode and AMCTS classes
│
├── examples.py                       # Example graph creation and simple run
└── README.md                         # This file

## Git and Jupyter Notebooks

To avoid git conflicts with Jupyter notebooks, we use two complementary tools:

### 1. nbstripout

Strips output cells before committing to reduce file size and conflicts:

```
conda activate amcts
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

### 2. nbdime

For improved diffing and merging of notebooks:

```
conda activate amcts
nbdime config-git --enable
```

This configures Git to use nbdime's content-aware diffing and merging tools for .ipynb files.

You can use these commands for better notebook collaboration:
- `nbdiff notebook1.ipynb notebook2.ipynb` - Show rich diff between notebooks
- `nbdiff-web notebook1.ipynb notebook2.ipynb` - Web-based visual diff
- `nbmerge notebook_base.ipynb notebook_local.ipynb notebook_remote.ipynb` - Three-way merge

For resolving conflicts during merge, use:
```
git mergetool --tool nbdime
```

## Best Practices

1. Each notebook should focus on a specific aspect of the implementation or experiments
2. Use consistent notebook naming conventions: `[number]-[description].ipynb` (e.g., `01-amcts-introduction.ipynb`)
3. Update the environment.yml file if you add new dependencies
4. Document your experiments and code clearly in notebooks
5. Use markdown cells to explain your implementation and findings
