# A-MCTS: Adaptive Monte Carlo Tree Search for Temporal Path Discovery

This repository contains our implementation of the paper "A-MCTS: Adaptive Monte Carlo Tree Search for Temporal Path Discovery".

- [Find the Paper here](https://ieeexplore.ieee.org/document/9580584)

## Setup Instructions

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)

### Getting Started

1. Clone the repository:
   ```
   git clone <repository-url>
   cd A-MCTS
   ```

2. Create and activate the conda environment:
   ```
   conda env create -f environment.yml
   conda activate amcts
   ```

3. Register the environment with Jupyter:
   ```
   python -m ipykernel install --user --name=amcts --display-name="A-MCTS"
   ```

4. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
   Or if using VS Code, open VS Code and select the "amcts" kernel when opening a notebook.

## Repository Structure

- `notebooks/`: Contains all Jupyter notebooks for the A-MCTS implementation
  - `examples/`: Example notebooks demonstrating the A-MCTS algorithm
- `data/`: Data used in experiments (add to .gitignore if files are large)

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
