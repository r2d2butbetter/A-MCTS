# A-MCTS: Adaptive Monte Carlo Tree Search for Temporal Path Discovery

An implementation of Monte Carlo Tree Search with reinforcement learning for finding optimal paths in temporal networks.

- 📄 **[Read the Paper](https://ieeexplore.ieee.org/document/9580584)**
- 🌐 **[Live Demo](https://amcts.vercel.app/)**

## Overview

A-MCTS efficiently discovers temporal paths in complex networks with multiple constraints, applicable to transportation, communication systems, and social network analysis.

## Quick Start

```bash
# Install dependencies
conda env create -f environment.yml  # or pip install -r requirements.txt
conda activate amcts

# Run locally
python app.py  # Then navigate to http://localhost:5000
```

## Repository Contents

- `src/`: Core algorithm implementation
- `data/`: Sample datasets
- `notebooks/`: Examples and demonstrations
- `app.py`: Web application

## Citation

```bibtex
@INPROCEEDINGS{9580584,
  author={Feng, Song and Song, Jiafei and Li, Xiuxian and Zeng, Xinwang},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)}, 
  title={A-MCTS: Adaptive Monte Carlo Tree Search for Temporal Path Discovery}, 
  year={2021},
  pages={1-8},
  doi={10.1109/IJCNN52387.2021.9580584}
}
```
