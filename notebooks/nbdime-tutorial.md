# Using nbdime for Jupyter Notebook Collaboration

This tutorial will guide you through using nbdime to effectively collaborate on Jupyter notebooks in this project.

## 1. Setup

nbdime is already included in the project environment. To use it:

```bash
conda activate amcts
```

If you haven't already configured Git to use nbdime:

```bash
nbdime config-git --enable
```

This command sets up nbdime as the diff and merge tool for `.ipynb` files in Git.

## 2. Basic Usage

### Comparing Notebooks

To see differences between two notebook versions:

```bash
# Command-line diff
nbdiff notebooks/examples/01-amcts-introduction.ipynb notebooks/examples/01-amcts-introduction-modified.ipynb

# Web-based visual diff (recommended)
nbdiff-web notebooks/examples/01-amcts-introduction.ipynb notebooks/examples/01-amcts-introduction-modified.ipynb
```

### Viewing Notebook Diffs in Git

```bash
# Show diff of staged notebooks
git nbdiff --staged

# Show diff of all modified notebooks
git nbdiff

# Web interface for viewing diffs
git nbdiff-web
```

## 3. Merging Notebooks

When collaborating, you might need to merge different versions:

```bash
# Three-way merge (base, local, remote)
nbmerge-web base.ipynb local.ipynb remote.ipynb
```

## 4. Resolving Merge Conflicts

When Git reports conflicts in notebook files:

```bash
# Use nbdime as the merge tool
git mergetool --tool=nbdime notebook-with-conflict.ipynb
```

This will open nbdime's web-based merge tool showing:
- The base version (common ancestor)
- Your local changes
- Remote changes
- The output/merged notebook

You can then select which cells to keep from each version or edit the merged result directly.

## 5. Integration with Git Workflows

### During Regular Git Operations

Once configured, nbdime will automatically be used:

```bash
# When viewing diffs
git diff

# When resolving merge conflicts
git merge another-branch
git mergetool
```

### Working with Pull Requests

Before submitting a PR:
1. Make sure to sync with the upstream branch
2. Use `nbdiff-web` to review your notebook changes
3. Resolve any conflicts that might arise

## 6. Tips for Effective Collaboration

1. **Combine with nbstripout**: Our project uses both nbdime and nbstripout. nbstripout removes outputs before commit, while nbdime helps during diff/merge.

2. **Review semantic changes**: nbdime highlights changes in code cells, markdown cells, and outputs separately, so focus on meaningful changes.

3. **Use web interface for complex diffs**: The web interface (`nbdiff-web` and `nbmerge-web`) is more user-friendly for complex differences.

4. **Commit frequently**: Smaller, logical commits make notebook diffs more manageable.

5. **Document significant changes**: Add markdown comments above modified code cells to explain major changes.