# Reverse PSICHIC: Sequence-based Protein Screening Against a Single Ligand

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BJozGtO3YuKyS48REjNoxEbos31z4KxC)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42256--024--00847--1-blue)](https://doi.org/10.1038/s42256-024-00847-1)

Reverse PSICHIC is a sequence-based virtual screening tool that evaluates thousands of proteins against a single ligand to predict binding affinity. This notebook was developed by **Logan Hessefort** as part of a US Department of Energy SCGSR Award at the National Renewable Energy Laboratory, with additional support from the US National Science Foundation (grant 2132183). @EvanKomp provided significant support for this project.

## Overview

This notebook implements a reverse screening workflow based on the PSICHIC architecture. Instead of screening many ligands against one protein (traditional virtual screening), Reverse PSICHIC screens many proteins against one ligand. This approach is ideal for identifying protein targets or off-target interactions for a specific compound.

## Features

- **Fast Virtual Screening**: Process 10,000+ protein-ligand pairs per hour
- **Large-Scale Capability**: Screen tens of thousands of proteins against one ligand
- **GPU Acceleration**: Automatic batch sizing based on available GPU memory
- **Memory-Efficient Processing**: Intelligent batching and memory management for large datasets
- **Sequence-Only Input**: No protein structures required

## Input Requirements

- **Protein Sequences**: CSV file containing:
  - `ID` column: Protein identifiers
  - `Protein` column: Amino acid sequences
- **Single Ligand**: One SMILES string for the compound of interest

## Use Cases

- **Target Discovery**: Identify protein targets for a specific ligand
- **Off-Target Prediction**: Evaluate potential off-target interactions
- **Protein Engineering**: Screen enzyme libraries for activity with a substrate
- **Selectivity Profiling**: Compare binding predictions across protein families

## How It Works

1. Upload a CSV file containing protein IDs and sequences
2. Input a single ligand SMILES string
3. Run the screening workflow
4. Receive predicted binding affinity scores for each protein

The notebook automatically detects your GPU type and optimizes batch size for maximum throughput while managing memory efficiently.

## Output

The screening produces a results CSV with:
- Protein ID and sequence
- Predicted binding affinity (pKd scale)
- Summary statistics (mean, median, top hits)

## Credits

This tool is based on PSICHIC (Physicochemical Graph Neural Network for Learning Protein-Ligand Interaction Fingerprints), developed by Koh et al.

**Original PSICHIC Publication:**  
Koh, H.Y., Nguyen, A.T.N., Pan, S., May, L.T., Webb, G.I. (2024). PSICHIC: physicochemical graph neural network for learning protein-ligand interaction fingerprints from sequence data. *Nature Machine Intelligence*, 6, 643–656. https://doi.org/10.1038/s42256-024-00847-1

**Original PSICHIC Repository:**  
https://github.com/huankoh/PSICHIC

## License

This project maintains the same license as the original PSICHIC repository.

## Contact

For questions or issues specific to Reverse PSICHIC, please contact Logan Hessefort via [LinkedIn](https://www.linkedin.com/in/logan-hessefort/) or open an issue on the [Reverse PSICHIC GitHub repository](https://github.com/Loganz97/Reverse-PSICHIC).
