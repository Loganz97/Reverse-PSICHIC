# Reverse PSICHIC: Sequence-based Protein Screening Against a Single Ligand

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BJozGtO3YuKyS48REjNoxEbos31z4KxC)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42256--024--00847--1-blue)](https://doi.org/10.1038/s42256-024-00847-1)

Reverse PSICHIC is a sequence-based virtual screening tool that evaluates thousands of proteins against a single ligand to predict binding affinity. This notebook was developed by **Logan Hessefort** as part of a US Department of Energy SCGSR Award at the National Renewable Energy Laboratory, with additional support from the US National Science Foundation (grant 2132183).

## Overview

This notebook implements a reverse screening workflow based on the PSICHIC architecture for preliminary target identification. Instead of screening many ligands against one protein (traditional virtual screening), Reverse PSICHIC screens many proteins against one ligand. This approach is useful for identifying potential protein targets or off-target interactions for a specific compound.

**Important:** This tool predicts binding affinity only, not catalytic function, activity, or biological relevance. Predictions must be validated experimentally. This tool should be viewed as one preliminary filter to narrow large search spaces (thousands of proteins) down to a manageable set for experimental validation, however, one *in silico* metric should never be used as the sole reason for screening *in vitro*.

## Features

- **Fast Virtual Screening**: Process 18,000+ protein-ligand pairs per hour (300 proteins/min on T4 GPU)
- **Large-Scale Capability**: Screen from hundreds to 100,000+ proteins against one ligand
- **Highly Reproducible**: 52% identical rankings across runs, 85% within 1 position (Spearman r=~1)
- **GPU Acceleration**: Automatic batch sizing based on available GPU memory
- **Memory-Efficient Processing**: Intelligent batching and memory management for large datasets
- **Sequence-Only Input**: No protein structures required

## Input Requirements

- **Protein Sequences**: CSV file containing:
  - `ID` column: Protein identifiers
  - `Protein` column: Amino acid sequences
- **Single Ligand**: One SMILES string for the compound of interest

## Use Cases

- **Preliminary Target Discovery**: Identify potential protein targets for a specific ligand before experimental validation
- **Off-Target Prediction**: Evaluate potential off-target interactions across protein families
- **Protein Engineering**: Pre-screen enzyme libraries for potential substrate binding before expression
- **Selectivity Profiling**: Compare predicted binding across protein families as a first-pass filter

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

## Performance and Stability

**Throughput (T4 GPU):**
- 300 proteins/minute for typical sequences (314 AA average)
- 18,000+ proteins/hour
- Example screening times:
  - 1,000 proteins: 3.3 minutes
  - 10,000 proteins: 33 minutes
  - 25,000 proteins: 1.4 hours
  - 100,000 proteins: 5.6 hours

**Prediction Stability:**
Validation across 5 independent runs on 25,000 proteins shows exceptional reproducibility:
- 52% of proteins maintain identical rankings
- 85% change by ≤1 rank position  
- Spearman rank correlation = 0.9999999959 (near perfect relative order preservation)
- Mean coefficient of variation = 0.0001% for affinity predictions
- Top-ranked hits (top 1-5%) show near-zero variability

These stability metrics indicate that predicted rankings are reproducible and suitable for preliminary screening. However, predictions represent binding affinity potential only and must be validated experimentally for catalytic function, expression, solubility, and biological relevance.

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
