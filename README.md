# DeepFusion: A Dive Into the MEDUSA Framework and its DeepSER Architecture.

**NTUA Speech and Language Processing (SLP) 2025 – Semester Project**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

Multimodal sentiment analysis faces fundamental challenges in effectively modeling interactions between heterogeneous modalities (text, audio, and vision), each characterized by distinct statistical properties, temporal dynamics, and varying noise profiles. 

**DeepFusion** introduces **Pairwise VarDepth DeepSER**, a novel architecture that addresses cross-modal fusion through explicit pairwise modeling. By processing text-audio, audio-vision, and vision-text interactions independently prior to final integration, our framework captures fine-grained cross-modal dependencies with high efficacy. Our variable-depth encoders strategically combine Transformer layers with convolutional kernels to capture both sequential dependencies and local patterns efficiently.

Extensive evaluations on the CMU-MOSEI dataset demonstrate consistent improvements of ~1% across all classification tasks (binary, 5-class, 7-class) compared to the baseline DeepSER architecture, achieving metrics comparable to state-of-the-art models.

### Key Contributions
- **Pairwise Fusion Strategy**: Explicit pairwise modeling of modalities before late integration.
- **Variable-Depth Encoders**: Hybrid integration of Transformer layers and convolutional kernels.
- **Enhanced Robustness**: Incorporation of modality-specific pretraining strategies, meta-classifier stacking, and model-soup weight averaging.

---

## Repository Structure & Submodules

This repository is structured as an integrated workspace that unifies several state-of-the-art frameworks in multimodal representation learning:

### 1. `deepser` & `mult` (Multimodal Transformer)
Implementation of the Multimodal Transformer (MulT) for unaligned multimodal language sequences. It provides the foundation for learning representations without relying on explicit temporal alignment between modalities.
*Reference:* Tsai et al., [Multimodal Transformer for Unaligned Multimodal Language Sequences](https://arxiv.org/abs/1906.00295), ACL 2019.

### 2. `multimodal-masking` ($M^3$)
Incorporates the $M^3$ (MultiModal Masking) methodology. This module applies a generic, lightweight regularization layer that randomly masks modalities to prevent over-reliance on any single data stream, enhancing the model's overall generalization.
*Reference:* Georgiou et al., [M3: MultiModal Masking Applied to Sentiment Analysis](https://www.isca-speech.org/archive/interspeech_2021/georgiou21_interspeech.html), INTERSPEECH 2021.

### 3. `slp` (Speech and Language Processing Framework)
A highly opinionated framework tailored for the fast and reproducible development of multimodal models, primarily emphasizing NLP tasks. This serves as the backbone for managing experiments, orchestrating PyTorch Lightning workflows, and streamlining data loading.
*Reference:* Paraskevopoulos, [slp Framework](https://github.com/georgepar/slp), 2020.

---

## Authors & Contributors

**National Technical University of Athens (NTUA) – School of ECE**

- **Karali Alexandra** (03121084)
- **Karatzanos Dimitrios** (03121083)
- **Pliatsika Magdalini** (03121220)
- **Raftopoulos Michail** (03120114)

**Supervisors:** 
- Georgiou Efthymios 
- Prof. Potamianos Alexandros

---

## Usage & Execution

Please refer to the specific `.md` files within each submodule (`deepser`, `mult`, `multimodal-masking`, `slp`) for specialized requirements. 

### General Prerequisites
- Python >= 3.7
- PyTorch >= 1.7
- [Poetry](https://python-poetry.org/) for dependency management (recommended for submodules).

### Setup Example
For exploring the masking models, please pull the required submodules correctly:
```bash
git clone --recurse-submodules https://github.com/your-repo/DeepFusion.git
cd DeepFusion/multimodal-masking
pip install poetry
poetry install
```

Detailed reproducibility commands and config variants are supplied in the individual module directories and the `report/` LaTeX source.

---

## References & Further Reading
A comprehensive analysis, model schematics, and performance tables are available in the project's compiled report located under `report/report.pdf`.
