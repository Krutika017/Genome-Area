# Genome Sequence Analysis and Classification Using Deep Learning

## Project Overview

This project focuses on genome sequence analysis and species classification using **Frequency Chaos Game Representation (FCGR)** and **Convolutional Neural Networks (CNNs)**. It converts DNA sequences obtained from FASTA files into visual representations, allowing deep learning models to identify and classify species based on genomic patterns.

Currently, the project focuses on **three bamboo species**, using their genomic sequences to generate FCGR images and train a CNN-based classification model.

## Objectives

- Process genomic DNA sequences from FASTA files.
- Perform preprocessing and remove invalid nucleotide characters.
- Convert DNA sequences into FCGR images.
- Extract genomic patterns through visual representations.
- Train a CNN model to classify three bamboo species.
- Evaluate model performance using training and validation metrics.

## Methodology

1. **Data Collection:** Collect genomic sequences of three bamboo species in FASTA format.
2. **Preprocessing:** Clean DNA sequences and handle invalid nucleotide characters.
3. **FCGR Generation:** Convert nucleotide sequences into graphical representations using the FCGR technique.
4. **Dataset Preparation:** Generate and organize FCGR images according to their respective species.
5. **Model Development:** Train a Convolutional Neural Network (CNN) for species classification.
6. **Evaluation:** Analyze model performance using accuracy, validation accuracy, and classification metrics.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Biopython
- Pillow (PIL)
- Jupyter Notebook

## Project Scope

The current implementation is limited to three bamboo species. The project aims to explore how genomic sequence patterns can be represented as images and used for automated species identification through deep learning.

## Future Improvements

- Expand the dataset to include more bamboo species.
- Experiment with different genomic feature extraction techniques.
- Optimize CNN architecture and hyperparameters.
- Improve classification performance and generalization.
- Explore Transformer-based models for genomic sequence analysis.

## Repository Status

**Status:** Research and Development 🚀

This project is an ongoing exploration of AI-driven genomic sequence analysis and deep learning-based species classification.
