Overview

This project is a PyTorch-based convolutional neural network pipeline for large-scale road surface classification using high-resolution aerial imagery.

The system classifies roads as paved or unpaved by:

Sampling image patches along road centerlines

Training a CNN on patch-level labels

Aggregating predictions to the road level

Applying noise-aware weighting strategies

The goal is scalable surface classification for geospatial applications such as cycling analytics, routing intelligence, and infrastructure mapping.

Problem

Surface labels in open mapping datasets are often incomplete or unreliable.

This project builds a data-driven classifier that:

Uses NAIP aerial imagery

Extracts structured road-aligned patches

Learns visual surface characteristics

Produces road-level predictions

The system is designed to operate at large scale (state-level road networks).

Current Implementation

All training and evaluation logic is implemented in:

50cnn2.py

This script includes:

Dataset loading

Patch sampling logic

CNN model definition

Training loop

Validation metrics

Road-level aggregation

Suspicious patch tracking

Weighted accuracy evaluation

Hyperparameters and training settings are defined at the top of the script for direct modification.

Model Architecture

PyTorch implementation

Pretrained ConvNeXt-Tiny backbone

Binary classification head (paved vs unpaved)

Class-weighted loss for imbalance handling

Noise-Aware Training Strategy

The training pipeline supports:

High-confidence (“gold”) labeled roads

Lower-confidence (“silver”) labeled roads

Adjustable silver weighting

Suspicious patch detection

Weighted road-level evaluation

This allows experimentation with label noise reduction and robustness strategies.

Road-Level Aggregation

Patch-level predictions are aggregated to produce:

Road-level probability

Weighted road accuracy

Suspicious road metrics

Road-level accuracy is treated as the primary evaluation metric.

Running Training

Install dependencies:

pip install -r requirements.txt

Run training:

python 50cnn2.py

To adjust hyperparameters, edit values at the top of the script.

Data

This repository does not include:

NAIP imagery

Patch caches

Model checkpoints

Large intermediate datasets

These are excluded due to size and storage constraints.

The pipeline expects externally prepared imagery and road geometry inputs.

Tech Stack

Python

PyTorch

NumPy

Geospatial preprocessing utilities

Future Improvements

Refactor into modular config-based system

Separate training and evaluation scripts

Improved label noise modeling

Multi-class surface classification

Cross-region generalization

Author

Zach Vorberger
Mechanical Engineer | Geospatial ML Developer
