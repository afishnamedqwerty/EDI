# EDI
## Celestial Orbital Forecasting Time Series Foundation Model

A Python implementation of a time series foundation model inspired by the Sundial paper, designed to predict orbital vector probabilities for celestial objects.

## Overview

This project leverages the Sundial architecture to forecast future positions and velocities of celestial objects based on their orbital calculations. The model is trained on precise orbital data extracted from the PDS Spice archives, ERA 5 timeseries datasets, and Horizons orbital calculations (compiled into HDF5 format) and aims to output probability distributions of possible future states while handling uncertainties inherent in space dynamics.

## Key Features

- **EDI Architecture**: Implements the core concepts from the Sundial paper for time series forecasting.
- **Probabilistic Predictions**: Generates distributions of possible future positions and velocities.
- **Handling Uncertainties**: Designed to account for orbital mechanics complexities and data noise.
- **Scalability**: Built to handle large-scale celestial datasets efficiently.

## File Structure

The project is organized into the following directories:

- **preprocess/**: Handles the data preprocessing pipeline for each dataset into HDF5 format.
- **crazy_train.py**: Original model training implementation based on the Sundial paper including flow matching and TimeFlowLoss.
- **nsa_train.py**: Modified variant of EDI architecture that implements native sparse attention.
- **inference.py**: Inference script for the trained models.

