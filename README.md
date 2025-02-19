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


## Data Quality

Timebench from the Sundial paper contains large quantities of timeseries data for continuous tokenization, and this includes undescribed synthetic data. Anyone with a disgusting amount of compute is welcome to run ahead of me on synthetic data generation with the following proposal in the 'millennium.md' that describes a phased approach for integrating the theoretical framework of Bassani and Magueijo in "How to Build a Universe" with the constants of the Millenium Simulation, a large-scale N-body simulation that models the formation of structure in the universe by simulating gravitational interactions of dark matter particles over time. A phased approach for testing this one evolving constant at a time (starting with gravity) is described in the 'millennium.md' file. It might be useful, might not idk let me live.


