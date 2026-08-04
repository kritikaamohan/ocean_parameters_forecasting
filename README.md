# Performance Benchmarking of Deep Learning Architectures for Satellite Ocean Parameter Forecasting

This repository provides a unified benchmarking framework to evaluate, compare, and visualize five distinct deep learning architectural paradigms for independent multi-day forecasting of satellite-derived ocean parameters: **Absolute Dynamic Topography (ADT)** and **Chlorophyll-a (Chl-a)**.

The framework supports data processing from NetCDF (`.nc`) files, feature extraction, spatiotemporal window construction, model training, and comprehensive evaluation against baseline models.

---

## 📌 Project Overview & Model Scope

Oceanographic parameters exhibit complex spatiotemporal dynamics, non-linear physical ocean behaviors, and missing values due to cloud coverage or land masks. This project benchmarks five deep learning models representing the evolution of spatiotemporal modeling:

1. **LSTM (Baseline Temporal Model):** Models 1D pixel-wise time series independently without spatial dependencies.
2. **CNN-LSTM (Grid-Based Spatiotemporal):** Uses 2D CNN layers (`TimeDistributed(Conv2D)`) as spatial feature extractors feeding into an LSTM sequence learner.
3. **ConvLSTM (Pure Spatiotemporal Convolutional):** Replaces internal LSTM matrix multiplications with 2D convolutional operations, maintaining 2D spatial structure directly within recurrent state transitions.
4. **Attention-based ConvLSTM (SOTA Spatiotemporal):** Integrates spatial and channel attention mechanisms into ConvLSTM units to focus dynamically on high-variance ocean features (e.g., mesoscale eddies in ADT, coastal plume blooms in Chl-a).
5. **GCN-LSTM (Graph Relational Model):** Reconstructs the ocean grid into a graph structure (nodes = sea points, edges = spatial proximity/connectivity) to handle irregular boundaries and coastlines without computing on land masks.

---

## 🌊 Target Parameters

- **Absolute Dynamic Topography (ADT) [m]:** Measures sea surface height anomalies; critical for mapping geostrophic currents, mesoscale eddies, and oceanic circulation patterns.
- **Chlorophyll-a (Chl-a) [mg/m^3]:** Measures marine phytoplankton concentration; exhibits log-normal distributions and high spatial variability due to river runoff and upwelling.

---

## 🛠️ Architecture Comparison Matrix

| Model | Input Tensor Representation | Spatial Processing | Primary Advantage |
| :--- | :--- | :--- | :--- |
| **Persistence** | $(N, \text{Spatial})$ | None | Baseline benchmarking standard |
| **LSTM** | $(N, \text{Time}, \text{Spatial})$ | None (Flattened Grid) | Fast computation, simple temporal tracking |
| **CNN-LSTM** | $(N, \text{Time}, \text{Lat}, \text{Lon}, 1)$ | 2D CNN Feature Extraction | Captures local spatial feature maps per timestep |
| **ConvLSTM** | $(N, \text{Time}, \text{Lat}, \text{Lon}, 1)$ | In-cell 2D Convolutions | Preserves full 2D spatial structures across time |
| **Attention-ConvLSTM** | $(N, \text{Time}, \text{Lat}, \text{Lon}, 1)$ | Spatial & Channel Attention | Dynamically weights critical eddy & bloom regions |
| **GCN-LSTM** | $(V, E, \text{Time})$ | Graph Convolutions ($A$) | Computes only over ocean nodes, avoiding land masks |

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                  # Place raw .nc files (ADT.nc, CHL.nc) adding dummy data 
│   └── processed/            # Processed Numpy tensors & adjacency matrices
├── models/
│   ├── lstm.py               # Standard LSTM implementation
│   ├── cnn_lstm.py           # TimeDistributed CNN-LSTM implementation
│   ├── convlstm.py           # ConvLSTM2D network
│   ├── attention_convlstm.py # Attention-gated ConvLSTM network
│   └── gcn_lstm.py           # Graph Convolutional LSTM pipeline
├── utils/
│   ├── preprocessing.py      # NetCDF loading, NaN filling, and log-scaling
│   ├── graph_builder.py      # Adjacency matrix generation for GCN
│   └── visualization.py      # Study area maps, heatmaps, & loss curves
├── benchmark_runner.py       # Main scheduler & benchmarking execution script
├── README.md
└── requirements.txt
