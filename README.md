# Performance Benchmarking of Deep Learning Architectures for Independent Forecasting of Satellite-Derived Ocean Parameters

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

Accurate multi-day forecasting of satellite-derived oceanographic variables is essential for maritime safety, climate modeling, and marine ecology tracking. Traditional numerical modeling approaches are computationally intensive, while standard sequence models (like 1D LSTMs) lose critical 2D spatial structures when processing satellite imagery.

This project implements a rigorous **spatiotemporal deep learning benchmarking framework** to evaluate **ConvLSTM** and **Attention-driven ConvLSTM (Att-ConvLSTM)** architectures. The models perform independent 15-day lead-time forecasts for two key satellite-derived ocean parameters across selected marine domains:
1. **Chlorophyll-a (Chl-a)** ($mg/m^3$) — A highly non-linear, log-normally distributed indicator of ocean biological activity.
2. **Absolute Dynamic Topography (ADT)** ($m$) — A key indicator of ocean circulation, sea surface height, and geostrophic currents.

---

## 🏗️ Architecture Benchmarks

This repository benchmarks spatial-temporal neural network paradigms against standard baseline models (e.g., Persistence):

1. **ConvLSTM (Convolutional Long Short-Term Memory):** Replaces internal matrix multiplications of standard LSTMs with 2D convolutional operations, allowing the network to simultaneously capture temporal dynamics and spatial features (such as eddies, fronts, and plumes).
2. **Attention-based ConvLSTM (Att-ConvLSTM):** Integrates spatial/channel attention mechanisms (e.g., CBAM/Self-Attention) into the ConvLSTM pipeline, enabling the model to dynamically focus on high-variability oceanographic structures (e.g., coastal upwelling and eddy propagation) over long forecast horizons.
3. **Persistence Model (Baseline):** A naive physical benchmark that projects the last observed time step ($t_0$) across all forecast horizons ($t+1 \dots t+15$).

---

## ⚙️ Key Technical Features

- **Data Ingestion & Masking:** Reads multi-variable NetCDF (`.nc`) files using `xarray`, handles land/ocean masks, and fills missing values using spatial-temporal interpolation.
- **Parametric Processing:**
  - **Chl-a Pipeline:** Log-transformation ($\ln(1 + x)$) to handle skewed, high-variability distributions, followed by Z-score normalization.
  - **ADT Pipeline:** Direct standard scaling or min-max normalization to preserve sea surface height anomaly gradients.
- **5D Sliding Window Tensors:** Automated construction of 5D input sequences of shape `(Samples, Time_Steps, Lat, Lon, Channels)`.
- **Geographical Visualization:** Integrated visualization scripts using `matplotlib` with custom land masks (`set_bad` color schemes) to render publication-grade study area maps.
- **Evaluation Engine:** Metric calculation ($RMSE$, $MAE$, and $R^2$ Score) computed across each day of a 15-day forecast horizon.

---

## 🛠️ Repository Structure

```text
├── data/                  # Sample or processed NetCDF datasets (CHL.nc, ADT.nc)
├── models/
│   ├── convlstm.py        # Standard 2D ConvLSTM implementation
│   └── att_convlstm.py    # Spatial/Channel Attention ConvLSTM architecture
├── utils/
│   ├── data_loader.py     # NetCDF loading, region slicing, and NaN handling
│   ├── preprocessing.py   # Normalization, log-scaling, and sliding window generators
│   └── metrics.py         # Per-day RMSE, MAE, and R2 calculation scripts
├── scripts/
│   ├── train_convlstm.py  # Script for training ConvLSTM models
│   └── plot_study_area.py # Plotting script for study area maps with land masking
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
