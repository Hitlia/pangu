# Pangu-Weather

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.8.0%2B-red)](https://pytorch.org/)

This is a PyTorch implementation of the Pangu-weather model based on Huawei's paper "Accurate medium-range global weather forecasting with 3D neural networks". This is a 3D neural network model for medium-range global weather forecasting, trained and tested using ERA5 reanalysis data.

## Model Overview

Pangu-Weather is an advanced global weather forecasting model that employs a 3D neural network architecture, capable of providing accurate medium-range weather forecasts. The model processes multiple pressure level and surface meteorological variables, achieving performance comparable to traditional numerical weather prediction methods.

## Requirements

- **Python**: 3.9+
- **PyTorch**: 2.8.0+
- **GPU**: NVIDIA RTX 3090 (or compatible CUDA device)
- **Memory**: Recommended 32GB or more
- **Storage**: Sufficient space for ERA5 dataset and model files

## Installation

Clone this repository:
```bash
git clone https://github.com/Hitlia/pangu.git
cd pangu
```

## Data Preparation

### ERA5 Dataset
The model uses ERA5 reanalysis data from 2018-2024, containing:

**13 Pressure Level Variables** (5 variables per level):
- t: Temperature
- q: Specific humidity
- z: Geopotential
- v: Meridional wind
- u: Zonal wind

**Surface Variables** (5 variables):
- t2m: 2-meter temperature
- d2m: 2-meter dewpoint temperature
- v10: 10-meter meridional wind
- u10: 10-meter zonal wind
- tp6hr: 6-hour total precipitation (calculated from ERA5 data, adjusted to mm/6h)

**Total**: 13 × 5 + 5 = 70 channels

### Constant Masks
The project includes pre-computed constant mask files located in the `constant_mask/` directory:
- `land_mask`: Land-sea mask
- `soil_type`: Soil type classification
- `topography`: Topographic data

## Project Structure

```
pangu-weather/
├── train.py              # Pre-training script
├── finetune.py           # Fine-tuning script
├── test.py               # Testing script
├── params.py             # Model parameter configuration
├── pangu.py              # Main model definition
├── weather_dataset.py    # Dataset construction
├── norms.py              # Training data statistics calculation
├── normalize.py          # CPU normalization calculation
├── constant_mask/        # Constant mask files
│   ├── land_mask
│   ├── soil_type
│   └── topography
└── utils/                # Basic modules
    ├── __init__.py
    ├── modules.py        # Model components
    └── ...
```

## Usage

### 1. Pre-training
```bash
python train.py
```
**Forecast Length**: 4 steps (24-hour forecast)

### 2. Fine-tuning
```bash
python finetune.py
```
**Forecast Length**: 8 steps (48-hour forecast)

### 3. Testing
```bash
python test.py
```
**Forecast Length**: 8 steps (48-hour forecast)

## Data Normalization

The model employs two normalization strategies:

1. **Min-Max Normalization** (for non-negative variables):
   - Specific humidity (q)
   - 6-hour total precipitation (tp6hr)

2. **Z-score Normalization** (for other variables):
   - Temperature, geopotential, wind speed, etc.

**Note**: Constant masks are not normalized.

## Loss Function

The model uses MAE (Mean Absolute Error) loss function with different weights for different variables:

- **Upper-level variables**: 0.001 × pressure value
- **Surface winds** (v10, u10): 0.1
- **Surface temperatures** (t2m, d2m) and **precipitation** (tp6hr): 1.0

All weights are normalized to sum to 1.0.

## Model Configuration

Main parameters are configured in `params.py`, including:
- Model dimensions
- Training parameters
- Data paths
- Hyperparameter settings

## Performance Characteristics

- **Input**: 70 channels of meteorological data
- **Output**: Weather forecasts for multiple future time steps
- **Temporal Resolution**: 6 hours
- **Pre-training Forecast Length**: 4 steps (24-hour forecast)
- **Fine-tuning/Testing Forecast Length**: 8 steps (48-hour forecast)
- **Spatial Resolution**: 0.25 degree
- **Forecast Range**: Medium-range weather forecasting

## Time Step Specification

The model uses a 6-hour time step with different forecast lengths for each phase:
- **Pre-training phase**: Predicts 4 future time steps, corresponding to 24-hour forecast
- **Fine-tuning and testing phase**: Predicts 8 future time steps, corresponding to 48-hour forecast

## Citation

If you use this project, please cite the original paper:

```bibtex
@article{bi2023accurate,
  title={Accurate medium-range global weather forecasting with 3D neural networks},
  author={Bi, Kaifeng and Xie, Lingxi and Zhang, Heng and Chen, Xin and Gu, Xiaotao and Tian, Qi},
  journal={Nature},
  year={2023},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Thanks to Huawei for the original Pangu-Weather model
- Thanks to ECMWF for providing ERA5 reanalysis data

---
