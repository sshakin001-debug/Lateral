# Lateral Project

A lane detection and 3D object recognition project based on Lateral-SOTA.

## Project Structure

```
Lateral/
├── .venv/                  ← Virtual environment (hidden)
├── .git/                   ← Git repository (root level)
├── src/                    ← Source code
│   ├── lateral_sota/       ← Lateral-SOTA package
│   │   ├── __init__.py
│   │   ├── lane_detection.py
│   │   ├── matching.py
│   │   └── ...
│   └── my_modules/         ← Custom modules
│       ├── __init__.py
│       └── enhancements.py
├── weights/                ← Pretrained models
├── data/                   ← Dataset (symlink to KaggleHub)
├── notebooks/              ← Jupyter notebooks
├── scripts/                ← Setup/download scripts
├── configs/                ← Configuration files
├── tests/                  ← Unit tests
├── requirements.txt        ← Dependencies
├── setup.py               ← Package installation
└── README.md               ← This file
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Weights

```bash
python scripts/download_weights.py
```

### 4. Setup Dataset

```bash
python scripts/setup_dataset.py
```

### 5. Install Package (Optional)

```bash
pip install -e .
```

## Usage

### Running original pipeline:

```bash
cd src/lateral_sota
python 1_Lane_2D.py
```

### Importing in your code:

```python
from lateral_sota.ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector
from lateral_sota.ransacPlaneobject import ransacPlaneobject
```

## Notes

- The `data/` directory contains a symlink to the KaggleHub cache
- Weights are stored in `weights/lane_detection/`
- Custom modules go in `src/my_modules/`
