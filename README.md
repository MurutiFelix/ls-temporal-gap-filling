# A pipeline for gap filling of landsat bands in the upper ewaso nyiro river basin, Kenya.

---

## Folder Structure


```
D:\ls-temporal-gap-filling
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/                              # ALL EXPERIMENT DATA MODALITIES
│   ├── raw/                            # Unprocessed Landsat, MODIS, AVHRR (.tif/.csv), ERA5
│   ├── static/                         # Landscape invariants (DEM, Elevation, Aspect, Distance to Water)
│   └── processed/                      # Output matrix cache, predicted GeoTIFFs, reports
│       └── eda/                        # Exploratory visual plots, heatmaps, summary metrics
│
├── notebooks/                          # STRICTLY EDA & EXPLORATORY VISUALIZATIONS
│   └── Temporal_gaps.py                # Visual gap analysis, heatmaps, missingness breakdown
│
└── src/                               # MAIN SOURCE CODE
    ├── config.yaml                     # Centralized pipeline config (paths, hyperparams, bands, seeds)
    ├── train.py                        # Root execution orchestrator (trains RBFN & baselines)
    ├── predict.py                      # Out-of-sample historical gap filler (1995-1999 inference)
    │
    ├── logs/                           # Automated cluster logs directory (Slurm execution stdout/stderr)
    │
    ├── data/                           # DATA ENGINEERING & RASTER PROCESSING
    │   ├── dataset.py                  # PyTorch Dataset/DataLoader for RBFN streaming
    │   ├── raster_processor.py         # Resampling, scale standardization (0-1), 3D-to-2D matrix flattening
    │   └── analyze_and_tune.py         # Validation splitting (18/5 holdout), hyperparam tuning, gap metrics
    │
    ├── models/                         # MODEL ARCHITECTURES & TRAINING PIPELINES
    │   ├── run_rbfn.sh                 # Slurm HPC script for PyTorch RBFN training
    │   ├── train_dl.py                 # Training loop, loss tracking, and metrics for RBFN
    │   └── rbfn.py                     # Multi-Output RBFN PyTorch Architecture (Gaussian Kernels)
    │    
    └── utils/                          # UTILITIES & GEOSPATIAL HELPERS
        ├── spatial.py                  # Spatial windowing, coordinate encoding, spatial weight matrices
        └── metrics.py                  # Downstream spectral index solvers (VHI, BSI, SMI) & RMSE/SSIM

```
The required bands/Target bands are Red, Green, Blue, NIR, SWIR and Thermal 