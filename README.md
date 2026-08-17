# A pipeline for gap filling of landsat bands in the upper ewaso nyiro river basin, Kenya.

---

## Folder Structure


```
\ls-temporal-gap-filling
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/                              # ALL EXPERIMENT DATA MODALITIES
│   ├── landsat/                        # Unprocessed Landsat bands (R,G,B,NIR,SWIR1,SWIR2,Thermal)
│   ├── avhrr/                          # avhrr ndvi for missing years
│   ├── modis/                          # modis ndvi/ modis bands
│   ├── static/                         # Landscape invariant -DEM
│   ├── era5/ 
│         ├── precip
│         └── temp/            # Landscape invariant -DEM
│   └── processed/                      # Output matrix cache, predicted GeoTIFFs, reports
│         └── eda/                      # Exploratory visual plots, heatmaps, summary metrics
│
├── notebooks/                          # STRICTLY EDA & EXPLORATORY VISUALIZATIONS
│   ├── temporal_gaps.py                # Visual gap analysis, heatmaps, missingness breakdown
│   └──  extract_gap_months.py
└── src/                               # MAIN SOURCE CODE
    ├── config.yaml                     # Centralized pipeline config (paths, hyperparams, bands, seeds)
    ├── train.py                        # Root execution orchestrator 
    ├── predict.py                      # Out-of-sample historical gap filler 
    │
    ├── logs/                           # Slurm execution stdout/stderr
    │
    ├── preprocessing/                 # DATA ENGINEERING & RASTER PROCESSING
    │   ├── dataset.py                  # PyTorch Dataset/DataLoader for RBFN streaming
    │   ├── raster_processor.py         # Resampling, scale standardization (0-1), 3D-to-2D matrix flattening
    │   └── indices.py                  # Downstream eval indices
    │
    ├── models/                        # MODEL ARCHITECTURES & TRAINING PIPELINES
    │   ├── run_rbfn.sh                 # Slurm HPC script for PyTorch RBFN training
    │   ├── train_rbfn.py               # Training loop, loss tracking, and metrics for RBFN
    │   ├── tuning.py
    │   └── rbfn.py                     # Multi-Output RBFN PyTorch Architecture (Gaussian Kernels)
    │    
    └── utils/                         # UTILITIES & GEOSPATIAL HELPERS
        ├── spatial.py                  # Spatial windowing, coordinate encoding, spatial weight matrices
        └── metrics.py                  # RMSE/SSIM/error scoring

```
The required bands/Target bands are Red, Green, Blue, NIR, SWIR(1&2) and Thermal 