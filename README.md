# INSPECT-MI
🧠 INSPECT-MI Framework
INSPECT-MI evaluates bias across three complementary dimensions:
- Predictive Performance
  → Subgroup-wise metrics (F1, AUROC, confidence)
- Fairness
  → Subgroup disparities (absolute differences)
- Interpretability
  → Occlusion-based saliency and TSS

------------

## Project Organization
```
INSPECT-MI/
│── src/
│ │── evaluate.py # Model evaluation pipeline
│ │── models/
│ │ ├── __init__.py # Wrapper for model selection
│ │ ├── densenet3d.py # DenseNet3D architecture
│ │ ├── resnet3d.py # ResNet3D architecture
│ │ └── cnn3d.py # Custom CNN model
│ └── utils/
│   ├── data_loader.py # Dataset loading utilities
│   ├── data_utils.py # Data preprocessing helpers
│   ├── gpu_utils.py # GPU utilization helpers
│   └── visualize.py # Visualization and plotting
│── data/
│ ├── scanlist.csv # Scanlist file containing demographic information
│ ├── test_data # Test dataset
│ ├── seg_data # Segmentation maps of test dataset
│ ├── models # Test model weights
│ └── output # Execution outputs
│── configs/
│ ├── config.yaml # Configuration file template for execution
│ └── fastsurfer_atlas_map.csv # Fastsurfer atlas index-location mapping file
├── requirements.txt # The requirements file for reproducing the analysis environment
│── run_inspectmi.py # Main execution file
│── pyproject.toml
└── README.md
```

## ▶️ How to Use
1. Install requirements => pip install -r requirements.txt
2. Keep test data in data/test_data
3. Keep confromed segmentation maps in data/seg_data
4. Keep test model weights in data/models
5. Keep scanlist file in data/scanlist.csv
6. Update configuration file - configs/config.yaml
7. Run evaluation => python run_inspectmi.py --config configs/config.yaml

## 📊 Outputs
The framework generates:
- ✅ Performance Metrics
  Subgroup-wise metrics (F1 score, AUROC, confidence)
  Worst-performing subgroup identification
- ⚖️ Bias Metrics
  Absolute subgroup differences for each metric
- 🧠 Interpretability
  ROI-wise saliency scores (hippocampus, amygdala)
  Total Saliency Score (TSS)

## 📌 Notes
- Saliency is computed using ROI occlusion based on FastSurfer segmentations.
- TSS reflects model reliance on disease-relevant neuroanatomy.
- Framework is designed for binary classification (AD vs HC) but can be extended.

## 🔧 Extending the Framework
You can:
- Add new models in models/
- Extend evaluation metrics in evaluate.py
- Add new ROIs for saliency analysis in config.yaml
- Integrate additional demographic variables in config.yaml

## 📝 License
For academic and non-commercial use only. Contact the author for collaboration or reuse.
