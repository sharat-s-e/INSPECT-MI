# INSPECT-MI
==============================
🧠 INSPECT-MI Framework
INSPECT-MI evaluates bias across three complementary dimensions:
Predictive Performance
→ Subgroup-wise metrics (F1, AUROC, confidence)
Fairness
→ Subgroup disparities (absolute differences)
Interpretability
→ Occlusion-based saliency and TSS

------------

## Project Organization
INSPECT-MI/
│── src/
│ │── evaluate.py # Main evaluation pipeline
│ │── models/
│ │ ├── densenet3d.py # DenseNet3D architecture
│ │ ├── resnet3d.py # ResNet3D architecture
│ │ └── cnn3d.py # Custom CNN model
│ └── utils/
│   ├── data_loader.py # Dataset loading utilities
│   ├── data_utils.py # Preprocessing helpers
│   └── visualize.py # Visualization and plotting
│── data/
│ ├── scanlist.csv # scanlist file contatining demographic information
│ ├── test_data # test dataset
│ └── models # test and benchmark models
│
│── configs/
│ └── config.yaml # Configuration file
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
└── README.md

## ▶️ How to Use
1. Configure settings - Edit configs/config.yaml
2. Run evaluation

## 📊 Outputs
The framework generates:
✅ Performance Metrics
Subgroup-wise metrics (F1 score, AUROC, confidence)
Worst-performing subgroup identification
⚖️ Bias Metrics
Absolute subgroup differences for each metric
🧠 Interpretability
ROI-wise saliency scores (hippocampus, amygdala)
Total Saliency Score (TSS)
🔁 Benchmark Comparison (if provided)
Performance degradation vs benchmark
Bias difference vs benchmark
TSS comparison

## 📌 Notes
Saliency is computed using ROI occlusion based on FastSurfer segmentations.
TSS reflects model reliance on disease-relevant neuroanatomy.
Framework is designed for binary classification (AD vs HC) but can be extended.

## 🔧 Extending the Framework
You can:
Add new models in models/
Extend metrics in evaluate.py
Add new ROIs for saliency analysis
Integrate additional demographic variables

## 📝 License
For academic and non-commercial use only. Contact the author for collaboration or reuse.