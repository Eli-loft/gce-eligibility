# GCE Eligibility Classifier

Binary classification pipeline predicting NYC Good Cause Eviction (GCE) law eligibility across 761,000+ residential parcels from MapPLUTO 25v3. Built as an ML final project at the University of Tennessee at Chattanooga and deployed to Azure Databricks with MLflow experiment tracking.

## Results

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.9733 | 0.7788 | 0.6519 | 0.9671 |
| Decision Tree | 0.9768 | 0.8040 | 0.6824 | 0.9782 |
| MLP (threshold 0.8) | 0.9875 | 0.8812 | 0.8197 | 0.9527 |

## MLflow Experiment (Azure Databricks)

12 runs logged across all three notebooks - logistic regression, decision tree, 8 MLP threshold sweep runs, and final MLP evaluation.

![MLflow Runs](mlflow_runs.png)

## Pipeline

Three notebooks run in sequence on Azure Databricks (Runtime 13.3 LTS, Spark 3.4, single-node cluster):

`01_data_prep` loads the CSV via Spark, applies feature selection, one-hot encodes borough, fits StandardScaler on continuous features, and writes stratified train/val/test splits (70/15/15) to a Unity Catalog Volume.

`02_baselines` loads the splits, trains logistic regression and decision tree classifiers, evaluates on the test set, and logs params/metrics/models to the `/gce_eligibility` MLflow experiment.

`03_mlp` loads `best_model.pth` (trained locally on RTX 3060 Ti, CUDA 12.1) into a CPU-only PyTorch context, runs a threshold sweep across 8 values (0.2 to 0.8), and logs each run to MLflow. Final evaluation at threshold 0.8.

## Model Architecture

```
GCE_MLP(
  Linear(14, 128) -> ReLU -> Dropout(0.3)
  Linear(128, 64) -> ReLU -> Dropout(0.3)
  Linear(64, 32)  -> ReLU
  Linear(32, 1)
)
Total parameters: 12,289
```

Training: Adam (lr=1e-3), BCEWithLogitsLoss with pos_weight=19.55 (class imbalance 1:19), batch_size=2048, early stopping patience=5, ReduceLROnPlateau scheduler. Trained for 24 epochs.

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud platform | Azure Databricks (13.3 LTS) |
| Distributed compute | Apache Spark 3.4, PySpark |
| Experiment tracking | MLflow |
| ML framework | PyTorch, scikit-learn |
| Data processing | pandas, NumPy, StandardScaler |
| Storage | Unity Catalog Volumes |

## Dataset

NYC MapPLUTO 25v3 - 766,941 residential parcels, filtered to 761,620 after null removal. Features: NumFloors, BldgArea, ResArea, LotArea, AssessLand, AssessTot, ExemptTot, NumBldgs, Latitude, Longitude, Borough (one-hot). Target: GCE eligibility flag (UnitsRes >= 3, YearBuilt <= 1974 were withheld from features to prevent leakage).

Class distribution: 4.87% eligible (1:19 imbalance).
