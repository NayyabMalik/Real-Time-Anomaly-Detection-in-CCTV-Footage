# Real-Time Anomaly Detection in CCTV Footage

## Overview

This project implements real-time anomaly detection in CCTV footage using deep learning techniques. It evaluates multiple neural network architectures (CNN, CNN+RNN, CNN+LSTM) to detect anomalies in video data, focusing on performance, overfitting, and generalization using the UCF-Crime dataset.

## Dataset

- **Source**: UCF-Crime Dataset
- **Content**: Videos categorized into classes like Abuse, Arrest, Explosion, Road Accidents, and Normal Videos
- **Splits**:
  - Training: \~900,000 samples
  - Validation: \~250,000 samples
  - Testing: \~80,000 samples
- **Challenge**: Significant class imbalance, with Normal Videos comprising &gt;75% of the data

## Models

Three architectures were implemented:

- **CNN**: Extracts spatial features using convolutional layers with 16, 32, and 64 filters, followed by dense layers for classification.
- **CNN+RNN**: Combines CNN for spatial feature extraction with RNN (LSTM with 64 units) for temporal dependencies.
- **CNN+LSTM**: Uses CNN for spatial features and LSTM for temporal dependencies, incorporating dropout and regularization to reduce overfitting.

## Methodology

- **Preprocessing**: Video frames processed for spatial and temporal feature extraction.
- **Training**: Models trained on the UCF-Crime dataset with Adam optimizer, categorical cross-entropy loss, and early stopping to prevent overfitting.
- **Evaluation**: Performance assessed using training/validation accuracy and loss, with classification reports for generalization analysis.

## Results

- **CNN**: Moderate training accuracy, fluctuating validation accuracy, partial overfitting due to class imbalance.
- **CNN+RNN**: High training accuracy (90.60%), low validation accuracy (52.60%), significant overfitting.
- **CNN+LSTM**: Stable high training and validation accuracy, low losses, best generalization among models.

## Critical Analysis

- **CNN+RNN**: Overfits due to model complexity and imbalanced data.
- **CNN**: Struggles with underrepresented classes, leading to unstable validation performance.
- **CNN+LSTM**: Best performer with consistent metrics, effective for sequential data.
- **Challenge**: Class imbalance affects detection of rare anomalies (e.g., Shooting, Vandalism).

## Dependencies

- Python 3.x
- Libraries: `tensorflow`, `keras`, `numpy`, `pandas`
- Install via:

  ```bash
  pip install tensorflow numpy pandas
  ```

## Usage

1. Download the UCF-Crime dataset.
2. Place the dataset in the project directory.
3. Run the training script:

   ```bash
   python anomaly_detection.py
   ```
4. Review model performance via accuracy/loss plots and classification reports.

## Future Work

- Address class imbalance with data augmentation or synthetic data generation.
- Explore hyperparameter tuning and advanced regularization.
- Test additional architectures (e.g., 3D CNNs) for improved performance.

## 