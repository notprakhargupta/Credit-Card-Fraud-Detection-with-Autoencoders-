# Credit-Card-Fraud-Detection-with-Autoencoders-
• Implemented a fraud detection system leveraging Autoencoder neural networks in Keras and TensorFlow, leveraging an architecture with 4 layers to reconstruct normal transactions and identify anomalies.

This repository contains code and resources for detecting credit card fraud using Autoencoders implemented in Keras with TensorFlow. The model is trained to identify anomalies in credit card transactions based on their features.

## Project Overview

Credit card fraud detection is a critical application in financial security. This project leverages Autoencoders, a type of neural network, to identify anomalous transactions that might indicate fraudulent activity. The model is trained on normal transactions and is then used to detect anomalies in new transactions.

## Dataset

The dataset used for this project is a credit card transaction dataset available on [Kaggle](https://www.kaggle.com/datasets?search=credit+card+fraud). It includes:

- `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
- `Amount`: Transaction amount.
- `Class`: Target variable indicating if the transaction is fraudulent (1) or normal (0).
- PCA-transformed features for privacy reasons.

## Installation

### Prerequisites

- Python 3.x
- TensorFlow 1.2
- Keras 2.0.4
- Other libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

You can install the required libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow==1.2 keras==2.0.4
