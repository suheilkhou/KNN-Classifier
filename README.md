# k-NN Classifier in Python

This project implements a simple k-Nearest Neighbors (k-NN) classifier from scratch using NumPy and pandas.  
It was developed as part of a university assignment to demonstrate core machine learning concepts such as distance metrics and majority voting.

## Features

- Custom implementation of k-NN
- Supports Minkowski distances (parameter `p`)
- Efficient majority voting using one-hot encoding
- Command-line interface for ease of use

## Files Included

- `kNN_implementation.py`: The main Python script implementing the classifier.
- `example_dataset.csv`: An example CSV file for testing the classifier.
- `README.md`: Project description and usage instructions.

## Requirements

- Python 3.x
- NumPy
- pandas

## How to Run

```bash
python kNN_implementation.py example_dataset.csv 3 2
