import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        distances = self._distances(X= X) # Distance matrix (Each test point to each training point).
        neighbor_indices = np.argpartition(distances, self.k - 1, axis= 1)[:, :self.k] # Indices of the k nearest neighbors for each test point.
        neighbor_labels = self.y_train[neighbor_indices] # Labels of the k nearest neighbors for each test point.
        y_prediction = self._vote(neighbor_labels= neighbor_labels) # Predict the label according to the majority.
        return y_prediction

        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


    def _distances(self, X:np.ndarray) -> np.ndarray:
        """
        Computes the Minkowski distance matrix between test set X  and training set self.X_train.

        :param X: Test set
        :return: Distance matrix
        """
        diffs = X[:, None, :] - self.X_train[None, :, :] # Reshape the vectors in order to subtract them.
        absVal = np.abs(diffs) # Take the absolute value of each difference component.
        powered = absVal ** self.p # Raise the absolute value to the power of p.
        summed = np.sum(powered, axis= 2) # Sum over the feature axis.
        return summed ** (1 / self.p) # Return the p-th root.
    
    def _vote(self, neighbor_labels: np.ndarray) -> np.ndarray:
        """
        Performs majority voting using neighbor labels.
        This method uses one-hot encoding to count the number of times each class
        appears among the k neighbors, then selects the most frequent one.

        :param neighbor_labels: Labels of nearest neighbors.
        :return: Predicted labels.
        """
        num_labels = np.max(self.y_train) + 1  # Total number of classes
        one_hot = np.eye(num_labels)[neighbor_labels]  # Convert labels to one-hot vectors for voting
        vote_count = np.sum(one_hot, axis=1)  # Count votes
        winner = np.argmax(vote_count, axis=1)  # Choose class with most votes
        return winner.astype(np.uint8)  # Return the "winner" labels as uint8

def main():

    print("*" * 20)
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")  
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)

if __name__ == "__main__":
    main()
