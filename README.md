# knnEnsemble

knnEnsemble for homogeneous data classification like MNIST, competitive with CNN architectures.

Dimensionality reduction: Using 2px and 3px cutoffs on the boundary of all train and eval data with kNN k=3 was same accuracy. We choose to use 3px for the smaller dimensionality of data. 28 x 28 (original size) = 22 x 22 (3px all edges cut off) = 11 x 11 (2x2 kernel 2 stride pooling.) 11x11=121-D / 784-D = 0.15433673469387755102040816326531 = 1 - 0.15433673469387755102040816326531 = 0.84566326530612244897959183673469 ~ 84.57% dimensional reduction applied to the data.

Results:

1.	We can use an ensemble of k-NN with majority vote, using singular votes per model in the ensemble we currently get: Best model: Ensemble (k=1) + (k=3) + (k=5) + (k=7) + (k=21) Accuracy: 0.9786, (k-fold confirm in progress) Mistakes: 214, second best ensemble is (k=1) + (k=3) + (k=5) + (k=7) + (k=15) Accuracy: 0.9783, Mistakes: 217.
![Mistakes in tSNE](best_model_errors_tsne.png)

2.	We can give classifiers multiple votes here is the simplest most successful model seen to date. Current best: 1×(k=1) + 1×(k=3) + 1×(k=5) + 2×(k=7) + 1×(k=21) Accuracy: 0.9789, Mistakes: 211, tied with many others like  2×(k=1) + 1×(k=3) + 1×(k=5) + 3×(k=7) + 1×(k=21) | 0.9789 | 211 | 2.11%.

## License Information

This project is licensed under the MIT License, allowing free use for both personal and commercial purposes. For full terms, see the `LICENSE` file.
