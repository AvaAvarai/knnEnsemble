import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import os
import gc

start_time = time.time()
num_cores = os.cpu_count()
print(f"Using {num_cores} CPU cores")

# Load the data
print("Loading data...")
data = pd.read_csv('mnist_train_3px_crop_11x11_avg_pool.csv')
test_data = pd.read_csv('mnist_test_3px_crop_11x11_avg_pool.csv')

# Extract features and labels
X = data.drop('label', axis=1)
y = data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Setup for 10-fold cross-validation with stratified sampling
n_splits = 10
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# KNN parameters
k_values = [1, 3, 5, 7, 21]
weights = [1, 1, 1, 2, 1]  # 1×(k=1) + 1×(k=3) + 1×(k=5) + 2×(k=7) + 1×(k=21)

# Lists to store metrics across folds
fold_accuracies = []
fold_confusion_matrices = []

print(f"Performing {n_splits}-fold cross-validation with stratified sampling...")
print(f"Using ensemble: 1×(k=1) + 1×(k=3) + 1×(k=5) + 2×(k=7) + 1×(k=21)")

fold_num = 1
# Cross-validation loop
for train_index, val_index in kf.split(X, y):
    print(f"\nFold {fold_num}/{n_splits}")
    
    # Split data for this fold
    X_train_fold = X.iloc[train_index]
    y_train_fold = y.iloc[train_index]
    X_val_fold = X.iloc[val_index]
    y_val_fold = y.iloc[val_index]
    
    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Train KNN models with different k values
    knn_models = {}
    knn_predictions = {}
    
    for k in k_values:
        model_name = f"KNN (k={k})"
        print(f"Training {model_name}...")
        
        # Create and train the model
        clf = KNeighborsClassifier(n_neighbors=k, metric='correlation')
        clf.fit(X_train_scaled, y_train_fold)
        
        # Make predictions on validation data
        y_pred = clf.predict(X_val_scaled)
        accuracy = accuracy_score(y_val_fold, y_pred)
        
        # Store model and predictions
        knn_models[model_name] = clf
        knn_predictions[model_name] = y_pred
        
        print(f"{model_name}: Validation Accuracy: {accuracy:.4f}")
        gc.collect()
    
    # Calculate ensemble predictions on validation data
    ensemble_predictions = []
    for i in range(len(y_val_fold)):
        vote_counts = {}
        for j, k in enumerate(k_values):
            model_name = f"KNN (k={k})"
            prediction = knn_predictions[model_name][i]
            weight = weights[j]
            
            # Add weighted vote
            if prediction in vote_counts:
                vote_counts[prediction] += weight
            else:
                vote_counts[prediction] = weight
        
        # Get digit with highest weighted vote
        ensemble_pred = max(vote_counts.items(), key=lambda x: x[1])[0]
        ensemble_predictions.append(ensemble_pred)
    
    # Calculate ensemble accuracy for this fold
    fold_accuracy = accuracy_score(y_val_fold, ensemble_predictions)
    fold_cm = confusion_matrix(y_val_fold, ensemble_predictions)
    
    fold_accuracies.append(fold_accuracy)
    fold_confusion_matrices.append(fold_cm)
    
    print(f"Fold {fold_num} Ensemble Validation Accuracy: {fold_accuracy:.4f}")
    fold_num += 1
    gc.collect()

# Calculate and display average metrics across folds
avg_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
avg_cm = np.mean(fold_confusion_matrices, axis=0)

print("\n=== Cross-Validation Results ===")
print(f"Ensemble: 1×(k=1) + 1×(k=3) + 1×(k=5) + 2×(k=7) + 1×(k=21)")
print(f"Average Validation Accuracy: {avg_accuracy:.6f} ± {std_accuracy:.6f}")
print(f"Individual Fold Validation Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")

# Now evaluate on the test data using a model trained on the entire training set
print("\n=== Training Final Model on Full Dataset ===")
# Scale the entire training data
full_scaler = MinMaxScaler()
X_train_full_scaled = full_scaler.fit_transform(X)
X_test_scaled = full_scaler.transform(X_test)

# Train final models using all training data
final_knn_models = {}
final_knn_predictions = {}

for k in k_values:
    model_name = f"KNN (k={k})"
    print(f"Training final {model_name} on full dataset...")
    
    # Train on all training data
    clf = KNeighborsClassifier(n_neighbors=k, metric='correlation')
    clf.fit(X_train_full_scaled, y)
    
    # Make predictions on test set
    y_pred = clf.predict(X_test_scaled)
    
    # Store model and predictions
    final_knn_models[model_name] = clf
    final_knn_predictions[model_name] = y_pred

# Calculate ensemble predictions on test set
test_ensemble_predictions = []
for i in range(len(y_test)):
    vote_counts = {}
    for j, k in enumerate(k_values):
        model_name = f"KNN (k={k})"
        prediction = final_knn_predictions[model_name][i]
        weight = weights[j]
        
        # Add weighted vote
        if prediction in vote_counts:
            vote_counts[prediction] += weight
        else:
            vote_counts[prediction] = weight
    
    # Get digit with highest weighted vote
    ensemble_pred = max(vote_counts.items(), key=lambda x: x[1])[0]
    test_ensemble_predictions.append(ensemble_pred)

# Calculate test metrics
test_accuracy = accuracy_score(y_test, test_ensemble_predictions)
test_mistakes = np.sum(y_test != test_ensemble_predictions)
test_cm = confusion_matrix(y_test, test_ensemble_predictions)

print(f"Test Set Accuracy: {test_accuracy:.6f}")
print(f"Test Set Mistakes: {test_mistakes}")
print(f"Test Set Error Rate: {(test_mistakes/len(y_test))*100:.4f}%")

# Save results to file
with open('cross_validation_results.txt', 'w') as f:
    f.write("=== Cross-Validation Results ===\n\n")
    f.write(f"Ensemble: 1×(k=1) + 1×(k=3) + 1×(k=5) + 2×(k=7) + 1×(k=21)\n")
    f.write(f"Number of folds: {n_splits}\n\n")
    
    f.write("=== Fold Results ===\n")
    for i, acc in enumerate(fold_accuracies, 1):
        f.write(f"Fold {i} Validation Accuracy: {acc:.6f}\n")
    
    f.write(f"\nAverage Validation Accuracy: {avg_accuracy:.6f} ± {std_accuracy:.6f}\n\n")
    
    f.write("=== Test Set Results ===\n")
    f.write(f"Accuracy: {test_accuracy:.6f}\n")
    f.write(f"Mistakes: {test_mistakes}\n")
    f.write(f"Error Rate: {(test_mistakes/len(y_test))*100:.4f}%\n")

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds") 