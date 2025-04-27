import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from itertools import product
import concurrent.futures
import os
import gc

start_time = time.time()
num_cores = os.cpu_count()
print(f"Using {num_cores} CPU cores")

train_data = pd.read_csv('mnist_train_3px_cutoff_12x12_avg_pool.csv')
test_data = pd.read_csv('mnist_test_3px_cutoff_12x12_avg_pool.csv')

X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Only create the specific KNN classifiers we need
k_values = [1, 3, 5, 7, 21]
knn_models = {}
knn_predictions = {}

for k in k_values:
    model_name = f"KNN (k={k})"
    print(f"Training {model_name}...")
    
    # Create and train the model
    clf = KNeighborsClassifier(n_neighbors=k, metric='correlation')
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    mistakes = np.sum(y_test != y_pred)
    
    # Store model and predictions
    knn_models[model_name] = clf
    knn_predictions[model_name] = y_pred
    
    print(f"{model_name}: Accuracy: {accuracy:.4f}, Mistakes: {mistakes}")
    gc.collect()

# Function to evaluate weighted ensemble vote
def evaluate_weight_combo(weight_combo):
    # Calculate weighted votes for each test sample
    combo_predictions = []
    for i in range(len(y_test)):
        vote_counts = {}
        for j, k in enumerate(k_values):
            model_name = f"KNN (k={k})"
            prediction = knn_predictions[model_name][i]
            weight = weight_combo[j]
            
            # Add weighted vote
            if prediction in vote_counts:
                vote_counts[prediction] += weight
            else:
                vote_counts[prediction] = weight
        
        # Get digit with highest weighted vote
        ensemble_pred = max(vote_counts.items(), key=lambda x: x[1])[0]
        combo_predictions.append(ensemble_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, combo_predictions)
    mistakes = np.sum(y_test != combo_predictions)
    
    # Create a descriptive name
    combo_name = " + ".join([f"{w}×(k={k})" for w, k in zip(weight_combo, k_values)])
    
    return (combo_name, accuracy, mistakes, weight_combo)

# Generate all weight combinations (1-10 for each classifier)
weight_range = range(1, 11)  # 1 to 10 inclusive
all_weight_combos = list(product(weight_range, repeat=5))
total_combos = len(all_weight_combos)
print(f"\nTesting {total_combos} weight combinations for ensemble (k=1, k=3, k=5, k=7, k=21)")

# Process weight combinations in parallel
results = []
best_combo = None
best_accuracy = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
    batch_size = 5000  # Process in batches to manage memory
    for i in range(0, total_combos, batch_size):
        batch = all_weight_combos[i:i+batch_size]
        batch_end = min(i+batch_size, total_combos)
        print(f"Processing combinations {i+1}-{batch_end} of {total_combos}...")
        
        # Submit batch of weight combinations for processing
        futures = [executor.submit(evaluate_weight_combo, combo) for combo in batch]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            combo_name, accuracy, mistakes, weights = future.result()
            results.append((combo_name, accuracy, mistakes, weights))
            
            # Track best combination
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combo = (combo_name, accuracy, mistakes, weights)
        
        # Print progress update with current best
        if best_combo:
            print(f"Current best: {best_combo[0]}")
            print(f"Accuracy: {best_combo[1]:.4f}, Mistakes: {best_combo[2]}")
        
        gc.collect()

# Sort results by accuracy (highest first)
results.sort(key=lambda x: x[1], reverse=True)

# Display top 20 weight combinations
print("\n--- Top 20 Weight Combinations ---")
print("Weight Combination | Accuracy | Mistakes | Error Rate")
print("-" * 70)

for i, (combo_name, accuracy, mistakes, weights) in enumerate(results[:20], 1):
    error_rate = (mistakes / len(y_test)) * 100
    print(f"{i}. {combo_name} | {accuracy:.4f} | {mistakes} | {error_rate:.2f}%")

# Display the best weight combination
print("\n=== Best Weight Combination ===")
best_name, best_acc, best_mistakes, best_weights = results[0]
print(f"Weights: {best_weights}")
print(f"Accuracy: {best_acc:.6f}")
print(f"Mistakes: {best_mistakes}")
print(f"Error Rate: {(best_mistakes/len(y_test))*100:.4f}%")

# Save results to file
with open('weight_combination_results.txt', 'w') as f:
    f.write("=== Weight Combination Testing Results ===\n\n")
    f.write(f"Ensemble: KNN with k={k_values}\n")
    f.write(f"Total combinations tested: {total_combos}\n\n")
    
    f.write("=== Best Weight Combination ===\n")
    for k, weight in zip(k_values, best_weights):
        f.write(f"KNN (k={k}): {weight} votes\n")
    
    f.write(f"\nAccuracy: {best_acc:.6f}\n")
    f.write(f"Mistakes: {best_mistakes}\n")
    f.write(f"Error Rate: {(best_mistakes/len(y_test))*100:.4f}%\n\n")
    
    f.write("--- Top 20 Weight Combinations ---\n")
    for i, (combo_name, accuracy, mistakes, weights) in enumerate(results[:20], 1):
        error_rate = (mistakes / len(y_test)) * 100
        f.write(f"{i}. {combo_name}\n")
        f.write(f"   Accuracy: {accuracy:.6f}, Mistakes: {mistakes}, Error Rate: {error_rate:.2f}%\n")
        f.write(f"   Weights: {weights}\n\n")

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")