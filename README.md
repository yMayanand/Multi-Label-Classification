# Multi-Label-Classification 🏄
Mutli-Label Classification is a task where there are multiple labels to classify for a single image
### 📌 Key changes we apply in Multi-label task as compared to Multi-class : -
- We change our targets from one-hot to Multi-hot encoding as there will be multiple true labels eg: [0 1 1 0 0 0 1 0]
- We change the loss function from `Cross-Entropy-Loss` to `Binary-Cross-Entropy-Loss` because former is designed to classify only one true label whereas 
  later allows multiple true labels to be classifed.

