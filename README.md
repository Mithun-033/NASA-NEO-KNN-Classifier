
# NASA-NEO-KNN-Classifier

A machine learning project using the **NASA Near-Earth Objects dataset from Kaggle** and **scikit-learn’s K-Nearest Neighbors (KNN)** algorithm to classify potentially hazardous asteroids. The pipeline preprocesses data, scales features, evaluates accuracy across multiple K values, and identifies the optimal K for the best model performance.

---

## 🚀 Overview

This project:

- Loads and cleans the NASA NEO dataset  
- Applies feature scaling using `StandardScaler`  
- Splits data into training and testing sets  
- Trains a KNN classifier for K = 1 to 99  
- Plots accuracy vs. K  
- Prints the best K value and corresponding accuracy  

---

## 🧠 Features Used

- `est_diameter_min`  
- `est_diameter_max`  
- `miss_distance`  
- `relative_velocity`  
- `absolute_magnitude`

---

## 📦 Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib

