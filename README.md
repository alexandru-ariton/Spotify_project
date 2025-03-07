# 🎵 Spotify Clustering & PCA Analysis

A **data analysis project** that explores **Spotify 2023 dataset** using **Hierarchical Clustering** and **Principal Component Analysis (PCA)**.

This project applies **machine learning techniques** to analyze and visualize **patterns in music data**.

---

## 📌 Project Overview

- **Dataset**: Spotify 2023  
- **Programming Language**: Python  
- **Libraries Used**: `pandas`, `matplotlib`, `seaborn`, `sklearn`, `scipy`, `numpy`  
- **Key Techniques**:  
  - Hierarchical Clustering 📊  
  - Principal Component Analysis (PCA) 🔍  
  - Data Preprocessing & Feature Scaling  

---

## 🚀 Features

✅ **Data Preprocessing** – Handling missing values and scaling features.  
✅ **Hierarchical Clustering** – Identifies natural groupings in the dataset.  
✅ **Silhouette Score Optimization** – Determines the best number of clusters.  
✅ **PCA Visualization** – Reduces dimensionality and visualizes data in 2D.  
✅ **Graphical Analysis** – Generates dendrograms, scatter plots, and PCA biplots.  

---

## 🏗️ System Architecture

This project is structured as follows:  

📂 **main.py** – Main script that runs the entire pipeline.  
📂 **functii.py** – Helper functions for data preprocessing.  
📂 **grafice.py** – Functions for plotting dendrograms, PCA biplots, and silhouette scores.  
📂 **spotify-2023.csv** – Dataset containing Spotify track information.  

---

## 📥 Installation & Setup

### 🔹 **Prerequisites**
Ensure you have the following installed:  

```sh
pip install pandas matplotlib seaborn scikit-learn scipy numpy
```

### 🔹 **Running the Project**
1️⃣ Clone the repository:  
```sh
git clone https://github.com/your-repo/spotify-clustering.git
```
2️⃣ Navigate to the project directory:  
```sh
cd spotify-clustering
```
3️⃣ Run the analysis:  
```sh
python main.py
```

---

## 📊 Data Flow & Analysis

### **📌 Data Preprocessing**
- Converts categorical features to numerical values.  
- Handles missing values using **mean imputation** for numerical data.  
- Scales numerical data for clustering and PCA.  

### **📍 Hierarchical Clustering Workflow**
1. Compute pairwise distances using **Ward's method**.  
2. Generate a **dendrogram** to visualize hierarchical relationships.  
3. Determine the optimal number of clusters using the **silhouette score**.  
4. Assign each song to a **cluster**.  

### **🎶 PCA for Dimensionality Reduction**
- Reduces data to **two principal components**.  
- Visualizes clusters in **2D space**.  
- Generates a **scree plot** to analyze explained variance.  
- Creates a **biplot** to interpret feature contributions.  

---

## 📊 Visualization & Outputs

📌 **Dendrogram** – Displays hierarchical relationships between clusters.  
📌 **PCA Scatter Plot** – Projects data onto two principal components.  
📌 **Silhouette Score Analysis** – Evaluates clustering quality.  
📌 **Scree Plot** – Shows the proportion of variance explained by each component.  

---

## 🎯 Future Enhancements

🔮 **Next Steps:**  
- Experiment with **K-Means & DBSCAN** clustering.  
- Incorporate **Spotify API** to enrich dataset.  
- Use **Deep Learning for Music Classification**.  

---
 
👥 Contributors:  
- **Ariton Alexandru**  

---
