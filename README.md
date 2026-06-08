
# EE656 SOC Course Project

Implementation and analysis of a research-paper-based clustering project for **EE656: Artificial Intelligence, Machine Learning, Deep Learning & Its Applications** at IIT Kanpur.

The project focuses on **Self-Organizing Clustering (SOC)** and related clustering methods, with emphasis on density-based behaviour, border detection, cluster validity, and visual comparison of clustering performance across different data geometries.

## Project Overview

Clustering is an important unsupervised learning task where the goal is to group similar data points without using labelled outputs. However, different clustering methods perform differently depending on the shape, density, overlap, and distribution of the data.

This project studies and implements clustering workflows inspired by a research paper on Self-Organizing Clustering and compares its behaviour with standard clustering methods.

## Key Objectives

- Study a research paper related to **Self-Organizing Clustering**
- Implement clustering algorithms and reproduce experimental workflows
- Compare clustering behaviour across different data geometries
- Generate visual outputs for cluster assignments and validity indices
- Understand the strengths and limitations of different clustering methods

## Methods Explored

The project includes experiments and visualizations related to:

- **K-Means Clustering**
- **DBSCAN**
- **Fuzzy C-Means**
- **Mountain Clustering**
- **Modified Mountain Clustering**
- **Self-Organizing Clustering**
- **Cluster Validity Indices**
- **Generalized Separation Index / clustering quality analysis**

## Tech Stack

- **Python**
- **MATLAB**
- **NumPy**
- **Matplotlib**
- Clustering and visualization scripts

## Repository Structure

```text
EE656_SOC_course-project/
│
├── EE656 Codes/
│   ├── SOC_Project.py
│   ├── Figure 1-Preferred Clustering Methods for Various Data Geometries.py
│   ├── Figure 2 and 3.py
│   └── output figures
│
├── SOC/
│   └── MATLAB / SOC-related implementation files
│
├── project_outputs/
│   ├── cluster_validity_indices_plot.png
│   ├── clustering_and_gsi_plot.png
│   ├── dbscan_preferred_shapes.png
│   ├── fuzzy_cmeans_preferred_shapes.png
│   ├── imc_preferred_shapes.png
│   ├── kmeans_preferred_shapes.png
│   ├── mmc_preferred_shapes.png
│   └── mountain_clustering_preferred_shapes.png
│
├── EE656_SOC_Clustering_compressed.pdf
├── SOC_Paper.pdf
├── SOC_Presentation.pdf
└── README.md
````

## Outputs

The project produces visual comparisons showing which clustering methods are better suited for different data shapes and distributions. The outputs include:

* Preferred clustering methods for different data geometries
* Cluster validity index plots
* Generalized separation / clustering quality visualizations
* Method-wise clustering behaviour plots

## Learning Outcomes

Through this project, I gained practical exposure to:

* Research paper implementation
* Unsupervised machine learning
* Clustering algorithm comparison
* Density-based clustering behaviour
* Cluster validity evaluation
* Python and MATLAB based experimentation
* Visual analysis of ML algorithms

## Course Context

This project was completed as part of:

**EE656 -- Artificial Intelligence, Machine Learning, Deep Learning & Its Applications**
Indian Institute of Technology Kanpur


```

