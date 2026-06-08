# This is code for the images generated in the report


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_kmeans_preferred_shapes():
    """K-Means prefers spherical, well-separated clusters"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Spherical clusters
    X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=1.0,
                        center_box=(-5.0, 5.0), random_state=42)
    axes[0].scatter(X1[:, 0], X1[:, 1], c=y1, cmap='viridis', alpha=0.7)
    axes[0].set_title('Spherical Clusters\n(Ideal for K-Means)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Compact globular clusters
    X2, y2 = make_blobs(n_samples=400, centers=3, cluster_std=0.8,
                        center_box=(-4.0, 4.0), random_state=123)
    axes[1].scatter(X2[:, 0], X2[:, 1], c=y2, cmap='plasma', alpha=0.7)
    axes[1].set_title('Compact Globular Clusters\n(K-Means Strength)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Equal-sized clusters
    centers = [(-2, -2), (2, -2), (0, 2)]
    X3, y3 = make_blobs(n_samples=300, centers=centers, cluster_std=1.0, random_state=0)
    axes[2].scatter(X3[:, 0], X3[:, 1], c=y3, cmap='coolwarm', alpha=0.7)
    axes[2].set_title('Equal-Sized Clusters\n(K-Means Assumption)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('K-Means: Preferred Cluster Shapes', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('kmeans_preferred_shapes.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_fuzzy_cmeans_preferred_shapes():
    """Fuzzy C-Means handles overlapping and soft boundaries"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Overlapping clusters
    np.random.seed(42)
    cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 150)
    cluster2 = np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 150)
    cluster3 = np.random.multivariate_normal([4, 2], [[0.8, 0.2], [0.2, 0.8]], 150)

    axes[0].scatter(cluster1[:, 0], cluster1[:, 1], alpha=0.6, label='Cluster 1', s=30)
    axes[0].scatter(cluster2[:, 0], cluster2[:, 1], alpha=0.6, label='Cluster 2', s=30)
    axes[0].scatter(cluster3[:, 0], cluster3[:, 1], alpha=0.6, label='Cluster 3', s=30)
    axes[0].set_title('Overlapping Clusters\n(Fuzzy C-Means Strength)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Soft boundaries
    X2, y2 = make_blobs(n_samples=400, centers=3, cluster_std=1.5,
                        center_box=(-3.0, 3.0), random_state=42)
    axes[1].scatter(X2[:, 0], X2[:, 1], c=y2, cmap='viridis', alpha=0.5, s=50)
    axes[1].set_title('Soft Boundaries\n(Membership Gradients)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Variable density clusters
    np.random.seed(123)
    dense_cluster = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 200)
    sparse_cluster = np.random.multivariate_normal([4, 4], [[2, 0], [0, 2]], 100)
    medium_cluster = np.random.multivariate_normal([0, 4], [[1, 0], [0, 1]], 150)

    axes[2].scatter(dense_cluster[:, 0], dense_cluster[:, 1], alpha=0.7, label='Dense', s=25)
    axes[2].scatter(sparse_cluster[:, 0], sparse_cluster[:, 1], alpha=0.7, label='Sparse', s=25)
    axes[2].scatter(medium_cluster[:, 0], medium_cluster[:, 1], alpha=0.7, label='Medium', s=25)
    axes[2].set_title('Variable Density\n(Fuzzy Membership)', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Fuzzy C-Means: Preferred Cluster Shapes', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('fuzzy_cmeans_preferred_shapes.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_mountain_clustering_preferred_shapes():
    """Mountain Clustering handles multi-modal and peak-based structures"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Multi-modal distribution
    np.random.seed(42)
    # Create multiple peaks
    peak1 = np.random.multivariate_normal([1, 1], [[0.3, 0], [0, 0.3]], 100)
    peak2 = np.random.multivariate_normal([3, 1], [[0.4, 0], [0, 0.4]], 80)
    peak3 = np.random.multivariate_normal([2, 3], [[0.2, 0], [0, 0.5]], 120)
    peak4 = np.random.multivariate_normal([0, 3], [[0.3, 0], [0, 0.3]], 90)

    all_peaks = np.vstack([peak1, peak2, peak3, peak4])
    labels = np.hstack([np.zeros(100), np.ones(80), np.full(120, 2), np.full(90, 3)])

    axes[0].scatter(all_peaks[:, 0], all_peaks[:, 1], c=labels, cmap='tab10', alpha=0.7)
    axes[0].set_title('Multi-Modal Peaks\n(Mountain Clustering)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Density-based clusters
    np.random.seed(123)
    # High density center, lower density surroundings
    center = np.random.multivariate_normal([2, 2], [[0.2, 0], [0, 0.2]], 150)
    surrounding = np.random.multivariate_normal([2, 2], [[1.5, 0], [0, 1.5]], 100)

    axes[1].scatter(surrounding[:, 0], surrounding[:, 1], alpha=0.4, s=20, label='Low Density')
    axes[1].scatter(center[:, 0], center[:, 1], alpha=0.8, s=30, label='High Density')
    axes[1].set_title('Density Gradients\n(Peak Detection)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Irregular shaped clusters with peaks
    theta = np.linspace(0, 2*np.pi, 200)
    r1 = 2 + 0.5 * np.sin(5*theta)
    r2 = 1.5 + 0.3 * np.cos(7*theta)

    x1 = r1 * np.cos(theta) + 2
    y1 = r1 * np.sin(theta) + 2
    x2 = r2 * np.cos(theta) - 2
    y2 = r2 * np.sin(theta) - 1

    axes[2].scatter(x1, y1, alpha=0.7, s=25, label='Peak Region 1')
    axes[2].scatter(x2, y2, alpha=0.7, s=25, label='Peak Region 2')
    axes[2].set_title('Irregular Peak Regions\n(Mountain Method)', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Mountain Clustering: Preferred Cluster Shapes', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('mountain_clustering_preferred_shapes.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_dbscan_preferred_shapes():
    """DBSCAN handles arbitrary shapes and noise"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Arbitrary shaped clusters
    X1, y1 = make_moons(n_samples=300, noise=0.1, random_state=42)
    axes[0].scatter(X1[:, 0], X1[:, 1], c=y1, cmap='viridis', alpha=0.7)
    axes[0].set_title('Crescent Shapes\n(DBSCAN Strength)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Nested circles
    X2, y2 = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=42)
    axes[1].scatter(X2[:, 0], X2[:, 1], c=y2, cmap='plasma', alpha=0.7)
    axes[1].set_title('Nested Circles\n(Non-Convex Shapes)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Clusters with noise and varying densities
    np.random.seed(42)
    # Dense cluster
    dense = np.random.multivariate_normal([0, 0], [[0.3, 0], [0, 0.3]], 150)
    # Sparse cluster
    sparse = np.random.multivariate_normal([3, 3], [[0.8, 0], [0, 0.8]], 100)
    # Elongated cluster
    elongated = np.random.multivariate_normal([0, 3], [[0.2, 0], [0, 1.2]], 120)
    # Noise points
    noise = np.random.uniform(-2, 5, (50, 2))

    axes[2].scatter(dense[:, 0], dense[:, 1], alpha=0.7, s=25, label='Dense Cluster')
    axes[2].scatter(sparse[:, 0], sparse[:, 1], alpha=0.7, s=25, label='Sparse Cluster')
    axes[2].scatter(elongated[:, 0], elongated[:, 1], alpha=0.7, s=25, label='Elongated Cluster')
    axes[2].scatter(noise[:, 0], noise[:, 1], alpha=0.5, s=15, c='gray', label='Noise')
    axes[2].set_title('Variable Density + Noise\n(DBSCAN Handles Well)', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('DBSCAN: Preferred Cluster Shapes', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('dbscan_preferred_shapes.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_all_comparisons():
    """Generate all clustering algorithm comparisons"""
    print("Generating K-Means preferred shapes...")
    generate_kmeans_preferred_shapes()

    print("Generating Fuzzy C-Means preferred shapes...")
    generate_fuzzy_cmeans_preferred_shapes()

    print("Generating Mountain Clustering preferred shapes...")
    generate_mountain_clustering_preferred_shapes()

    print("Generating DBSCAN preferred shapes...")
    generate_dbscan_preferred_shapes()

    print("All images generated successfully!")
    print("Files saved:")
    print("- kmeans_preferred_shapes.png")
    print("- fuzzy_cmeans_preferred_shapes.png")
    print("- mountain_clustering_preferred_shapes.png")
    print("- dbscan_preferred_shapes.png")

if _name_ == "_main_":
    generate_all_comparisons()