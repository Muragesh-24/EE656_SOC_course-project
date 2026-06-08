# This is code for the images generated in the report

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform

def generate_data():
    """Generates synthetic data for clustering."""
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=1.5)
    return X

def calculate_pi_si_di(X, labels, centroids, n_samples, n_clusters):
    """
    Calculates Partition Index (PI), Separation Index (SI), and Dunn Index (DI)
    for hard clustering (K-Means results).
    """
    pi_score = 0
    si_score = 0
    di_score = 0

    if n_clusters <= 1:
        # PI, SI, DI are not well-defined for 1 or fewer clusters
        return np.nan, np.nan, np.nan

    # Calculate within-cluster sum of squares (SSW_m) for each cluster
    intra_cluster_ssw = np.zeros(n_clusters)
    cluster_sizes = np.zeros(n_clusters, dtype=int)

    for m in range(n_clusters):
        cluster_points = X[labels == m]
        cluster_sizes[m] = len(cluster_points)
        if cluster_sizes[m] > 0:
            intra_cluster_ssw[m] = np.sum(np.linalg.norm(cluster_points - centroids[m], axis=1)**2)

    # Calculate squared distances between all centroids
    centroid_distances_sq = pairwise_distances(centroids, metric='euclidean')**2

    # --- Partition Index (PI) Calculation ---
    # PI = sum_m [SSW_m / (|X_m| * sum_k_ne_m ||c_k - c_m||^2)]
    sum_pi_terms = 0
    for m in range(n_clusters):
        if cluster_sizes[m] > 0:
            # Sum of squared distances from centroid m to all other centroids
            sum_dist_to_other_centroids_sq_m = np.sum(centroid_distances_sq[m, np.arange(n_clusters) != m])

            if sum_dist_to_other_centroids_sq_m > 0: # Avoid division by zero
                term_m = intra_cluster_ssw[m] / (cluster_sizes[m] * sum_dist_to_other_centroids_sq_m)
                sum_pi_terms += term_m
            else:
                # If sum_dist_to_other_centroids_sq_m is zero, it means all centroids are at the same point
                # This indicates degenerate clustering, PI approaches infinity or undefined
                sum_pi_terms += np.inf # Or handle as desired
        else:
            sum_pi_terms += 0 # Empty clusters contribute 0 to sum

    pi_score = sum_pi_terms

    # --- Separation Index (SI) Calculation ---
    # SI = inertia / (n * min_k_ne_m ||c_k - c_m||^2)
    inertia = np.sum(intra_cluster_ssw) # Total within-cluster sum of squares

    min_inter_centroid_distance_sq = np.inf
    if n_clusters > 1:
        # Get all pairwise distances excluding self-distances
        temp_dist_sq = centroid_distances_sq.copy()
        np.fill_diagonal(temp_dist_sq, np.inf) # Set diagonal to inf to ignore self-distances
        min_inter_centroid_distance_sq = np.min(temp_dist_sq)

    if n_samples > 0 and min_inter_centroid_distance_sq > 0 and min_inter_centroid_distance_sq != np.inf:
        si_score = inertia / (n_samples * min_inter_centroid_distance_sq)
    else:
        si_score = np.nan # Undefined or degenerate case

    # --- Dunn Index (DI) Calculation ---
    # DI = min_inter_cluster_distance_numerator / max_intra_cluster_diameter

    # Calculate max intra-cluster diameter
    max_intra_cluster_diameter = -np.inf
    for m in range(n_clusters):
        cluster_points = X[labels == m]
        if len(cluster_points) >= 2:
            # Calculate all pairwise distances within the cluster
            pairwise_dists = pdist(cluster_points, metric='euclidean')
            diameter_m = np.max(pairwise_dists)
            max_intra_cluster_diameter = max(max_intra_cluster_diameter, diameter_m)
        elif len(cluster_points) == 1:
            diameter_m = 0 # Diameter of a single point is 0
            max_intra_cluster_diameter = max(max_intra_cluster_diameter, diameter_m)
        # For empty clusters, diameter remains -inf, will be handled later

    # Handle case where max_intra_cluster_diameter is still -inf (e.g., all clusters empty or single point)
    if max_intra_cluster_diameter == -np.inf:
        max_intra_cluster_diameter = 0 # Set to 0 to avoid division by inf, will result in np.inf for DI
    if max_intra_cluster_diameter == 0:
        # If max diameter is 0, it means all points within all clusters are identical.
        # This implies perfect compactness, making DI potentially infinite.
        # Handle this as np.inf as per typical DI definition.
        di_denominator = 1e-10 # Use a small number to avoid division by exact zero for practical calculation
    else:
        di_denominator = max_intra_cluster_diameter

    # Calculate min inter-cluster distance (d(X_m,X_k) as per formula)
    min_inter_cluster_distance_numerator = np.inf

    # Iterate through all unique pairs of clusters (m, k)
    for m in range(n_clusters):
        for k in range(m + 1, n_clusters): # Ensure k > m to avoid duplicates and self-comparison
            cluster_m_points = X[labels == m]
            cluster_k_points = X[labels == k]

            if len(cluster_m_points) > 0 and len(cluster_k_points) > 0:
                v_m = centroids[m]
                v_k = centroids[k]

                # Sum of distances from points in cluster m to centroid v_k
                sum_dist_m_to_vk = np.sum(np.linalg.norm(cluster_m_points - v_k, axis=1))
                # Sum of distances from points in cluster k to centroid v_m
                sum_dist_k_to_vm = np.sum(np.linalg.norm(cluster_k_points - v_m, axis=1))

                # Denominator for d(X_m, X_k)
                d_mk_denom = (len(cluster_m_points) + len(cluster_k_points))

                if d_mk_denom > 0:
                    d_mk = (sum_dist_m_to_vk + sum_dist_k_to_vm) / d_mk_denom
                    min_inter_cluster_distance_numerator = min(min_inter_cluster_distance_numerator, d_mk)
                else:
                    # Should not happen if both clusters have points, but for robustness
                    min_inter_cluster_distance_numerator = 0 # Or handle as desired if clusters are empty

    if min_inter_cluster_distance_numerator == np.inf:
        di_score = np.nan # Undefined if no valid inter-cluster distances
    else:
        di_score = min_inter_cluster_distance_numerator / di_denominator

    return pi_score, si_score, di_score


def cluster_and_plot_all_indices(X, k_max=6):
    """
    Performs K-Means clustering, calculates GSI, PI, SI, DI,
    and plots clustering results and index variations.
    """
    gsi_scores = []
    pi_scores = []
    si_scores = []
    di_scores = []
    k_range = range(2, k_max + 1)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    # --- Plot 1: Clustering Visualizations (5 subplots) and GSI vs k (1 subplot) ---
    fig_combined, axes_combined = plt.subplots(2, 3, figsize=(18, 12)) # 2 rows, 3 columns = 6 subplots
    axes_combined = axes_combined.flatten() # Flatten for easy indexing

    for i, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        # Calculate all indices
        gsi = silhouette_score(X, labels)
        pi, si, di = calculate_pi_si_di(X, labels, centroids, X.shape[0], k)

        gsi_scores.append(gsi)
        pi_scores.append(pi)
        si_scores.append(si)
        di_scores.append(di)

        # Plot clustering visualization in the first 5 subplots
        if i < 5: # For k=2,3,4,5,6, these will occupy axes_combined[0] to axes_combined[4]
            ax = axes_combined[i]
            for j in range(k):
                mask = labels == j
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[j % len(colors)], alpha=0.7, s=30)

            ax.scatter([c[0] for c in centroids], [c[1] for c in centroids],
                      c='black', marker='x', s=100, linewidths=2)
            ax.set_title(f'k={k}, GSI={gsi:.4f}')
            ax.grid(True, alpha=0.3)

    # Plot GSI vs k in the 6th subplot (axes_combined[5])
    ax_gsi_combined = axes_combined[5]
    ax_gsi_combined.plot(k_range, gsi_scores, 'bo-', linewidth=2, markersize=8)
    ax_gsi_combined.set_xlabel('Number of Clusters (k)')
    ax_gsi_combined.set_ylabel('Global Silhouette Index')
    ax_gsi_combined.set_title('Global Silhouette Index vs. k')
    ax_gsi_combined.grid(True, alpha=0.3)

    fig_combined.tight_layout() # Corrected call
    fig_combined.savefig('clustering_and_gsi_plot.png', dpi=300, bbox_inches='tight')
    print("Clustering visuals and GSI plot saved as 'clustering_and_gsi_plot.png'")

    # --- Plot 2: PI, SI, DI vs k (separate image) ---
    fig_other_indices, ax_other_indices = plt.subplots(1, 1, figsize=(10, 6))
    ax_other_indices.plot(k_range, pi_scores, 'gv-', linewidth=2, markersize=8, label='Partition Index (PI)')
    ax_other_indices.plot(k_range, si_scores, 'rs-', linewidth=2, markersize=8, label='Separation Index (SI)')
    ax_other_indices.plot(k_range, di_scores, 'c^-', linewidth=2, markersize=8, label='Dunn Index (DI)')
    ax_other_indices.plot(k_range, gsi_scores, 'bo-', linewidth=2, markersize=8, label='Global Silhouette Index (GSI)')
    ax_other_indices.set_xlabel('Number of Clusters (k)')
    ax_other_indices.set_ylabel('Index Value')
    ax_other_indices.set_title('Cluster Validity Indices vs. Number of Clusters')
    ax_other_indices.legend()
    ax_other_indices.grid(True, alpha=0.3)
    fig_other_indices.tight_layout() # Corrected call
    fig_other_indices.savefig('other_indices_vs_k_plot.png', dpi=300, bbox_inches='tight')
    print("PI, SI, DI vs k plot saved as 'other_indices_vs_k_plot.png'")

    # Find optimal k based on GSI (maximize), PI (minimize), SI (minimize), DI (maximize)
    optimal_k_gsi = k_range[np.argmax(gsi_scores)] if len(gsi_scores) > 0 else np.nan
    optimal_k_pi = k_range[np.nanargmin(pi_scores)] if len(pi_scores) > 0 and not np.all(np.isnan(pi_scores)) else np.nan
    optimal_k_si = k_range[np.nanargmin(si_scores)] if len(si_scores) > 0 and not np.all(np.isnan(si_scores)) else np.nan
    optimal_k_di = k_range[np.nanargmax(di_scores)] if len(di_scores) > 0 and not np.all(np.isnan(di_scores)) else np.nan

    print(f"Optimal k based on GSI (max): {optimal_k_gsi}")
    print(f"Optimal k based on PI (min): {optimal_k_pi}")
    print(f"Optimal k based on SI (min): {optimal_k_si}")
    print(f"Optimal k based on DI (max): {optimal_k_di}")

    return {
        'optimal_k_gsi': optimal_k_gsi,
        'optimal_k_pi': optimal_k_pi,
        'optimal_k_si': optimal_k_si,
        'optimal_k_di': optimal_k_di,
        'gsi_scores': gsi_scores,
        'pi_scores': pi_scores,
        'si_scores': si_scores,
        'di_scores': di_scores
    }

# Run analysis
X = generate_data()
results = cluster_and_plot_all_indices(X)