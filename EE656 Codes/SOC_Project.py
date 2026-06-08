import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage import data
from skimage.transform import resize
import pandas as pd


def factorcal(x, nk, val):
    return np.ones(nk)


def compute_imc(X, labels, centroids):
    compactness = 0
    for k in range(len(centroids)):
        cluster_k = X[labels == k]
        if len(cluster_k) > 0:
            compactness += np.mean(np.linalg.norm(cluster_k - centroids[k], axis=1))
    return compactness / len(centroids)

def compute_imc2(X, labels, centroids):
    n_clusters = centroids.shape[0]
    separation = 0
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            separation += np.linalg.norm(centroids[i] - centroids[j])
    return separation / (n_clusters * (n_clusters - 1) / 2 + 1e-10)


def soc(x, nk, factor):
    x = np.array(x, dtype=float)
    n, k = x.shape
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    u = (x - x_min) / np.where(x_max - x_min == 0, 1, x_max - x_min)
    U = u.copy()

    m = np.zeros(nk, dtype=int)
    t = np.zeros(nk + 1, dtype=int)
    t[0] = n
    sl = np.zeros((n, nk + 1), dtype=int)
    sl[:, 0] = np.arange(n)
    clst = np.zeros((n, k, nk))
    SL = np.zeros((n, nk), dtype=int)
    cc_norm = np.zeros((nk, k))
    d1 = np.zeros(nk)
    idx = np.zeros(n, dtype=int)
    dd = np.zeros((n, nk))
    part = np.zeros((n, nk))
    P = np.zeros((n, nk))

    for v in range(nk):
        if t[v] != 0:
            d = 0
            for j in range(t[v]):
                xi = x[sl[j, v]]
                sxi = np.sum(xi)
                if sxi != 0:
                    d += np.min(xi) / sxi
            d1[v] = ((1 / (2 * t[v])) * d) * factor[v]

            ur = u[sl[:t[v], v]]
            diff_matrix = ur[:, np.newaxis, :] - ur[np.newaxis, :, :]
            sq_dists = np.sum(diff_matrix**2, axis=-1)
            P[:t[v], v] = np.sum(np.exp(-sq_dists / (d1[v] ** 2)), axis=1)

            zmax = np.argmax(P[:t[v], v])
            cc_norm[v, :] = ur[zmax, :]

            m_v, next_sl, next_u = 0, [], []
            for r in range(t[v]):
                dist = np.sum((ur[r, :] - cc_norm[v, :]) ** 2)
                if dist <= d1[v]:
                    SL[m_v, v] = sl[r, v]
                    clst[m_v, :, v] = x[sl[r, v]]
                    m_v += 1
                else:
                    next_sl.append(sl[r, v])
                    next_u.append(ur[r])
            m[v] = m_v
            t[v + 1] = len(next_sl)
            if len(next_sl) > 0:
                sl[:t[v + 1], v + 1] = next_sl
                u[:t[v + 1], :] = np.array(next_u)

    if t[nk] != 0:
        ur = u[:t[nk], :]
        cc = cc_norm[:nk, :]
        diff_matrix = ur[:, np.newaxis, :] - cc[np.newaxis, :, :]
        sq_dists = np.sum(diff_matrix ** 2, axis=-1)
        assignments = np.argmin(sq_dists, axis=1)
        for r, v in enumerate(assignments):
            m[v] += 1
            SL[m[v] - 1, v] = sl[r, nk]
            clst[m[v] - 1, :, v] = x[sl[r, nk]]

    for r in range(n):
        for v in range(nk):
            if m[v] != 0:
                dd[r, v] = np.sum((U[r, :] - cc_norm[v]) ** 2)
        idx[r] = np.argmin(dd[r])
        part[r, idx[r]] = 1

    return {
        'idx': idx,
        'cc_norm': cc_norm,
    }


def show_segmented_image(X, labels, shape, title):
    segmented_img = np.zeros_like(X)
    for label in np.unique(labels):
        mask = (labels == label)
        segmented_img[mask] = np.mean(X[mask], axis=0)
    segmented_img = segmented_img.reshape(shape[0], shape[1], 3).astype(np.uint8)
    plt.figure(figsize=(6, 6))
    plt.imshow(segmented_img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def analyze():
    img = data.rocket()
    img = resize(img, (32, 32), anti_aliasing=True)
    X = (img * 255).astype(np.uint8).reshape((-1, 3))
    shape = img.shape[:2]

    k_range = range(2, 9)
    metrics = []
    soc_segmented_results = {}

    for k in k_range:
        fac = factorcal(X, k, 1)

       
        soc_result = soc(X, k, fac)
        soc_labels = soc_result['idx']
        soc_centroids = soc_result['cc_norm']
        soc_sil = silhouette_score(X, soc_labels)
        soc_imc = compute_imc(X, soc_labels, soc_centroids)
        soc_imc2 = compute_imc2(X, soc_labels, soc_centroids)

        soc_segmented_results[k] = soc_labels

  
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_sil = silhouette_score(X, kmeans_labels)
        kmeans_imc = compute_imc(X, kmeans_labels, kmeans.cluster_centers_)
        kmeans_imc2 = compute_imc2(X, kmeans_labels, kmeans.cluster_centers_)

        metrics.append({
            'k': k,
            'SOC_Silhouette': soc_sil,
            'SOC_IMC': soc_imc,
            'SOC_IMC2': soc_imc2,
            'KMeans_Silhouette': kmeans_sil,
            'KMeans_IMC': kmeans_imc,
            'KMeans_IMC2': kmeans_imc2,
        })

    df = pd.DataFrame(metrics)
    best_k = df.loc[df['SOC_Silhouette'].idxmax(), 'k']
    return df.round(4), X, shape, soc_segmented_results[int(best_k)], img


def plot_metrics(df):
    metrics = ['Silhouette', 'IMC', 'IMC2']
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for i, metric in enumerate(metrics):
        axs[i].plot(df['k'], df[f'SOC_{metric}'], marker='o', label='SOC')
        axs[i].plot(df['k'], df[f'KMeans_{metric}'], marker='s', label='KMeans')
        axs[i].set_title(f'{metric} vs k')
        axs[i].set_xlabel('k')
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()

df_results, X, shape, best_soc_labels, original_img = analyze()


plt.figure(figsize=(6, 6))
plt.imshow(original_img)
plt.title("Original Image")
plt.axis('off')
plt.tight_layout()
plt.show()


show_segmented_image(X, best_soc_labels, shape, "Final SOC Segmented Image (Best k)")


plot_metrics(df_results)


print("\nFinal Metric Table:")
print(df_results.to_string(index=False))
