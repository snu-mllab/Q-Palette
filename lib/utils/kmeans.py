import torch
from typing import List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from flash1dkmeans import kmeans_1d

def kmeans_flash1d(data, k, max_data=int(1e8)):
    """
    data: torch.Tensor, shape: (n, vec_sz)
    k: int
    """
    data_np = data.cpu().numpy()
    rand_indices = np.random.choice(data_np.shape[0], min(max_data, data_np.shape[0]), replace=False)
    data_sampled = data_np[rand_indices]
    centroids, _ = kmeans_1d(data_sampled.reshape(-1), k)
    return torch.tensor(centroids, device=data.device, dtype=data.dtype).view(-1, 1)

def kmeans_sklearn(data, k, max_data=int(1e7), tol=1e-6):
    """
        data: torch.Tensor, shape: (n, vec_sz)
        k: int
    """
    data_np = data.cpu().numpy()
    data_sampled = data_np[np.random.choice(data_np.shape[0], min(max_data, data_np.shape[0]), replace=False)]
    kmeans = KMeans(n_clusters=k, tol=tol).fit(data_sampled)
    centroids = torch.tensor(kmeans.cluster_centers_, device=data.device, dtype=data.dtype)
    print(kmeans.n_iter_)
    return centroids


@torch.compile
def _kmeans_greedy_init(data: torch.Tensor, k: int) -> torch.Tensor:
    """Get initial clusters by iteratively choosing a vector that is the farthest from already selected clusters"""
    clusters = torch.zeros(k, data.shape[1], device=data.device, dtype=data.dtype)
    running_min_distances = torch.full((data.shape[0],), torch.inf, device=data.device, dtype=data.dtype)
    data_norm_squared = data.norm(p=2, dim=1).square()

    for i in range(k):
        clusters[i] = data[running_min_distances.argmax()]
        distances_to_cluster_i = data_norm_squared - 2 * data @ clusters[i] + clusters[i].norm().square()
        running_min_distances = torch.minimum(running_min_distances, distances_to_cluster_i, out=running_min_distances)
    return clusters

@torch.compile
def fit_kmeans(
    data: torch.Tensor,
    k: int,
    max_iter: int = 1000,
    check_every: int = 10,
    rtol: float = 1e-06,
    atol: float = 1e-08,
    greedy_init: bool = False,
    block_size_vals: int = 2**30,
    devices: Optional[List[torch.device]] = None,
):
    """
    :param data: [nsamples, dim]
    :param k: number of centroids
    :param max_iter: run at most this many iterations
    :param check_every: check for convergence (allclose(new_centroids, old_centroids)) once in this many steps
    :param rtol: early stopping relative tolerance for centroids
    :param atol: early stopping absolute tolerance for centroids
    :param greedy_init: if True, init by greedily selecting the point that is farthest from any cluster
        if False (default), initialize with random points using pytorch global RNG
    :param block_size_vals: how many dot products to compute at a time
    :param devices: if specified, run kmeans in data-parallel mode across these devices
    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    if devices is None:
        devices = [data.device]

    if greedy_init:
        clusters = _kmeans_greedy_init(data, k)
    else:
        clusters = data[torch.randperm(data.shape[0])[:k], :]  # [k, dim]

    block_size = block_size_vals // k
    shard_size = (len(data) - 1) // len(devices) + 1
    data = [
        data[gi * shard_size : (gi + 1) * shard_size].to(devices[gi], non_blocking=True) for gi in range(len(devices))
    ]
    nearest_indices = [torch.empty(len(data[gi]), dtype=torch.int64, device=devices[gi]) for gi in range(len(devices))]
    clusters = [clusters.to(device, non_blocking=True) for device in devices]

    for i in range(max_iter):
        for block_start in range(0, shard_size, block_size):
            for gi in range(len(devices)):
                nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                    torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                    data[gi][block_start : block_start + block_size],
                    clusters[gi].T,
                    beta=-0.5,
                ).argmax(1)
            # note: the above formula equals to - 0.5 || data[:, None, :] - clusters[None, :, :] || ^ 2 + const

        if len(devices) == 1:
            new_clusters = [
                clusters[0]
                .clone()
                .index_reduce_(dim=0, index=nearest_indices[0], source=data[0], reduce="mean", include_self=False)
            ]
        else:
            cluster_sums = [
                torch.zeros_like(clusters[gi])
                .index_add(dim=0, index=nearest_indices[gi], source=data[gi])
                .to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            cluster_counts = [
                torch.bincount(nearest_indices[gi], minlength=k).to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            for gi in range(1, len(devices)):
                cluster_sums[0] += cluster_sums[gi]
                cluster_counts[0] += cluster_counts[gi]

            new_clusters = [cluster_sums[0] / cluster_counts[0].unsqueeze(1).clamp_min(1)]
            new_clusters[0] += (cluster_counts[0].unsqueeze(1) == 0) * clusters[0]
            for gi in range(1, len(devices)):
                new_clusters.append(new_clusters[0].to(devices[gi], non_blocking=True))

        if i % check_every == 0:
            if torch.allclose(new_clusters[0], clusters[0], rtol=rtol, atol=atol):
                break
        clusters = new_clusters
    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                data[gi][block_start : block_start + block_size],
                clusters[gi].T,
                beta=-0.5,
            ).argmax(1)

    clusters = clusters[0]
    nearest_indices = torch.cat([nearest_indices[gi].to(devices[0]) for gi in range(len(devices))], dim=0)
    reconstructed_data = clusters[nearest_indices]
    return clusters, nearest_indices, reconstructed_data


def kmeanspp(data, k):
    """
    :param data: [nsamples, dim]
    :param k: number of centroids
    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    # Move data to GPU if available
    device = data.device
    
    # Cache for storing distances
    distances_cache = torch.zeros(data.shape[0], device=device)
    
    # kmeans++ initialization 
    centroids = []
    for i in range(k):
        if i % 10000 == 0:
            print(i, k)
        if i == 0:
            idx = torch.randint(0, data.shape[0], (1,)).item()
            centroids.append(data[idx])
            # Initialize distances cache
            distances_cache = torch.sum((data - centroids[0])**2, dim=1)
        else:
            # Update distances cache efficiently
            last_centroid = centroids[-1].unsqueeze(0)
            new_distances = torch.sum((data - last_centroid)**2, dim=1)
            distances_cache = torch.minimum(distances_cache, new_distances)
            
            # Sample next centroid proportional to squared distances
            probs = distances_cache / torch.sum(distances_cache)
            idx = torch.multinomial(probs, 1).item()
            centroids.append(data[idx])
    
    clusters = torch.stack(centroids)
    
    # Assign each data point to nearest centroid
    distances = torch.cdist(data, clusters)
    nearest_indices = torch.argmin(distances, dim=1)
    reconstructed_data = clusters[nearest_indices]
    
    return clusters, nearest_indices, reconstructed_data


def faiss_kmeans(data, k):
    assert len(data.shape) == 2
    n_samples, dim = data.shape
    data_np = data.cpu().numpy().astype(np.float32)
    kmeans = faiss.Kmeans(dim, k, niter=100, gpu=torch.cuda.is_available())
    kmeans.train(data_np)
    faiss_centroids = torch.tensor(kmeans.centroids, device=data.device)
    return faiss_centroids

def test_fit_kmeans():
    """
    Test our kmeans implementation against Faiss kmeans and kmeanspp
    """
    import numpy as np
    # import faiss

    # Generate random data
    n_samples = 100000
    dim = 1
    k = 16
    
    # Create random data
    data = torch.randn(n_samples, dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run our kmeans
    print("Running our kmeans implementation...")
    our_clusters, our_indices, our_reconstructed = fit_kmeans(
        data, 
        k=k, 
        max_iter=1000, 
        check_every=5,
        devices=[torch.device('cuda')] if torch.cuda.is_available() else None
    )
    print("our_clusters", our_clusters)
    
    # Run our kmeanspp
    print("Running our kmeanspp implementation...")
    kmeanspp_clusters, kmeanspp_indices, kmeanspp_reconstructed = kmeanspp(data, k)
    
    # # Run Faiss kmeans
    # print("Running Faiss kmeans implementation...")
    # data_np = data.cpu().numpy().astype(np.float32)
    
    # kmeans = faiss.Kmeans(dim, k, niter=100, gpu=torch.cuda.is_available())
    # kmeans.train(data_np)
    
    # Get Faiss results
    # faiss_centroids = torch.tensor(kmeans.centroids, device=data.device)
    # _, faiss_indices = kmeans.index.search(data_np, 1)
    # faiss_indices = torch.tensor(faiss_indices.squeeze(), device=data.device)
    # faiss_reconstructed = faiss_centroids[faiss_indices]
    
    # Compare results
    # Note: The clusters may not be in the same order, so we compare reconstruction error
    our_error = (data - our_reconstructed).pow(2).mean()
    # kmeanspp_error = torch.norm(data - kmeanspp_reconstructed).item()
    # faiss_error = torch.norm(data - faiss_reconstructed).item()
    
    print(f"Our kmeans reconstruction error: {our_error:.6f}")
    # print(f"Our kmeanspp reconstruction error: {kmeanspp_error:.6f}")
    # print(f"Faiss kmeans reconstruction error: {faiss_error:.6f}")
    # print(f"Relative difference (kmeans vs Faiss): {abs(our_error - faiss_error) / faiss_error:.6f}")
    # print(f"Relative difference (kmeanspp vs Faiss): {abs(kmeanspp_error - faiss_error) / faiss_error:.6f}")

if __name__ == "__main__":
    test_fit_kmeans()