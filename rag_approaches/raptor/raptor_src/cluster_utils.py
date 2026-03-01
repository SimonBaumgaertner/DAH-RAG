import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node
# Import necessary methods from other modules
from .utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    # UMAP requires n_neighbors > 1 and n_neighbors < len(embeddings)
    # If we have 2 or fewer embeddings, we can't use UMAP, so return as-is
    if len(embeddings) <= 2:
        # Return original embeddings (or pad/truncate to dim if needed)
        if embeddings.shape[1] == dim:
            return embeddings
        # If dimension mismatch, use PCA or just return first dim components
        return embeddings[:, :dim] if embeddings.shape[1] > dim else embeddings
    
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    
    # Ensure n_neighbors is at least 2 and less than len(embeddings)
    n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
    
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    # UMAP requires n_neighbors > 1 and n_neighbors < len(embeddings)
    # If we have 2 or fewer embeddings, we can't use UMAP, so return as-is
    if len(embeddings) <= 2:
        # Return original embeddings (or pad/truncate to dim if needed)
        if embeddings.shape[1] == dim:
            return embeddings
        # If dimension mismatch, use PCA or just return first dim components
        return embeddings[:, :dim] if embeddings.shape[1] > dim else embeddings
    
    # Ensure num_neighbors is at least 2 and less than len(embeddings)
    num_neighbors = max(2, min(num_neighbors, len(embeddings) - 1))
    
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 10, random_state: int = RANDOM_SEED
) -> int:
    # If we have 2 or fewer embeddings, we can only have 1 cluster
    if len(embeddings) <= 2:
        return 1
    
    # Limit max_clusters to be at most len(embeddings) - 1 to avoid ill-defined covariance
    # GaussianMixture needs at least n_components samples to fit properly
    max_clusters = min(max_clusters, len(embeddings) - 1, 50)
    
    # Need at least 2 clusters to compare, but if we only have 3 samples, max is 2
    if max_clusters < 2:
        return 1
    
    n_clusters = np.arange(1, max_clusters + 1)
    bics = []
    
    for n in n_clusters:
        try:
            # Add regularization to help with numerical stability
            gm = GaussianMixture(
                n_components=n, 
                random_state=random_state,
                reg_covar=1e-6,  # Regularization for covariance
                max_iter=100
            )
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        except (ValueError, np.linalg.LinAlgError) as e:
            # If fitting fails (e.g., ill-defined covariance), skip this number of clusters
            # Use a high BIC value so it won't be selected
            bics.append(float('inf'))
            logging.debug(f"Failed to fit GMM with {n} components: {e}")
    
    # If all fits failed, return 1 cluster
    if all(bic == float('inf') for bic in bics):
        logging.warning(f"All GMM fits failed for {len(embeddings)} embeddings, using 1 cluster")
        return 1
    
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Performing GMM clustering on {len(embeddings)} embeddings")
    # If we have 2 or fewer embeddings, return a single cluster
    if len(embeddings) <= 2:
        # All embeddings belong to cluster 0
        labels = [[0] for _ in range(len(embeddings))]
        return labels, 1
    
    n_clusters = get_optimal_clusters(embeddings)
    
    try:
        # Add regularization to help with numerical stability
        gm = GaussianMixture(
            n_components=n_clusters, 
            random_state=random_state,
            reg_covar=1e-6,  # Regularization for covariance
            max_iter=100
        )
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters
    except (ValueError, np.linalg.LinAlgError) as e:
        # If fitting fails, fall back to single cluster
        logging.warning(f"GMM clustering failed for {len(embeddings)} embeddings: {e}. Using single cluster.")
        # All embeddings belong to cluster 0
        labels = [[0] for _ in range(len(embeddings))]
        return labels, 1


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 4000,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 50,
        threshold: float = 0.3,
        verbose: bool = True,
        recursion_depth: int = 0,
        max_recursion_depth: int = 5,
    ) -> List[List[Node]]:
        print(f"Performing clustering with {len(nodes)} nodes")
        # Get the embeddings from the nodes
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        # Perform the clustering
        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                # Check recursion depth to prevent infinite recursion
                if recursion_depth >= max_recursion_depth:
                    if verbose:
                        logging.warning(
                            f"Max recursion depth ({max_recursion_depth}) reached for cluster with {len(cluster_nodes)} nodes "
                            f"(total_length={total_length}). Returning cluster as-is."
                        )
                    node_clusters.append(cluster_nodes)
                else:
                    if verbose:
                        logging.info(
                            f"reclustering cluster with {len(cluster_nodes)} nodes (depth={recursion_depth})"
                        )
                    node_clusters.extend(
                        RAPTOR_Clustering.perform_clustering(
                            cluster_nodes,
                            embedding_model_name,
                            max_length_in_cluster,
                            tokenizer,
                            reduction_dimension,
                            threshold,
                            verbose,
                            recursion_depth + 1,
                            max_recursion_depth,
                        )
                    )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
