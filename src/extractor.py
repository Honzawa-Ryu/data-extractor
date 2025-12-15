import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
import sklearn.decomposition as skd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import umap
import matplotlib.pyplot as plt

from .utils import BaseExtractor

class LeidenRepresentativeSelector(BaseExtractor):
    """
    Leidenクラスタリングを用いてデータをクラスタリングし、各クラスタの代表点を選択するクラス
    代表点は各クラスタの中心に最も近い点として選択される
    使い方の例:
        selector = LeidenRepresentativeSelector(criteria=0.5, and_logic=False)
        preprocessed_data = selector.preprocess(dataframe, start_index=1, end_index=None)
        clustered_data = selector.clustering(preprocessed_data)
    ここで、dataframeはPandasのDataFrame形式のデータを指す
    なお、start_indexとend_indexは前処理に使用する列の範囲を指定するためのものである
    例えば、start_index=1, end_index=Noneとすると、1列目以降の全ての列が前処理に使用される 
    """
    def __init__(self, n_neighbors: int=10, resolution: float=1.0, pca: bool=False, pca_threshold: float=0.9, strategy: str='mean'):
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.pca = pca
        self.pca_threshold = pca_threshold
        self.strategy = strategy

    def _preprocess(self, data: np.ndarray):
        """
        データの前処理を行う
        具体的には、欠損値の補完、標準化、必要に応じてPCAによる次元削減を行う
        """
        # データクリーニング

        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)

        # 標準化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)

        # 次元削減
        if self.pca and data_scaled.shape[1] > 1:
            pca = PCA(n_components=self.pca_threshold)  
            data_scaled = pca.fit_transform(data_scaled)

        return data_scaled
    
    def _make_graph(self, data: np.ndarray):
        """
        k-NNグラフを構築する
        """
        knn_graph = kneighbors_graph(data, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
        sources, targets = knn_graph.nonzero()
        edges = list(zip(sources, targets))
        g = ig.Graph(edges=edges, directed=False)
        return g

    def _clustering(self, g: ig.Graph):
        """
        Leidenクラスタリングを実行する
        """
        # Leidenクラスタリングの実行
        partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=self.resolution)
        labels = np.array(partition.membership)
        print(f"Clustering completed: {len(set(labels))} clusters found.")

        return labels
    
    def _select_representatives(self, data: np.ndarray, g: ig.Graph, labels: np.ndarray):
        """ 
        各クラスタの代表点を選択する
        """
        # 各クラスタの代表点の選択
        representative_indices = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_data = data[cluster_indices]

            if self.strategy == 'mean':
                cluster_center = np.mean(cluster_data, axis=0)
            elif self.strategy == 'median':
                cluster_center = np.median(cluster_data, axis=0)

            elif self.strategy == 'degree':
                g_sub = g.subgraph(cluster_indices)
                degrees = g_sub.degree()
                cluster_center = np.argmax(degrees)

            elif self.strategy == 'closeness':
                g_sub = g.subgraph(cluster_indices)
                closeness = g_sub.closeness()
                cluster_center = np.argmax(closeness)

            elif self.strategy == 'betweenness':
                g_sub = g.subgraph(cluster_indices)
                betweenness = g_sub.betweenness()
                cluster_center = np.argmax(betweenness)

            else:
                raise ValueError(f"Invalid strategy: {self.strategy}.")
            
            distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
            representative_index_within_cluster = np.argmin(distances)
            
            representative_index = cluster_indices[representative_index_within_cluster]
            representative_indices.append(representative_index)

        print(f"{len(representative_indices)} representative points selected.")
        return representative_indices

    def _visualize(self, data: np.ndarray, labels: np.ndarray, representative_indices: list):
        """
        UMAPを用いてクラスタリング結果と代表点を可視化する
        """
    
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data)

        plt.figure(figsize=(10, 6))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
        rep_embedding = embedding[representative_indices]
        plt.scatter(rep_embedding[:, 0], rep_embedding[:, 1], c='black', s=50, marker='x', label='Representatives')
        plt.title('Leiden Clustering with Representatives')
        plt.legend()
        plt.show()
    
    def fit_transform(self, data: pd.DataFrame, feature_cols: list=None, visualize: bool=False):
        if feature_cols is not None:
            feature_data = data[feature_cols].values
        else:
            feature_data = data.values
            
        preprocessed_data = self._preprocess(feature_data)
        g = self._make_graph(preprocessed_data)
        labels = self._clustering(g)
        representative_indices = self._select_representatives(preprocessed_data, g, labels)
        if visualize:
            self._visualize(preprocessed_data, labels, representative_indices)
        return data.iloc[representative_indices], representative_indices, labels
