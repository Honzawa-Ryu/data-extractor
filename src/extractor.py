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
    def __init__(self, n_neighbors: int=10, resolution: float=1.0, pca_threshold: float=0.9):
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.pca_threshold = pca_threshold

    def preprocess(self, data: pd.DataFrame, start_index: int=0, end_index: int=None):
        """
        クラスタリングに供するにあたっての前処理を担当する
        データ丸ごとを受け取る必要はないかも、

        具体的には以下のステップ
        ・データクリーニング
            欠損値の補完を行う
        ・標準化
            データの標準化を行う。基本的にはzスコアへの変換を行うが、Min-Max scalingを行う選択肢も入れるかも？　あんまないかも
        ・次元削減
            必要性は低いかも。データのスパース性が高い、要素間の相関が高い（これは可能性がなかなかあるかも）場合PCAによる次元削減が有効な可能性がある。
            累積分散率が70%, 90%になるまで主成分を取ってくる、みたいな方針がいいのかなと思っています
        """
        # データクリーニング
        if end_index == None:
            end_index = data.shape[1]
        data_to_process = data.iloc[:, start_index:end_index]

        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data_to_process)

        # 標準化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)

        # 次元削減
        if data_scaled.shape[1] > 1:
            pca = PCA()  
            pca.fit(data_scaled)

            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

            n_components = np.where(cumulative_variance >= self.pca_threshold)[0]
            if len(n_components) == 0:
                n_components = data_scaled.shape[1]
            else:
                n_components = n_components[0] + 1
            if n_components < data_scaled.shape[1]:
                pca = PCA(n_components=n_components)
                data_scaled = pca.fit_transform(data_scaled)

        return data_scaled

    def clustering(self, data: np.ndarray, visualize: bool=False):
        """
        Leidenクラスタリングを実行し、各クラスタの代表点を選択する
        """
        # k-NNグラフの構築
        knn_graph = kneighbors_graph(data, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
        sources, targets = knn_graph.nonzero()
        edges = list(zip(sources.tolist(), targets.tolist()))
        g = ig.Graph(edges=edges, directed=False)

        # Leidenクラスタリングの実行
        partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=self.resolution)
        labels = np.array(partition.membership)

        # 各クラスタの代表点の選択
        representative_indices = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_data = data[cluster_indices]

            # クラスタ中心の計算
            cluster_center = np.mean(cluster_data, axis=0)

            # クラスタ中心に最も近い点を代表点として選択
            distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
            representative_index_within_cluster = np.argmin(distances)
            representative_index = cluster_indices[representative_index_within_cluster]

            representative_indices.append(representative_index)
        representative_data = data[representative_indices]
        print(f"Clustering completed: {len(unique_labels)} clusters found, {len(representative_indices)} representative points selected.")

        if visualize:
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(data)

            plt.figure(figsize=(10, 6))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
            rep_embedding = reducer.transform(representative_data)
            plt.scatter(rep_embedding[:, 0], rep_embedding[:, 1], c='black', s=50, marker='x', label='Representatives')
            plt.title('Leiden Clustering with Representatives')
            plt.legend()
            plt.show()

        return representative_data, representative_indices
    
    def fit_transform(self, data: pd.DataFrame, start_index: int=0, end_index: int=None, visualize: bool=False):
        preprocessed_data = self.preprocess(data, start_index, end_index)
        representative_data, representative_indices = self.clustering(preprocessed_data, visualize=visualize)
        return representative_data, representative_indices