# 12/16 複数のメソッドで共通して用いる部分は一つのメソッドにまとめてもよいと思う

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
    
    def _value_clusters(self, data: np.ndarray, cluster_labels: np.ndarray):
        """
        同一物質を投与されているデータが同一クラスタにまとまっているか確認する
        """
        compound_labels = data["COMPOUND_NAME"].tolist()
        unique_compounds = sorted(list(set(compound_labels)))
        mapping_dict = {element: i for i, element in enumerate(unique_compounds)}
        compound_indices = np.array([mapping_dict[name] for name in compound_labels])
        compound_cluster_dict = {}
        sum_clusters = 0
        for compound in unique_compounds:
            indices = np.where(compound_indices == mapping_dict[compound])[0]
            clusters = set(cluster_labels[indices])
            compound_cluster_dict[compound] = clusters
            sum_clusters += len(clusters)
            # print("Compund: %-20s, %2d, Clusters: %2d" % (compound, len(indices), len(clusters)))
        print(f"Over_partition: {sum_clusters / len(unique_compounds):.2f}")    
        return compound_cluster_dict

    def _select_representatives(self, data: np.ndarray, g: ig.Graph, labels: np.ndarray):
        """ 
        各クラスタの代表点候補のリスト（インデックス順）を計算して辞書で返す
        """
        unique_labels = np.unique(labels)
        center_list_dict = {}

        for label in unique_labels:
            # np.where はタプルを返すため [0] を指定
            cluster_indices = np.where(labels == label)[0]
            
            # dataがDataFrameかndarrayかで使い分け（ここではndarray想定）
            cluster_data = data[cluster_indices]

            if self.strategy == 'mean':
                cluster_center = np.mean(cluster_data, axis=0)
                distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
                center_list = np.argsort(distances) # 昇順（近い順）

            elif self.strategy == 'median':
                cluster_center = np.median(cluster_data, axis=0)
                distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
                center_list = np.argsort(distances) # 昇順

            elif self.strategy in ['degree', 'closeness', 'betweenness']:
                g_sub = g.subgraph(cluster_indices)
                
                if self.strategy == 'degree':
                    scores = g_sub.degree()
                elif self.strategy == 'closeness':
                    scores = g_sub.closeness()
                elif self.strategy == 'betweenness':
                    scores = g_sub.betweenness()
                
                # 【重要】リストにマイナスは使えないため np.array に変換
                center_list = np.argsort(-np.array(scores)) # 降順（スコア高い順）

            else:
                raise ValueError(f"Invalid strategy: {self.strategy}.")
            
            center_list_dict[label] = center_list

        return center_list_dict

            
    def _check_representatives(self, data: pd.DataFrame, center_list_dict: dict, labels: np.ndarray, g_value: dict=None):
        """
        候補リストに基づき、g_value条件を考慮して最終的な代表点を選択する
        """
        representative_indices = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            center_list = center_list_dict[label] # このクラスタ内での順位リスト(0~N-1)

            # --- デフォルト設定（条件に合わない場合はここに戻る）---
            # クラスタ内1位のローカルインデックスを取得
            top_local_index = center_list[0]
            # 全体インデックスに変換（これを初期値とする）
            final_representative_index = cluster_indices[top_local_index]

            # --- g_value がある場合の優先選択ロジック ---
            if g_value is not None:
                found_flag = False
                # クラスタサイズが10未満の場合のエラー回避
                check_limit = min(10, len(center_list))

                for i in range(check_limit):
                    local_idx = center_list[i]
                    current_global_idx = cluster_indices[local_idx]

                    # 化合物名を取得して判定
                    # dataはDataFrame想定
                    compound_name = data.iloc[current_global_idx]['COMPOUND_NAME']
                    
                    if g_value.get(compound_name) == 1:
                        print(f"Representative for cluster {label} selected: {i}th candidate (ID: {current_global_idx}).")
                        final_representative_index = current_global_idx
                        found_flag = True
                        break
                
                if not found_flag:
                    print(f"cluster {label}: No point satisfying g_value found within top {check_limit}. Selecting the top 1.")
                    # found_flagがFalseなら、final_representative_index は初期値（1位）のままなのでOK

            representative_indices.append(final_representative_index)

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
        g_value = self._value_clusters(data, labels)
        center_list_dict = self._select_representatives(preprocessed_data, g, labels)
        representative_indices = self._check_representatives(data, center_list_dict, labels, g_value)
        if visualize:
            self._visualize(preprocessed_data, labels, representative_indices)
        return data.iloc[representative_indices], representative_indices, labels

    def value_clusters(self, data: pd.DataFrame, feature_cols: list=None):
        if feature_cols is not None:
            feature_data = data[feature_cols].values
        else:
            feature_data = data.values
        preprocessed_data = self._preprocess(feature_data)
        g = self._make_graph(preprocessed_data)
        labels = self._clustering(g)
        return self._value_clusters(data, labels)