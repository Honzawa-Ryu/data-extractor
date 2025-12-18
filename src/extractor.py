import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import umap
import matplotlib.pyplot as plt

from utils import BaseExtractor # 環境に合わせてコメントアウトを外してください
class BaseExtractor: pass # 動作確認用のダミー

class LeidenRepresentativeSelector(BaseExtractor):
    def __init__(self, n_neighbors: int=10, resolution: float=1.0, 
                 pca: bool=False, pca_threshold: float=0.9, strategy: str='mean'):
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.pca = pca
        self.pca_threshold = pca_threshold
        self.strategy = strategy
        
        # 計算結果を保持するメンバ変数
        self.labels_ = None
        self.preprocessed_data_ = None
        self.g_ = None

    def _preprocess(self, data: np.ndarray):
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)
        
        if self.pca and data_scaled.shape[1] > 1:
            pca = PCA(n_components=self.pca_threshold)  
            data_scaled = pca.fit_transform(data_scaled)
            
        return data_scaled
    
    def _make_graph(self, data: np.ndarray):
        knn_graph = kneighbors_graph(data, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
        sources, targets = knn_graph.nonzero()
        edges = list(zip(sources, targets))
        return ig.Graph(edges=edges, directed=False)

    def _clustering(self, g: ig.Graph):
        partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=self.resolution)
        labels = np.array(partition.membership)
        print(f"Clustering completed: {len(set(labels))} clusters found.")
        return labels

    # 【改善1】共通処理をまとめるメソッドを作成
    def _execute_clustering_pipeline(self, feature_data: np.ndarray):
        """前処理からクラスタリングまでを一括で行う"""
        self.preprocessed_data_ = self._preprocess(feature_data)
        self.g_ = self._make_graph(self.preprocessed_data_)
        self.labels_ = self._clustering(self.g_)
        return self.preprocessed_data_, self.g_, self.labels_
    
    # 【改善3】Pandasのgroupbyを使って高速化
    def _value_clusters(self, data: pd.DataFrame, cluster_labels: np.ndarray):
        """
        同一物質(COMPOUND_NAME)がいくつのクラスタに分散しているか確認する。
        戻り値: {compound_name: クラスタ数の整数} (判定用に変更)
        """
        # データフレームに一時的にクラスタラベルを付与
        df_temp = data.copy()
        df_temp['Cluster_Label'] = cluster_labels
        
        # groupbyで集計 (setの長さ = 所属するクラスタの種類数)
        compound_stats = df_temp.groupby(['EXP_ID', 'GROUP_ID'])['Cluster_Label'].agg(lambda x: len(set(x)))
        
        # 全体の分散具合（Over_partition）の計算
        avg_clusters = compound_stats.mean()
        print(f"Over_partition (Avg clusters per compound): {avg_clusters:.2f}")
        
        # {化合物名: クラスタ数} の辞書を返す
        return compound_stats.to_dict()

    def _select_representatives(self, data: np.ndarray, g: ig.Graph, labels: np.ndarray):
        unique_labels = np.unique(labels)
        center_list_dict = {}

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_data = data[cluster_indices]

            if self.strategy == 'mean':
                cluster_center = np.mean(cluster_data, axis=0)
                distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
                center_list = np.argsort(distances) # 昇順

            elif self.strategy == 'median':
                cluster_center = np.median(cluster_data, axis=0)
                distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
                center_list = np.argsort(distances) 

            elif self.strategy in ['degree', 'closeness', 'betweenness']:
                g_sub = g.subgraph(cluster_indices)
                if self.strategy == 'degree':
                    scores = g_sub.degree()
                elif self.strategy == 'closeness':
                    scores = g_sub.closeness()
                elif self.strategy == 'betweenness':
                    scores = g_sub.betweenness()
                center_list = np.argsort(-np.array(scores)) # 降順

            else:
                raise ValueError(f"Invalid strategy: {self.strategy}.")
            
            center_list_dict[label] = center_list

        return center_list_dict

    def _check_representatives(self, data: pd.DataFrame, center_list_dict: dict, labels: np.ndarray, g_value: dict=None):
        representative_indices = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            center_list = center_list_dict[label] 
            
            # デフォルト：1位を選択
            top_local_index = center_list[0]
            final_representative_index = cluster_indices[top_local_index]

            if g_value is not None:
                check_limit = min(10, len(center_list))
                found_flag = False

                for i in range(check_limit):
                    local_idx = center_list[i]
                    current_global_idx = cluster_indices[local_idx]
                    
                    compound_name = tuple(data.iloc[current_global_idx][["EXP_ID", "GROUP_ID"]])
                    
                    if g_value.get(compound_name) == 1:
                        print(f"Representative for cluster {label}: Found clean compound '{compound_name}' at rank {i}")
                        final_representative_index = current_global_idx
                        found_flag = True
                        break
                
                if not found_flag:
                    print(f"Cluster {label}: No clean compound found in top {check_limit}.")

            representative_indices.append(final_representative_index)

        print(f"{len(representative_indices)} representative points selected.")
        return representative_indices

    def _visualize(self, data: np.ndarray, labels: np.ndarray, representative_indices: list):
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(data)

        plt.figure(figsize=(10, 6))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5, alpha=0.6)
        rep_embedding = embedding[representative_indices]
        plt.scatter(rep_embedding[:, 0], rep_embedding[:, 1], c='black', s=50, marker='x', label='Representatives')
        plt.title('Leiden Clustering with Representatives')
        plt.legend()
        plt.show()
    
    def fit_transform(self, data: pd.DataFrame, feature_cols: list=None, visualize: bool=False):
        # 特徴量の抽出
        if feature_cols is not None:
            feature_data = data[feature_cols].values
        else:
            feature_data = data.values
            
        # 【改善1】共通パイプラインの実行
        preprocessed_data, g, labels = self._execute_clustering_pipeline(feature_data)
        
        # 評価値計算と代表点選択
        # g_value は {化合物名: 所属クラスタ数} の辞書として返される
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
        
        # 【改善1】ここでも共通パイプラインを使用
        _, _, labels = self._execute_clustering_pipeline(feature_data)
        
        return self._value_clusters(data, labels)