import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class BaseFilter():
    """
    フィルタとしての実装を予定
    クラスとして作成しているが静的なメソッドか動的なメソッドかも考えた方がいいかもしれない

    __init__
        初期化を行う。引数としてはクライテリアを取るのがいいか（標準偏差の何倍までを落とす、とか）。
    
    fit
        データのフィルタリングを行う。入れられたデータに対して統計量を保持する形とかがいいかも。
        フィルタをかける要素をリストで渡す、などができると便利でしょうね
    """
    def __init__(self):
        raise NotImplementedError
    
    def fit_transformD():
        raise NotImplementedError

class BaseExtractor():
    """
    抽出器として作成

    __init__
        初期化を行う。あんまりイメージがわいていない
    
    fit
        データの抽出を行う。具体的には、クラスタリングを行い、寄与の高い要素を出力、または重要度の高いクラスタを返す？
        病理所見とかの離散データを受け取ってOnehot表現にして保持、とかかも
        引数としてはクラスタリング手法、とかになるかなと
        手法ごとにクラスを分けるのも全然アリだと思っていますが
    """
    def __init__(self):
        raise NotImplementedError
    
    def preprocess(self):
        raise NotImplementedError
    
    def clustering(self):
        raise NotImplementedError
    
    def extract(self):
        raise NotImplementedError
        
    def fit_transform(self):
        raise NotImplementedError

class UMAPVisualizer():
    """
    UMAPによる可視化を行うクラス
    主に2次元、3次元への可視化を担当する
    """
    def __init__(self, n_components: int=2, n_neighbors: int=15, min_dist: float=0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def preprocess(self, data: pd.DataFrame, start_index: int=0, end_index: int=None):
        
        if end_index == None:
            end_index = data.shape[1]
        data_to_process = data.iloc[:, start_index:end_index]

        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data_to_process)

        # 標準化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)

        return data_scaled

    def fit_transform(self, data: pd.DataFrame, start_index: int=0, end_index: int=None):
        """
        UMAPによる次元削減と可視化を行う
        """
        labels = data["COMPOUND_NAME"].tolist()
        unique_elements = sorted(list(set(labels)))

        mapping_dict = {element: i for i, element in enumerate(unique_elements)}
        print(f"マッピング辞書: {mapping_dict}")

        encoded_list = [mapping_dict[item] for item in labels]
        labels = encoded_list
        data = self.preprocess(data, start_index, end_index)
        reducer = umap.UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors, min_dist=self.min_dist)
        embedding = reducer.fit_transform(data)
        plt.figure(figsize=(10, 6))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
        plt.title('UMAP with compounds name')
        plt.legend()
        plt.show()

        return embedding