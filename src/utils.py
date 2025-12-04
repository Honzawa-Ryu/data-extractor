import numpy as np
import pandas as pd

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

