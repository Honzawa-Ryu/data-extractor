import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import normalize
from statsmodels.stats import diagnostic
from utils import BaseFilter

class GaussianOutlierFilter(BaseFilter):
    """
    選定に用いる要素をリストで受け取り、平均から十分離れているデータのみを選定してくるフィルタ
    正規分布を仮定し、選定範囲として標準偏差の何倍の領域を取るかを指定する
    複数の要素を指定された際、少なくとも一つ、かすべてについて、かをboolで指定する
    Lilieforsの正規性検定を行い、正規性、対数正規性を検定する
    確認できなければIQR法によりデータの選別を行う
    複数ラベルに対し行う場合多種の手法を並列することをどうとらえるべきか……
    対数正規性と単純正規性の優先順位はこれでいいのか
    """
    def __init__(self, criteria: float, and_logic: bool=False):
        self.criteria = criteria
        self.and_logic = and_logic
    
    def fit_transform(self, data: pd.DataFrame, labels: list[str],) -> pd.DataFrame:
        mask_series = pd.Series(self.and_logic, index=data.index)        

        for label in labels:
            _, p_value = diagnostic.lilliefors(data[label].dropna())
            # 正規性の確認
            if p_value < 0.05:
                _, p_value = diagnostic.lilliefors(np.log(data[label].dropna().to_numpy()))
                # 対数正規性の確認
                if p_value < 0.05:
                    print(f'Alert! {label} の正規性が否定されました (p={p_value:.4f})。IQR法に切り替えます。')
                    Q1 = data[label].quantile(0.25)
                    Q3 = data[label].quantile(0.75)
                    IQR = Q3 - Q1

                    upper_bound = Q3 + IQR*self.criteria
                    lower_bound = Q1 - IQR*self.criteria

                    label_mask = (data[label] > upper_bound) | (data[label] < lower_bound)

                else:
                    print(f'{label} は対数正規性を確認できました。対数Zスコアを使用します')
                    mean = np.log(data[label].to_numpy()).mean()
                    std = np.log(data[label].to_numpy()).std()
                    
                    upper_bound = mean + std*self.criteria
                    lower_bound = mean - std*self.criteria

                    label_mask = (np.log(data[label]) > upper_bound) | (np.log(data[label]) < lower_bound)        

            else:
                print(f'{label} は正規性が確認できました。Zスコア法を使用します')

                mean = data[label].mean()
                std = data[label].std()
            
                upper_bound = mean + std*self.criteria
                lower_bound = mean - std*self.criteria

                label_mask = (data[label] > upper_bound) | (data[label] < lower_bound)

            if self.and_logic:
                mask_series = mask_series & (label_mask)
            else:
                mask_series = mask_series | (label_mask)
            
        print(f"Filtering completed {len(data)} -> {len(data[mask_series])}")
        
        return data[mask_series]
    
class ConditionFilter(BaseFilter):
    """
    指定した条件に合致するデータを抽出するフィルタ
    """
    def __init__(self, condition_dict: dict, and_logic: bool=True):
        self.condition_dict = condition_dict
        self.and_logic = and_logic
    
    def fit_transform(self, data: pd.DataFrame,) -> pd.DataFrame:
        mask_series = pd.Series(self.and_logic, index=data.index)

        for label, condition in self.condition_dict.items():
            label_mask = (data[label] == condition)
            if self.and_logic:
                mask_series = mask_series & label_mask
            else:
                mask_series = mask_series | label_mask
        
        print(f"Filtering completed {len(data)} -> {len(data[mask_series])}")
        
        return data[mask_series]

class PolarsConditionFilter(BaseFilter):
    def __init__(self, condition_dict: dict, and_logic: bool=True):
        self.condition_dict = condition_dict
        self.and_logic = and_logic

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        # 遅延評価式（Expression）のリストを作成
        expressions = [pl.col(col) == val for col, val in self.condition_dict.items()]
        
        if not expressions:
            return data

        # 論理演算を一括適用
        if self.and_logic:
            # すべての条件を満たす (all_horizontal)
            combined_condition = pl.all_horizontal(expressions)
        else:
            # いずれかの条件を満たす (any_horizontal)
            combined_condition = pl.any_horizontal(expressions)

        initial_len = len(data)
        filtered_data = data.filter(combined_condition)
        print(f"Filtering completed {initial_len} -> {len(filtered_data)}")
        
        return filtered_data
    
class NaNFilter(BaseFilter):
    """
    一定以上の個数NaNを含むデータを除外するフィルタ
    """
    def __init__(self, n: int=10):
        self.n = n
    
    def fit_transform(self, data: pd.DataFrame,) -> pd.DataFrame:
        initial_len = len(data)
        filtered_data = data[data.isna().sum(axis=1) < self.n]
        final_len = len(filtered_data)
        print(f"Filtering completed {initial_len} -> {final_len}")
        return filtered_data
    
class PolarsNaNFilter(BaseFilter):
    def __init__(self, n: int=10):
        self.n = n

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        # data.null_count() は列方向なので、sum_horizontal(pl.all().is_null()) を使う
        null_counts = pl.sum_horizontal(pl.all().is_null())
        
        filtered_data = data.filter(null_counts < self.n)
        
        print(f"Filtering completed {len(data)} -> {len(filtered_data)}")
        return filtered_data

class CosSimFilter(BaseFilter):
    """
    データ全体のコサイン類似度の平均と標準偏差を計算し、
    各実験群のコサイン類似度の平均が全体の平均から
    指定された標準偏差倍数以上離れている場合にその実験群を選定するフィルタ
    """
    def __init__(self, criteria: float=3.0, pass_ratio: float=1.0):
        self.criteria = criteria
        self.pass_ratio = pass_ratio
    
    def _calculate_average(self, data: pd.DataFrame, feature_cols: list=None):
        if feature_cols is not None:
            feature_data = data[feature_cols].values
        else:
            feature_data = data.values

        n = feature_data.shape[0]
        X_norm = normalize(feature_data, axis=1, norm='l2')
        sum_vector = X_norm.sum(axis=0)
        squared_norm_sum = np.dot(sum_vector, sum_vector)
        sum_of_similarities = squared_norm_sum - n
        if n > 1:
            mean_similarity = sum_of_similarities / (n * (n - 1))
        else:
            # データ点が1つ以下の場合は類似度を0とする
            mean_similarity = 0.0        
        return mean_similarity
    
    def _calculate_std(self, data: pd.DataFrame, feature_cols: list=None, n_samples: int=100_000):
        if feature_cols is not None:
            feature_data = data[feature_cols].values
        else:
            feature_data = data.values
        
        n = feature_data.shape[0]
        X_norm = normalize(feature_data, axis=1, norm='l2')

        idx_a = np.random.randint(0, n, size=n_samples)
        idx_b = np.random.randint(0, n, size=n_samples)

        mask = idx_a != idx_b
        idx_a = idx_a[mask]
        idx_b = idx_b[mask]

        vecs_a = X_norm[idx_a]
        vecs_b = X_norm[idx_b]

        similarities = np.einsum('ij,ij->i', vecs_a, vecs_b)

        variance = np.var(similarities)
        std_dev = np.std(similarities)
        return std_dev

    def fit_transform(self, data: pd.DataFrame, feature_cols: list=None) -> pd.DataFrame:
        mean_similarity = self._calculate_average(data, feature_cols)
        std_dev = self._calculate_std(data, feature_cols)
        selected_indices = []

        for (exp_id, group_id), group_df in data.groupby(['EXP_ID', 'GROUP_ID']):
            group_mean = self._calculate_average(group_df, feature_cols)
            z_score = (group_mean - mean_similarity) / std_dev

            if z_score >= self.criteria:
                selected_indices.extend(group_df.index.tolist())

        print(f"Filtering completed {len(data)} -> {len(selected_indices)}")
        return data.loc[selected_indices]


