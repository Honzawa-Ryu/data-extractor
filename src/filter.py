import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import normalize, StandardScaler
from scipy.stats import norm
from statsmodels.stats import diagnostic
from utils import BaseFilter

import matplotlib.pyplot as plt

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
    def __init__(self, criteria: float = 3.0, strategy: str = "IQR", random_state: int = 42):
        self.criteria = criteria
        self.strategy = strategy
        self.random_state = random_state

    def _get_normalized_features(self, data: pd.DataFrame, feature_cols: list = None):
        """特徴量を取得して正規化する共通メソッド"""
        if feature_cols is not None:
            feature_data = data[feature_cols].values
        else:
            feature_data = data.values
        # L2正規化 (コサイン類似度の計算準備)
        feature_data = StandardScaler().fit_transform(feature_data)
        return normalize(feature_data, axis=1, norm='l2')

    def _calculate_average(self, X_norm: np.ndarray):
        """
        正規化済みデータから平均コサイン類似度を高速に計算
        計算量: O(N * D)
        """
        n = X_norm.shape[0]
        if n <= 1:
            return 0.0

        # 全ベクトルの和のノルムの二乗を利用して、ペアごとの内積の総和を計算
        # sum(v_i . v_j) = ||sum(v)||^2
        # ここには i=j の要素(1.0)が含まれるため、それを引く
        sum_vector = X_norm.sum(axis=0)
        squared_norm_sum = np.dot(sum_vector, sum_vector)
        
        sum_of_similarities = squared_norm_sum - n
        
        # ペアの数 n*(n-1) で割る
        mean_similarity = sum_of_similarities / (n * (n - 1))
        return mean_similarity

    def _calculate_criteria(self, X_norm: np.ndarray, strategy: str = "IQR", n_samples: int = 1_000_000):
        """
        Cosine Similarityの閾値を計算。 Strategyにより計算方法を変更。
        データ数が多い場合はサンプリング、少ない場合は全計算。
        """
        n = X_norm.shape[0]
        if n <= 1:
            return 0.0

        # ペアの総数
        total_pairs = n * (n - 1)

        # ペア数がサンプル数より少ない、あるいは現実的なサイズなら全計算して正確な値を出す
        # (例: データ数が2000以下ならペアは約400万以下なので計算可能とする等、環境に合わせて調整)
        if total_pairs < n_samples:
            # 行列積で一気に計算 (メモリ注意)
            sim_matrix = np.dot(X_norm, X_norm.T)
            # 対角成分(自分自身)を除外して1次元配列化
            mask = ~np.eye(n, dtype=bool)
            similarities = sim_matrix[mask]
        else:
            # サンプリングによる推定
            rng = np.random.RandomState(self.random_state)
            
            # n_samples分のペアをランダムに生成
            idx_a = rng.randint(0, n, size=n_samples)
            idx_b = rng.randint(0, n, size=n_samples)

            # 自分自身とのペアを除外
            mask = idx_a != idx_b
            idx_a = idx_a[mask]
            idx_b = idx_b[mask]

            if len(idx_a) == 0:
                return 0.0

            vecs_a = X_norm[idx_a]
            vecs_b = X_norm[idx_b]

            # einsumで高速に内積計算
            similarities = np.einsum('ij,ij->i', vecs_a, vecs_b)
        
        #分布の大雑把な確認用、コメントアウト
        _, p_value = diagnostic.lilliefors(similarities)
        print(f"Cosine Similarity distribution Lilliefors test p-value: {p_value:.4f}")
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.hist(similarities, bins=50)
        ax.set_title('Cosine Similarity histogram')
        ax.set_xlabel('Similarity')
        ax.set_ylabel('Frequency')
        fig.show()

        if strategy == "IQR":
            median = np.median(similarities)
            Q1 = np.percentile(similarities, 25)
            Q3 = np.percentile(similarities, 75)
            IQR = Q3 - Q1
            return median + IQR*self.criteria
        elif strategy == "std":
            mean = np.mean(similarities)
            std = np.std(similarities)
            return mean + std*self.criteria
        elif strategy == "percentile":
            percentile_in_normal_dist = norm.cdf(self.criteria) * 100
            upper_perc = np.percentile(similarities, percentile_in_normal_dist)
            return upper_perc
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Choose from 'IQR', 'std', or 'percentile'.")

    def fit_transform(self, data: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
        # 全体データの正規化
        X_norm_all = self._get_normalized_features(data, feature_cols)
        
        # 全体の平均と標準偏差を計算
        # mean_similarity = self._calculate_average(X_norm_all)
        criteria_dev = self._calculate_criteria(X_norm_all, strategy=self.strategy)
        
        print(f"Global criteria calculated: std_dev >= {criteria_dev:.4f}")

        if criteria_dev == 0:
            print("Standard deviation is 0. Returning empty or all data depending on policy.")
            return data # 変化がない場合はそのまま返す等の処理

        selected_indices = []

        # グループごとの処理
        # note: groupbyループ内で毎回データ抽出・正規化を行うと重いため、
        # indexを使って X_norm_all からスライスする方が高速ですが、
        # 実装の複雑さを避けるため、ここでは元のロジックを尊重しつつ整理します。
        
        for (exp_id, group_id), group_indices in data.groupby(['EXP_ID', 'GROUP_ID']).indices.items():
            # グループに対応する正規化済みベクトルを抽出
            # (iloc等は遅いのでnumpy配列から直接取る)
            group_vectors = X_norm_all[group_indices]
            
            # グループ内平均類似度の計算
            group_mean = self._calculate_average(group_vectors)
            
            # print(f"Group (EXP_ID={exp_id}, GROUP_ID={group_id}): mean_cos_sim = {group_mean:.4f}")
            # 閾値と比較して選定
            if group_mean >= criteria_dev:
                selected_indices.extend(group_indices)
                # print(f"  -> Selected (|{group_mean:.4f}| >= {criteria_dev:.4f})")
            else:
                # print(f"  -> Not selected (|{group_mean:.4f}| < {criteria_dev:.4f})")
                pass
        print(f"Filtering completed {len(data)} -> {len(selected_indices)}")
        
        # 元のインデックスを保持したまま抽出
        return data.iloc[selected_indices]

class L2NormFilter(BaseFilter):
    """
    データ全体のL2ノルムの平均と標準偏差を計算し、
    各実験群のL2ノルムの平均が全体の平均から
    指定された標準偏差倍数以上離れている場合にその実験群を選定するフィルタ
    """
    def __init__(self, criteria: float = 3.0, strategy: str = "IQR", random_state: int = 42):
        self.criteria = criteria
        self.strategy = strategy
        self.random_state = random_state
        self.scaler = StandardScaler()

    def _get_normalized_features(self, data: pd.DataFrame, feature_cols: list = None):
        """特徴量を取得して正規化する共通メソッド"""
        if feature_cols is not None:
            feature_data = data[feature_cols].values
        else:
            feature_data = data.values
        # L2正規化 (コサイン類似度の計算準備)
        log_data = np.log1p(feature_data)
        return self.scaler.fit_transform(log_data)
    
    def _calculate_average(self, X_norm: np.ndarray):
        """
        正規化済みデータから平均L2ノルムを高速に計算
        計算量: O(N * D)
        """
        n = X_norm.shape[0]
        if n <= 1:
            return 0.0

        # 全ベクトルの和のノルムの二乗を利用して、ペアごとの内積の総和を計算
        # sum(||v_i - v_j||^2) = n * sum(||v_i||^2) - ||sum(v)||^2
        sum_vector = X_norm.sum(axis=0)
        squared_norm_sum = np.dot(sum_vector, sum_vector)
        individual_norms = np.sum(np.einsum('ij,ij->i', X_norm, X_norm))
        
        sum_of_distances = 2 * n * individual_norms - 2 * squared_norm_sum
        
        # ペアの数 n*(n-1) で割る
        mean_distance = sum_of_distances / (n * (n - 1))
        return mean_distance

    def _calculate_criteria(self, X_norm: np.ndarray, strategy: str = "IQR", n_samples: int = 1_000_000):
        """
        L2ノルムの閾値を計算。 Strategyにより計算方法を変更。
        データ数が多い場合はサンプリング、少ない場合は全計算。
        """
        n = X_norm.shape[0]
        if n <= 1:
            return 0.0

        # ペアの総数
        total_pairs = n * (n - 1)

        # ペア数がサンプル数より少ない、あるいは現実的なサイズなら全計算して正確な値を出す
        # (例: データ数が2000以下ならペアは約400万以下なので計算可能とする等、環境に合わせて調整)
        if total_pairs < n_samples:
            # 対角成分(自分自身)を除外して1次元
            mask = ~np.eye(n, dtype=bool)
            similarities = np.power(np.linalg.norm(X_norm[:, np.newaxis] - X_norm[np.newaxis, :], axis=2)[mask], 2)
        else:
            # サンプリングによる推定
            rng = np.random.RandomState(self.random_state)
            
            # n_samples分のペアをランダムに生成
            idx_a = rng.randint(0, n, size=n_samples)
            idx_b = rng.randint(0, n, size=n_samples)

            # 自分自身とのペアを除外
            mask = idx_a != idx_b
            idx_a = idx_a[mask]
            idx_b = idx_b[mask]

            if len(idx_a) == 0:
                return 0.0

            vecs_a = X_norm[idx_a]
            vecs_b = X_norm[idx_b]

            # einsumで高速に内積計算
            similarities = np.power(np.linalg.norm(vecs_a - vecs_b, axis=1), 2)
        
        #分布の大雑把な確認用、コメントアウト
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.hist(similarities, bins=50)
        ax.set_title('L2norm Similarity histogram')
        ax.set_xlabel('Similarity')
        ax.set_ylabel('Frequency')
        fig.show()

        if strategy == "IQR":
            median = np.median(similarities)
            Q1 = np.percentile(similarities, 25)
            Q3 = np.percentile(similarities, 75)
            IQR = Q3 - Q1
            return median - IQR*self.criteria
        elif strategy == "std":
            mean = np.mean(similarities)
            std = np.std(similarities)
            return mean - std*self.criteria
        elif strategy == "percentile":
            percentile_in_normal_dist = 100 - norm.cdf(self.criteria) * 100
            print(f"Percentile for L2 norm criteria: {percentile_in_normal_dist}")
            upper_perc = np.percentile(similarities, percentile_in_normal_dist)
            return upper_perc
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Choose from 'IQR', 'std', or 'percentile'.")

    def fit_transform(self, data: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
        # 全体データの正規化
        X_norm_all = self._get_normalized_features(data, feature_cols)
        
        # 全体の平均と標準偏差を計算
        # mean_similarity = self._calculate_average(X_norm_all)
        criteria_dev = self._calculate_criteria(X_norm_all, strategy=self.strategy)
        
        print(f"Global criteria calculated: criteria_dev <= {criteria_dev:.4f}")

        if criteria_dev == 0:
            print("Standard deviation is 0. Returning empty or all data depending on policy.")
            return data # 変化がない場合はそのまま返す等の処理

        selected_indices = []

        # グループごとの処理
        # note: groupbyループ内で毎回データ抽出・正規化を行うと重いため、
        # indexを使って X_norm_all からスライスする方が高速ですが、
        # 実装の複雑さを避けるため、ここでは元のロジックを尊重しつつ整理します。
        
        for (exp_id, group_id), group_indices in data.groupby(['EXP_ID', 'GROUP_ID']).indices.items():
            # グループに対応する正規化済みベクトルを抽出
            # (iloc等は遅いのでnumpy配列から直接取る)
            group_vectors = X_norm_all[group_indices]
            
            # グループ内平均類似度の計算
            group_mean = self._calculate_average(group_vectors)
            
            # print(f"Group (EXP_ID={exp_id}, GROUP_ID={group_id}): mean_cos_sim = {group_mean:.4f}")
            # 閾値と比較して選定
            if group_mean <= criteria_dev:
                selected_indices.extend(group_indices)
                # print(f"  -> Selected (|{group_mean:.4f}| >= {criteria_dev:.4f})")
            else:
                # print(f"  -> Not selected (|{group_mean:.4f}| < {criteria_dev:.4f})")
                pass
        print(f"Filtering completed {len(data)} -> {len(selected_indices)}")
        
        # 元のインデックスを保持したまま抽出
        return data.iloc[selected_indices]



