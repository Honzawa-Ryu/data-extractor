import numpy as np
import pandas as pd
import sklearn
from statsmodels.stats import diagnostic
from .utils import BaseFilter

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
    
