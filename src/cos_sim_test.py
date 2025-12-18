#%%
from filter import GaussianOutlierFilter, ConditionFilter, NaNFilter, CosSimFilter, L2NormFilter
from extractor import LeidenRepresentativeSelector
from utils import UMAPVisualizer

import pandas as pd
import polars as pl
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt


#%%
# データ読み込み
data = pd.read_csv("/workspace/01_data-extractor/data/TG_GATE_tables/merged_animal_data_final.csv", encoding='shift-jis')
feature = ['TERMINAL_BW(g)', 'LIVER(g)', 'KIDNEY_TOTAL(g)', 'KIDNEY_R(g)', 'KIDNEY_L(g)', 'ALP(IU/L)', 'TC(mg/dL)', 'TG(mg/dL)', 'PL(mg/dL)', 'TBIL(mg/dL)', 'DBIL(mg/dL)', 'GLC(mg/dL)', 'BUN(mg/dL)', 'CRE(mg/dL)', 'Na(meq/L)', 'K(meq/L)', 'Cl(meq/L)', 'Ca(mg/dL)', 'IP(mg/dL)', 'TP(g/dL)', 'RALB(g/dL)', 'A/G', 'AST(IU/L)', 'ALT(IU/L)', 'LDH(IU/L)', 'GTP(IU/L)', 'RBC(x10_4/ul)', 'Hb(g/dL)', 'Ht(%)', 'MCV(fL)', 'MCH(pg)', 'MCHC(%)', 'Ret(%)', 'Plat(x10_4/uL)', 'WBC(x10_2/uL)', 'Neu(%)', 'Eos(%)', 'Bas(%)', 'Mono(%)', 'Lym(%)', 'PT(s)', 'APTT(s)', 'Fbg(mg/dL)']


#%%
nanfilter = NaNFilter(n=1)
data_filtered = nanfilter.fit_transform(data)


#%%
condition_filter = ConditionFilter({"DOSE_LEVEL": "High", "SINGLE_REPEAT_TYPE": "Repeat", "ADMINISTRATION_ROUTE_TYPE": "Gavage", "SACRIFICE_PERIOD": "29 day"}, and_logic=True)
conditioned_data = condition_filter.fit_transform(data_filtered)


#%%
cossimfilter = CosSimFilter(criteria=3.0, strategy="std")
data_cossim_filtered = cossimfilter.fit_transform(data_filtered, feature_cols=feature)
print(data_cossim_filtered[["EXP_ID", "GROUP_ID"]])


#%%
l2normfilter = L2NormFilter(criteria=2.9, strategy="percentile")
data_l2norm_filtered = l2normfilter.fit_transform(data_filtered, feature_cols=feature)
print(data_l2norm_filtered[["EXP_ID", "GROUP_ID"]])


# %%
data_l2norm_filtered.to_csv("/workspace/01_data-extractor/data/TG_GATE_tables/filtered_l2norm_data.csv", index=False)
# %%
