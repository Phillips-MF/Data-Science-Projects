# %%[markdown]
# # Informações importantes
# métrica de uso -> Erro médio quadrado
# %%[markdown]
# # Preparando o ambiente
# %%
from pathlib import Path
# %%
ROOT = Path(__file__).resolve().parent.parent
# %%[markdown]
# # Importação dos dados
# %%
train_path = ROOT / "data/train_energy_data.csv"
test_path = ROOT / "data/test_energy_data.csv"
import pandas as pd
train_df = pd.read_csv(train_path, header=0)
test_df = pd.read_csv(test_path, header=0)
# %%[markdown]
# # Conhecendo os dados
# %%
train_df.head()
# %%
train_df.info()
# %%
train_df.describe()
# %%
import matplotlib.pyplot as plt
# %%
train_df.hist(bins=50, figsize=(12,10))
# %%
from pandas.plotting import scatter_matrix
train_num = train_df.select_dtypes(include=(float, int))
scatter_matrix(train_num, figsize=(12,10))
plt.show()
# %%
train_num.corr()
# %%[markdown]
# Existem valores de correlação muito forte entre Consumo de Energia e Metragem  
#  
# E outras fracas entre Consumo de energia e número de habitantes e aparelhos usados.
#   
# Diagnóstico inicial: Treinar 2 modelos de ML com regressão linear, um com os atributos fracos e outro sem
# %%[markdown]
# # Testando combinações entre os atributos
# %%
