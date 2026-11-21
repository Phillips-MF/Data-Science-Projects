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
building_type_labels = train_df["Building Type"].unique()
building_type_values = train_df["Building Type"].value_counts().unique()
# %%
building_bar = plt.bar(x=building_type_labels, height=building_type_values)
plt.bar_label(building_bar, padding=3)
plt.show()
# %%
building_energy_mean = train_df.groupby("Building Type")["Energy Consumption"].mean()
print(type(building_energy_mean))
building_mean_values = building_energy_mean.values
print(type(building_type_values))
# %%
building_mean_bar = plt.bar(x=building_type_labels, height=building_mean_values)
plt.bar_label(building_mean_bar, padding=3)
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
train_df["occupant_per_square"] = train_df["Square Footage"] / train_df["Number of Occupants"]
train_df["appliance_per_square"] = train_df["Square Footage"] / train_df["Appliances Used"]
# %%
train_df.head(5)
# %%
# Verificando a correlação com as novas variáveis
train_num = train_df.select_dtypes(include=(float, int))
print(train_num.columns)
# %%
scatter_matrix(train_num, figsize=(20,12))
# %%[markdown]
# Bem, aparentemente as combinações não renderam nenhum fruto portanto podem ser retiradas
# %%
train_df.drop("occupant_per_square", inplace=True, axis=1)
train_df.drop("appliance_per_square", inplace=True, axis=1)
train_df.head(5)
# %%[markdown]
# Preparação dos dados para algoritmos de ML
# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# %%
preprocessing = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["Building Type", "Day of Week"])
    ],
        remainder="passthrough",
        verbose_feature_names_out=False
)
# %%
preprocessing.set_output(transform="pandas")
# %%
train_df_processed = preprocessing.fit_transform(train_df)
# %%[markdown]
# Aplicando o 
# %%
