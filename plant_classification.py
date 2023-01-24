import numpy as np  
import pandas as pd
import sklearn
from sklearn import datasets

iris = datasets.load_iris()

dados = iris.data 

dados = pd.DataFrame(dados)

classes = iris.target

classes = pd.DataFrame(classes)

df_iris = pd.concat([dados,classes],axis=1)

colnames = iris.feature_names

colnames.append('classification')

df_iris.columns = colnames

df_iris['classification']=df_iris['classification'].replace({0:'setosa', 1:'versicolor', 2:'virginica'})

counts = df_iris['classification'].value_counts()

print(df_iris)
print('-------------------------------------------')
print(counts)
print('-------------------------------------------')