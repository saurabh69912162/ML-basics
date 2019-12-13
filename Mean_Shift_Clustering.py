from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split


df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body','name','boat','ticket','home.dest'],1, inplace=True)
df.fillna('0.0', inplace=True)
df['cabin'].fillna('0.0', inplace=True)


print(len(df))

print(df['cabin'].unique())
print(len(df['cabin'].unique()))

cabin_encoder = preprocessing.LabelEncoder()
cabin = df['cabin']
cabin_cat = cabin_encoder.fit_transform(cabin)
df.drop(['cabin'],1, inplace=True)
df['cabin_encoded'] = cabin_cat


sex_encoder = preprocessing.LabelEncoder()
sex = df['sex']
sex_cat = sex_encoder.fit_transform(sex)
print(sex_encoder.classes_)
df.drop(['sex'],1, inplace=True)
df['sex_encoded'] = sex_cat

df['embarked'].fillna('0.0', inplace=True)
print(df['embarked'].unique())
print(df['embarked'])

embarked_encoder = preprocessing.LabelEncoder()
embarked = df['embarked']
embarked_cat = embarked_encoder.fit_transform(embarked)
print(embarked_encoder.classes_)
df.drop(['embarked'],1, inplace=True)
df['embarked_encoded'] = embarked_cat


print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
n_clusters_ = len(np.unique(labels))

cluster_centers = clf.cluster_centers_
original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]
    
survival_rates = {}

for i in range(n_clusters_): 
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]
    survival_rate = len(survival_cluster) / len(temp_df)
    #print(i,survival_rate)
    survival_rates[i] = survival_rate
    
print(survival_rates)
print(original_df[ (original_df['cluster_group']==0) ].describe())
print(original_df[ (original_df['cluster_group']==1) ].describe())
print(original_df[ (original_df['cluster_group']==2) ].describe())
print(original_df[ (original_df['cluster_group']==3) ].describe())
print(original_df[ (original_df['cluster_group']==4) ].describe())
# print(original_df[ (original_df['cluster_group']==5) ].describe())

# cluster_0 = original_df[ (original_df['cluster_group']==2)]
# cluster_0_fc = cluster_0[ (cluster_0['pclass']==1) ]
# cluster_0_fc
