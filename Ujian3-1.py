import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv', index_col=False)
df.drop('Unnamed: 0', axis=1, inplace=True)
# print(df)

##Target
df['target'] = df.apply(
    lambda x : 1 if (x['Age'] <= 25) and (x['Overall'] >= 80) and (x['Potential'] >= 80) else 0, axis=1
)

##Split
xtr, xts, ytr, yts = train_test_split(
    df[['Age', 'Overall', 'Potential']], df['target'], test_size = .1
)

##Logistic Regression + val score
model = LogisticRegression(solver='lbfgs', multi_class='auto')
print(model.fit(xtr, ytr))
print(np.mean(cross_val_score(LogisticRegression(solver='lbfgs', multi_class='auto'), xtr, ytr, cv=10)))

##DecisionTree
model1 = DecisionTreeClassifier()
print(model1.fit(xtr,ytr))
print(np.mean(cross_val_score(DecisionTreeClassifier(), xtr, ytr, cv=10)))

##RandomForest
model2 = RandomForestClassifier(n_estimators=100)
print(model2.fit(xtr,ytr))
print(np.mean(cross_val_score(RandomForestClassifier(n_estimators=3),xtr,ytr, cv=10)))

## Model terbaik adalah Decision Tree Classifier yaitu 1.0

dfPlayer = pd.DataFrame({
    'Name':['Andik Vermansyah','Awan Setho Raharjo','Bambang Pamungkas','Cristian Gonzales','Egy Maulana Vikri','Evan Dimas','Febri Hariyadi','Hansamu Yama Pranata','Septian David Maulana','Stefano Lilipaly'],
    'Age':[27,22,38,43,18,24,23,24,22,29],
    'Overall':[87,75,85,90,88,85,77,82,83,88],
    'Potential':[90,83,75,85,90,87,80,85,80,86]
})
# print(dfPlayer)

## MODEL untuk prediksi menggunakan Decision Tree Classifier
dfPlayer['Predict'] = model1.predict(dfPlayer.drop('Name',axis=1))
dfPlayer['Predict'] = dfPlayer['Predict'].apply(lambda i: 'Target' if i==1 else 'Non Target')
print(dfPlayer)