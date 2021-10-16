import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('multiplelr.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])], remainder= 'passthrough')
x = ct.fit_transform(x)

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_tr,y_tr)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

y_pred = regress.predict(x_te)
mae = mean_absolute_error(y_te,y_pred)
mse = mean_squared_error(y_te,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_te,y_pred)

print("Mean absolute error: "+str(mae))
print("Mean Square error: "+str(mse))
print("Root mean square error: "+str(rmse))
print("Accuracy score: "+str(r2))