import pandas as pd
import numpy as np


model_data=pd.read_csv("model_data.csv")
ipl_data=pd.read_csv("edited.csv")

k=ipl_data.groupby(['match_id','innings']).total_runs.sum()
k=k.reset_index()
k=k.drop(columns=['match_id','innings'])
model_data=model_data[['match_id','innings','venue','batting_team','bowling_team','striker','bowler']]
model_data=model_data.merge(k,left_index=True,right_index=True)
model_data=model_data.rename(columns={'striker':'batsmen'})
model_data=model_data.drop(columns=['match_id'])



dataset = model_data
dataset.head()

X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

import joblib
joblib.dump(regressor,"randomforest_model.joblib")