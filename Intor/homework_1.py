import pandas as pd
import numpy as np

df = pd.read_csv('master_housing.csv')

q2 = len(df.columns)

cols = df.columns
has_na = []
for col in cols:
    if df[col].hasnans:
        has_na.append(col)

q4 = df['ocean_proximity'].nunique()
q5 = df[df['ocean_proximity'] == 'NEAR BAY']['median_house_value'].mean()

# Q6
mean_before = df['total_bedrooms'].mean()
df['total_bedrooms'].fillna(value=mean_before, inplace=True)
mean_after = df['total_bedrooms'].mean()
q6 = mean_before == mean_after

# Q7: Regression using Normal Equation

X = np.array(df.head()[['housing_median_age', 'total_rooms', 'total_bedrooms']])
XTX = np.matmul(X.T, X)
XTX_inv = np.linalg.inv(XTX)
y = np.array([950, 1300, 800, 1000, 1300])
w = np.dot(XTX_inv, X.T).dot(y)

