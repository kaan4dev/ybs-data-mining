import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

X = data.drop("charges", axis=1)
y = data[["charges"]]
target1 = data.iloc[:, 2:-1]

cat_cols = ["sex", "smoker", "region"]
pre = ColumnTransformer([("cat", OneHotEncoder(drop="first"), cat_cols)], remainder="passthrough")

lr = Pipeline([("pre", pre), ("model", LinearRegression())])
rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train.values.ravel())

lr_pred_train = lr.predict(X_train)
lr_pred_test = lr.predict(X_test)
rf_pred_train = rf.predict(X_train)
rf_pred_test = rf.predict(X_test)

lr_mae_train = mean_absolute_error(y_train, lr_pred_train)
lr_mae_test = mean_absolute_error(y_test, lr_pred_test)
lr_mse_train = mean_squared_error(y_train, lr_pred_train)
lr_mse_test = mean_squared_error(y_test, lr_pred_test)
lr_rmse_train = np.sqrt(lr_mse_train)
lr_rmse_test = np.sqrt(lr_mse_test)
lr_mape_train = mean_absolute_percentage_error(y_train, lr_pred_train)
lr_mape_test = mean_absolute_percentage_error(y_test, lr_pred_test)

rf_mae_train = mean_absolute_error(y_train, rf_pred_train)
rf_mae_test = mean_absolute_error(y_test, rf_pred_test)
rf_mse_train = mean_squared_error(y_train, rf_pred_train)
rf_mse_test = mean_squared_error(y_test, rf_pred_test)
rf_rmse_train = np.sqrt(rf_mse_train)
rf_rmse_test = np.sqrt(rf_mse_test)
rf_mape_train = mean_absolute_percentage_error(y_train, rf_pred_train)
rf_mape_test = mean_absolute_percentage_error(y_test, rf_pred_test)

print("===== LINEAR REGRESSION RESULTS =====")
print("MAE Train:", lr_mae_train)
print("MAE Test:", lr_mae_test)
print("MSE Train:", lr_mse_train)
print("MSE Test:", lr_mse_test)
print("RMSE Train:", lr_rmse_train)
print("RMSE Test:", lr_rmse_test)
print("MAPE Train:", lr_mape_train)
print("MAPE Test:", lr_mape_test)

print("\n===== RANDOM FOREST RESULTS =====")
print("MAE Train:", rf_mae_train)
print("MAE Test:", rf_mae_test)
print("MSE Train:", rf_mse_train)
print("MSE Test:", rf_mse_test)
print("RMSE Train:", rf_rmse_train)
print("RMSE Test:", rf_rmse_test)
print("MAPE Train:", rf_mape_train)
print("MAPE Test:", rf_mape_test)
