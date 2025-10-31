import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

cali = fetch_california_housing(as_frame=True)
data = cali.frame
inputs = data.drop("MedHouseVal", axis=1)
target2 = data[["MedHouseVal"]]
target1 = data.iloc[:, 2:-1]

X_train, X_test, y_train, y_test = train_test_split(inputs, target2, test_size=0.2, random_state=42)

lr = LinearRegression()
dtr = DecisionTreeRegressor(random_state=42)

lr.fit(X_train, y_train)
dtr.fit(X_train, y_train)

pred_train = lr.predict(X_train)
pred_test = lr.predict(X_test)
dtr_pred_train = dtr.predict(X_train)
dtr_pred_test = dtr.predict(X_test)

lr_mae_train = mean_absolute_error(y_train, pred_train)
lr_mae_test = mean_absolute_error(y_test, pred_test)
lr_mse_train = mean_squared_error(y_train, pred_train)
lr_mse_test = mean_squared_error(y_test, pred_test)
lr_rmse_train = np.sqrt(lr_mse_train)
lr_rmse_test = np.sqrt(lr_mse_test)
lr_mape_train = mean_absolute_percentage_error(y_train, pred_train)
lr_mape_test = mean_absolute_percentage_error(y_test, pred_test)

dtr_mae_train = mean_absolute_error(y_train, dtr_pred_train)
dtr_mae_test = mean_absolute_error(y_test, dtr_pred_test)
dtr_mse_train = mean_squared_error(y_train, dtr_pred_train)
dtr_mse_test = mean_squared_error(y_test, dtr_pred_test)
dtr_rmse_train = np.sqrt(dtr_mse_train)
dtr_rmse_test = np.sqrt(dtr_mse_test)
dtr_mape_train = mean_absolute_percentage_error(y_train, dtr_pred_train)
dtr_mape_test = mean_absolute_percentage_error(y_test, dtr_pred_test)

print("===== LINEAR REGRESSION RESULTS =====")
print("MAE Train:", lr_mae_train)
print("MAE Test:", lr_mae_test)
print("MSE Train:", lr_mse_train)
print("MSE Test:", lr_mse_test)
print("RMSE Train:", lr_rmse_train)
print("RMSE Test:", lr_rmse_test)
print("MAPE Train:", lr_mape_train)
print("MAPE Test:", lr_mape_test)

print("\n===== DECISION TREE RESULTS =====")
print("dtr_MAE Train:", dtr_mae_train)
print("dtr_MAE Test:", dtr_mae_test)
print("dtr_MSE Train:", dtr_mse_train)
print("dtr_MSE Test:", dtr_mse_test)
print("dtr_RMSE Train:", dtr_rmse_train)
print("dtr_RMSE Test:", dtr_rmse_test)
print("dtr_MAPE Train:", dtr_mape_train)
print("dtr_MAPE Test:", dtr_mape_test)

print("\nCoefficients:", lr.coef_)
