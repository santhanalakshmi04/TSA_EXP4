# DEVELOPED BY : SANTHANA LAKSHMI k
# REG NO. : 212222240091
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the provided power consumption dataset
file_path = '/content/powerconsumption.csv'
data = pd.read_csv(file_path)

# Extract the 'PowerConsumption_Zone1' column for modeling
data_values = data['PowerConsumption_Zone1'].dropna().values

# 1. ARMA(1,1) Model for PowerConsumption_Zone1

# Fit the ARMA(1,1) model
arma11_model = ARIMA(data_values, order=(1, 0, 1))
arma11_fit = arma11_model.fit()

# Plot the fitted ARMA(1,1) time series
plt.figure(figsize=(10, 6))
plt.plot(data_values, label='Original Data')
plt.plot(arma11_fit.fittedvalues, label='Fitted ARMA(1,1)', color='red')
plt.title('ARMA(1,1) Fitted Process - PowerConsumption_Zone1')
plt.xlabel('Time')
plt.ylabel('Power Consumption')
plt.legend()
plt.grid(True)
plt.show()

# Display ACF and PACF plots for the actual data
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data_values, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(data_values, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for PowerConsumption_Zone1')
plt.tight_layout()
plt.show()

# 2. ARMA(2,2) Model for PowerConsumption_Zone1

# Fit the ARMA(2,2) model
arma22_model = ARIMA(data_values, order=(2, 0, 2))
arma22_fit = arma22_model.fit()

# Plot the fitted ARMA(2,2) time series
plt.figure(figsize=(10, 6))
plt.plot(data_values, label='Original Data')
plt.plot(arma22_fit.fittedvalues, label='Fitted ARMA(2,2)', color='red')
plt.title('ARMA(2,2) Fitted Process - PowerConsumption_Zone1')
plt.xlabel('Time')
plt.ylabel('Power Consumption')
plt.legend()
plt.grid(True)
plt.show()

# Display ACF and PACF plots for the actual data
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data_values, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(data_values, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for PowerConsumption_Zone1')
plt.tight_layout()
plt.show()
```

# OUTPUT:
# SIMULATED ARMA(1,1) PROCESS:

![image](https://github.com/user-attachments/assets/71a78346-e56d-43e1-890d-dc97c729263b)

# Partial Autocorrelation and Autocorrelation

![image](https://github.com/user-attachments/assets/c0eb4d5d-18cb-48a3-98c2-b42e33c14722)

# SIMULATED ARMA(2,2) PROCESS:

![image](https://github.com/user-attachments/assets/01ab2691-d902-481f-9901-6c86749f8fda)

# Partial Autocorrelation and Autocorrelation

![image](https://github.com/user-attachments/assets/399becdf-3033-47c4-a439-545700f42967)

# RESULT:
Thus, a python program is successfully created to fit ARMA Model.
