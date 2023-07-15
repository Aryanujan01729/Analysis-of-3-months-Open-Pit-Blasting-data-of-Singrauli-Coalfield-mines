#!/usr/bin/env python
# coding: utf-8

# In[1076]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.impute import SimpleImputer
from scipy.interpolate import interp1d
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf


# In[1077]:


df=pd.read_csv('Blasting_dataset.csv')


# In[ ]:





# In[ ]:





# In[1078]:


df.head()


# In[1079]:


#df=Dataset.interpolate(method='cubic')


# In[1080]:


df


# In[1081]:


df.columns =df.columns.str.strip()


# In[1300]:


columns_to_fill = ['PM10 (µg/m3)', 'PM2.5 (µg/m3)', 'NO2 (µg/m3)', 'NOX (ppb)', 'CO (mg/m3)','NO (µg/m3)','NH3 (µg/m3)','SO2 (µg/m3)', 'Ozone (µg/m3)', 'Benzene (µg/m3)']

# Perform linear interpolation for the specified columns
df[columns_to_fill] = df[columns_to_fill].interpolate(method='linear',direaction='both')

# Print the updated DataFrame
print(df)


# In[1301]:


df.info()


# In[1084]:


df


# In[1085]:


#imputer = SimpleImputer(strategy='most_frequent')
#imputed_df = pd.DataFrame(imputer.fit_transform(Dataset))
#imputed_df.columns = Dataset.columns
#imputed_df.index = Dataset.index


# In[1086]:


df.describe()


# In[1087]:


df_PM2=imputed_df.loc[0:8640:96,'PM2.5 (µg/m3)']


# In[1414]:


start_time=datetime.datetime(2023,2,1,0,0)
end_time=datetime.datetime(2023,2,1,23,45)
time_delta=datetime.timedelta(minutes=15)
time=[]
current_time=start_time
while(end_time>=current_time):
    time.append(current_time)
    current_time=current_time+time_delta


# In[1089]:


'''start_time=datetime.datetime(2023,2,1,13,45)
end_time=datetime.datetime(2023,5,1,13,15)
time_delta=datetime.timedelta(days=1)
time=[]
current_time=start_time
while(end_time>=current_time):
    time.append(current_time)
    current_time=current_time+time_delta'''


# # Time Series of each pollutants in 1 Day

# ## PM10(µg/m3)

# In[1090]:


i=0
D=[]
for i in range(96):
    df_PM10S=df.loc[i:8640:96,'PM10 (µg/m3)']
    D.append(df_PM10S.mean())
   # print(i," ",df_PM10S.mean())


# In[1091]:


plt.figure(figsize=(12,6))
plt.plot(time,D)
plt.grid(True)
plt.xlabel('Time in hours')
plt.ylabel('Concentration')
plt.title('Graph of PM10 Throughout Day')
plt.xticks(rotation=45)
plt.show()


# ## PM2.5(µg/m3)

# In[1092]:


i=0
D1=[]
for i in range(96):
    df_PM2=df.loc[i:8640:96,'PM2.5 (µg/m3)']
    D1.append(df_PM2.mean())
    #print(i," ",df_PM2.mean())


# In[1093]:


plt.plot(time,D1)
plt.xticks(rotation=45)
plt.show()


# ## NOX (ppb)

# In[1169]:


df_NOX=df.loc[0:8640:96,'NOX (ppb)']
df_NOX


# In[1170]:


i=0
D2=[]
for i in range(96):
    df_NOX=df.loc[i:8640:96,'NOX (ppb)']
    D2.append(df_NOX.mean())
    #print(i," ",df_PM2.mean())


# In[1171]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.xlabel('Time in Hours')
plt.ylabel('Concentration')
plt.title('NOX (ppb)')
plt.xticks(rotation=45)
plt.plot(time,D2)
plt.xticks(rotation=45)
plt.show()


# In[1174]:


b=np.argmin(D2)


# In[1175]:


time[b]


# ## CO (mg/m3)

# In[1194]:


df_CO=df.loc[0:8640:98,'CO (mg/m3)']


# In[1195]:


i=0
D3=[]
for i in range(96):
    df_CO=df.loc[i:8640:96,'CO (mg/m3)']
    D3.append(df_CO.mean())
    #print(i," ",df_PM2.mean())


# In[1334]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.xlabel('Time in hours')
plt.ylabel('Concentration')
plt.title('CO (mg/m3)')
plt.xticks(rotation=45)
plt.plot(time,D3)
plt.xticks(rotation=45)
plt.show()


# In[1207]:


d=np.argmin(D3)
time[d]


# In[1204]:


np.max(D3[:30])


# ## 'NO2 (µg/m3)'

# In[1283]:


df_NO2E=df.loc[0:8640:96,'NO2 (µg/m3)']


# In[1284]:


i=0
D4=[]
for i in range(96):
    df_NO2=df.loc[i:8640:96,'NO2 (µg/m3)']
    D4.append(df_NO2.mean())
    #print(i," ",df_PM2.mean())   


# In[1285]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time,D4)
plt.xlabel('Time in hours')
plt.ylabel('Concentration')
plt.title('Average Day Plot')
plt.xticks(rotation=45)
plt.show()


# In[1292]:


time[np.argmin(D4)]


# In[1293]:


np.min(D4)


# ## NH3 (µg/m3)

# In[1103]:


df_NH3=df.loc[0:8640:96,'NH3 (µg/m3)']


# In[1104]:


i=0
D5=[]
for i in range(96):
    df_NH3=df.loc[i:8640:96,'NH3 (µg/m3)']
    D5.append(df_NH3.mean())
    #print(i," ",df_NH3.mean())


# In[1105]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time,D5)
plt.xlabel('Time in hours')
plt.ylabel('Concentration')
plt.title('Graph of NH3')
plt.xticks(rotation=45)
plt.show()


# In[1220]:


a=np.argmin(D5)
time[a]


# In[1221]:


np.min(D5)


# ## NO (µg/m3)

# In[1302]:


df_NO=df.loc[0:8640:96,'NO (µg/m3)']


# In[1303]:


df


# In[1304]:


i=0
D6=[]
for i in range(96):
    df_NO=df.loc[i:8640:96,'NO (µg/m3)']
    D6.append(df_NO.mean())
    #print(i," ",df_PM2.mean())


# In[1312]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time,D6)
plt.xticks(rotation=45)
plt.xlabel('Time in hours')
plt.ylabel('Concentration')
plt.show()


# In[1432]:





# ## SO2 (µg/m3)

# In[1415]:


df_SO2=df.loc[0:8640:96,'SO2 (µg/m3)']


# In[1416]:


i=0
D7=[]
for i in range(96):
    df_SO2=df.loc[i:8640:96,'SO2 (µg/m3)']
    D7.append(df_SO2.mean())
    #print(i," ",df_PM2.mean())


# In[1417]:


plt.figure(figsize=(12, 6))
plt.grid(True)
plt.plot(time,D7)
plt.xlabel('Time in hours')
plt.ylabel('Concentration')
plt.title('Graphg of SO2')
plt.xticks(rotation=45)
plt.show()


# In[1420]:





# In[1421]:





# ## Benzene (µg/m3)

# In[1246]:


df_Benzene=imputed_df.loc[0:8640:96,'Benzene (µg/m3)']


# In[1309]:


i=0
D8=[]
for i in range(96):
    df_Benzene=df.loc[i:8640:96,'Benzene (µg/m3)']
    D8.append(df_Benzene.mean())
    #print(i," ",df_PM2.mean())


# In[1310]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time,D8)
plt.xlabel('time in hours')
plt.ylabel('concentration')
plt.title('graph of Benzene')
plt.xticks(rotation=45)
plt.show()


# ## Ozone (µg/m3)

# In[1116]:


df_Ozone=df.loc[0:8640:96,'Ozone (µg/m3)']


# In[1117]:


i=0
D9=[]
for i in range(96):
    df_Ozone=df.loc[i:8640:96,'Ozone (µg/m3)']
    D9.append(df_Ozone.mean())
    #print(i," ",df_PM2.mean())


# In[1244]:



plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time,D9)
plt.xlabel('Time in hours')
plt.ylabel('Concentration')
plt.title('graph of Ozone')
plt.xticks(rotation=45)
plt.show()


# In[1242]:


time[np.argmin(D9)]


# In[1243]:


np.min(D9)


# # Time Series of per day average throughout 3 Months 

# # PM2.5(µg/m3)

# In[1119]:


start_time1=datetime.datetime(2023,2,1,0,0)
end_time1=datetime.datetime(2023,5,1,0,0)
time_delta1=datetime.timedelta(days=1)
time1=[]
current_time1=start_time1
while(end_time1>=current_time1):
    time1.append(current_time1)
    current_time1=current_time1+time_delta1


# In[1120]:


i=0
D10=[]
for i in range(0,8640,96):
    df_PM2=df.loc[i:(96+i),'PM2.5 (µg/m3)']
    D10.append(df_PM2.mean())


# In[1121]:


plt.figure(figsize=(12, 6))
plt.grid(True)
plt.plot(time1,D10)

plt.xticks(rotation=45)
plt.show()


# In[1332]:


plt.figure(figsize=(12,6))

plot_acf(D10, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average PM2.5")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# In[ ]:





# In[1122]:


mean = np.mean(D10)
max_lag=20
autocovariance = []
lags = range(1, max_lag + 1)  # Specify the maximum lag value

for lag in lags:
    shifted_data = D10[lag:] - mean
    lagged_data = D10[:-lag] - mean
    cov = np.mean(shifted_data * lagged_data)
    autocovariance.append(cov)


# In[1123]:


autocovariance


# In[1124]:



plt.figure(figsize=(12,6))
plt.grid(True)
plt.stem(range(len(autocovariance)), autocovariance)  # Plot the ACF values
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()


# In[1125]:


plt.figure(figsize=(12,6))
plot_pacf(D10, lags=20)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# # PM10 (µg/m3)

# In[1126]:


i=0
D11=[]
for i in range(0,8640,96):
    df_PM10=df.loc[i:(96+i),'PM10 (µg/m3)']
    D11.append(df_PM10.mean())


# In[1127]:


plt.figure(figsize=(12,8))
plt.plot(time1,D11)
plt.xlabel('Time in Days')
plt.ylabel('Concentration')
plt.title('Graph of PM10')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# In[1330]:


plt.figure(figsize=(12,6))

plot_acf(D11, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average PM10")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# In[ ]:





# In[1128]:


plt.figure(figsize=(12,6))
plot_pacf(D11, lags=20)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1129]:


mean = np.mean(D11)
max_lag=19
autocovariance = []
lags = range(1, max_lag + 1)  # Specify the maximum lag value

for lag in lags:
    shifted_data = D11[lag:] - mean
    lagged_data = D11[:-lag] - mean
    cov = np.mean(shifted_data * lagged_data)
    autocovariance.append(cov)


# In[1130]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.stem(range(len(autocovariance)), autocovariance)  # Plot the ACF values
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## NOX (ppb)

# In[1131]:


i=0
D12=[]
for i in range(0,8640,96):
    df_NOX=df.loc[i:(96+i),'NOX (ppb)']
    D12.append(df_NOX.mean())


# In[1132]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time1,D12)
plt.xlabel('Time in Months')
plt.ylabel('Concentration')
plt.title('Nox(ppb)')
plt.xticks(rotation=45)
plt.show()


# In[1333]:


plt.figure(figsize=(12,6))

plot_acf(D12, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average NOx")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# In[1147]:





# In[1133]:


plt.figure(figsize=(12,6))
plot_pacf(D12, lags=20)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1134]:


mean = np.mean(D12)
max_lag=20
autocovariance = []
lags = range(1, max_lag + 1)  # Specify the maximum lag value

for lag in lags:
    shifted_data = D12[lag:] - mean
    lagged_data = D12[:-lag] - mean
    cov = np.mean(shifted_data * lagged_data)
    autocovariance.append(cov)


# In[1135]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.stem(range(len(autocovariance)), autocovariance)  # Plot the ACF values
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()


# ## CO (mg/m3)

# In[1189]:


i=0
D13=[]
for i in range(0,8640,96):
    df_CO=df.loc[i:(96+i),'CO (mg/m3)']
    D13.append(df_CO.mean())


# In[1190]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time1,D13)
plt.xticks(rotation=45)
plt.show()


# In[1327]:


plt.figure(figsize=(12,6))

plot_acf(D13, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average CO")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# In[1192]:


np.min(D13)


# In[998]:


plt.figure(figsize=(12,6))
plot_pacf(D13, lags=30)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1000]:


mean = np.mean(D13)
max_lag=20
autocovariance = []
lags = range(1, max_lag + 1)  # Specify the maximum lag value

for lag in lags:
    shifted_data = D13[lag:] - mean
    lagged_data = D13[:-lag] - mean
    cov = np.mean(shifted_data * lagged_data)
    autocovariance.append(cov)


# In[1001]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.stem(range(len(autocovariance)), autocovariance)  # Plot the ACF values
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()


# ## NH3 (µg/m3)

# In[939]:


i=0
D14=[]
for i in range(0,8640,96):
    df_NH3=df.loc[i:(96+i),'NH3 (µg/m3)']
    D14.append(df_NH3.mean())


# In[1208]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time1,D14)
plt.xlabel('Time in days')
plt.ylabel('Concentration')
plt.title('Graph of NH3')
plt.xticks(rotation=45)
plt.show()


# In[1213]:


e=np.argmax(D14)
time1[e]


# In[1212]:


np.min(D14)


# In[1054]:


plt.figure(figsize=(12,6))

plot_acf(D14, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average NH3")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# In[941]:


plot_pacf(D14, lags=24)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1009]:


mean = np.mean(D14)
max_lag=16
autocovariance = []
lags = range(1, max_lag + 1)  # Specify the maximum lag value

for lag in lags:
    shifted_data = D14[lag:] - mean
    lagged_data = D14[:-lag] - mean
    cov = np.mean(shifted_data * lagged_data)
    autocovariance.append(cov)


# In[1010]:


plt.stem(range(len(autocovariance)), autocovariance)# Plot the ACF values
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()


#  ## NO (µg/m3)

# In[1063]:


i=0
D15=[]
for i in range(0,8640,96):
    df_NO=imputed_df.loc[i:(96+i),'NO (µg/m3)']
    D15.append(df_NO.mean())


# In[1066]:



plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time1,D15)
plt.xlabel('Time in months')
plt.ylabel('Concentration')
plt.xticks(rotation=45)
plt.show()


# In[1434]:





# In[ ]:





# In[1267]:


plot_pacf(D15, lags=24)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1268]:


plt.figure(figsize=(12,6))

plot_acf(D15, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average Ozone")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# ## SO2 (µg/m3)

# In[1426]:


i=0
D16=[]
for i in range(0,8640,96):
    df_SO2=imputed_df.loc[i:(96+i),'SO2 (µg/m3)']
    D16.append(df_SO2.mean())


# In[1436]:





# In[1427]:


plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time1,D16)
plt.xlabel('Time in Days')
plt.ylabel('Concentration')
plt.title('Graph of SO2 ')
plt.xticks(rotation=45)
plt.show()


# In[1430]:





# In[1338]:


plt.figure(figsize=(12,6))

plot_acf(D15, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average NO")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# In[ ]:





# In[1319]:


plt.figure(figsize=(12,6))
plot_pacf(D16, lags=20)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1320]:


plt.figure(figsize=(12,6))

plot_acf(D16, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average Ozone")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# ## Ozone (µg/m3)

# In[1225]:


i=0
D17=[]
for i in range(0,8640,96):
    df_Ozone=df.loc[i:(96+i),'Ozone (µg/m3)']
    D17.append(np.mean(df_Ozone))


# In[1228]:


plt.figure(figsize=(12,6))
plt.grid(True)

plt.plot(time1,D17)
plt.xlabel('Time in days')
plt.ylabel('concentration')
plt.title('Daily Average of Ozone')
plt.xticks(rotation=45)
plt.show()


# In[1245]:


plt.figure(figsize=(12,6))
plot_pacf(D17, lags=30)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1235]:


np.min(D17)


# In[1024]:


plot_pacf(D17, lags=19)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[ ]:





# In[1027]:


mean = np.mean(D17)
max_lag=16
autocovariance = []
lags = range(1, max_lag + 1)  # Specify the maximum lag value

for lag in lags:
    shifted_data = D17[lag:] - mean
    lagged_data = D17[:-lag] - mean
    cov = np.mean(shifted_data * lagged_data)
    autocovariance.append(cov)


# In[1045]:


#df_daily_avg = df['NH3 (µg/m3)'].resample('D').mean()

# Plot ACF for daily average of NH3
#plot_acf(df_daily_avg, lags=30)
#plt.show()
#plt.figure(figsize=(12,6))
#plt.stem(range(len(autocovariance)), autocovariance)  # Plot the ACF values#

#plt.xlabel('Lag')
#plt.ylabel('Autocorrelation')
#plt.title('Autocorrelation Function (ACF) Plot')
#lt.show()

plt.figure(figsize=(12,6))

plot_acf(D17, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average Ozone")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# ## Benzene (µg/m3)

# In[1260]:


i=0
D18=[]
for i in range(0,8640,96):
    df_Benzene=df.loc[i:(96+i),'Benzene (µg/m3)']
    D18.append(df_Benzene.mean())


# In[1261]:



plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(time1,D18)
plt.xlabel('Time in months')
plt.ylabel('Cincentration')
plt.xticks(rotation=45)
plt.show()


# In[1262]:


plot_pacf(D18, lags=25)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1263]:


plt.figure(figsize=(12,6))

plot_acf(D18, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average Benzene")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# ##  NO2 (µg/m3)

# In[1276]:


i=0
D19=[]
for i in range(0,8640,96):
    df_NO2=df.loc[i:(96+i),'NO2 (µg/m3)']
    D19.append(df_NO2.mean())


# In[1277]:


plt.figure(figsize=(12,6))
plt.plot(time1,D19)
plt.xticks(rotation=45)
plt.xlabel('Time in months')
plt.ylabel('Concentration')
plt.title('Monthly NO2 Graph')
plt.grid(True)
plt.show()


# In[1280]:


time1[np.argmin(D19)]


# In[1281]:


np.min(D19)


# In[1052]:


plot_pacf(D19, lags=19)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[1051]:


plt.figure(figsize=(12,6))

plot_acf(D19, lags=30)
plt.grid(True)
plt.title("ACF - Daily Average NO2")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")

# Display the plot
plt.show()


# In[1325]:


import matplotlib.pyplot as plt

# Data series
columns = ['PM2.5 (µg/m3)', 'PM2.5 (µg/m3)', 'NO (µg/m3)', 'NO2 (µg/m3)', 'NOX (ppb)',
           'CO (mg/m3)', 'SO2 (µg/m3)', 'NH3 (µg/m3)', 'Ozone (µg/m3)', 'Benzene (µg/m3)']

data_series = []

# Iterate over columns and extract data
for column in columns:
    i = 0
    data = []
    for i in range(0, 8640, 96):
        df_column = df.loc[i:(96 + i), column]
        data.append(df_column.mean())
    data_series.append(data)

# Plotting
plt.figure(figsize=(22, 15))  # Increase figure size

# Plot each data series
for i, data in enumerate(data_series):
    plt.plot(time1, data, label=columns[i])

plt.xticks(rotation=45)
plt.legend()  # Add legend
plt.grid(True)  # Add grid

plt.show()


# In[1335]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# Generate sample data
#D18 = np.random.normal(loc=0, scale=1, size=100)  # Replace with your actual data array
qqplot(D13, line='s')
# Create QQ plot
#plt.figure(figsize=(12,6))
#stats.probplot(D18, dist="norm", plot=plt)
#plt.title("QQ Plot - CO")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Values")
plt.show()


# In[1336]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate sample data
D13= np.random.normal(loc=0, scale=1, size=100)  # Replace with your actual data array

# Create QQ plot
plt.figure(figsize=(12,6))
stats.probplot(D13, dist="norm", plot=plt)
plt.title("QQ Plot -CO ")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Values")
plt.show()


# In[1351]:


plt.plot(range(len(D10)), D10, label='Original Data')
plt.plot(range(len(D10), len(D10)+11), forecast_values, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()


# In[1356]:


df.describe()


# In[1365]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Assuming D10 is a numpy array or a pandas Series

# Fit ARIMA model
model = ARIMA(D10, order=(1, 0, 1))  # Replace p, d, q with appropriate values
model_fit = model.fit()

# Generate forecasts
forecast_values = model_fit.predict(start=len(D10), end=len(D10)+8)  # Replace n with the number of future time steps to forecast

# Plot original data and forecasts
plt.plot(range(len(D10)), D10, label='Original Data')
plt.plot(range(len(D10), len(D10)+9), forecast_values, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()


# In[1407]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Assuming D10 is a numpy array or a pandas Series
# Fit ARIMA model
model = ARIMA(D14, order=(2, 0, 3))  # Replace p, d, q with appropriate values
model_fit = model.fit()

# Generate forecasts
forecast_values = model_fit.predict(start=len(D14), end=len(D14)+8)  # Replace n with the number of future time steps to forecast

# Plot original data in blue
plt.plot(range(len(D14)), D14, color='blue', label='Original Data')

# Plot forecasts in red
plt.plot(range(len(D14), len(D14)+9), forecast_values, color='red', label='Forecast')

# Connect the two plots with a black line
plt.plot([len(D14)-1, len(D14)], [D14[-1], forecast_values[0]], 'k-',color='red')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('concentration')
plt.title('ARMA Forecast')
plt.legend()
plt.show()


# In[1437]:


df.describe()


# In[ ]:




