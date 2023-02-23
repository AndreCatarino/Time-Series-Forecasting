# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:37:03 2022

"""

import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsaplots

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm  

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_absolute_error

#ELETRICIDADE
df2=pd.read_excel('C:/Users/saral/Desktop/Trabalhos/Trabalho MP/Eletricidade/Consumo eletricidade mensal 2010-2021.xlsx', header=0, index_col=0, parse_dates=True)
df2.index.freq = 'MS' #Dizer que os dados são mensais

print(df2.head())
print(df2.info())

df2.plot()

                                    #DECOMPOSIÇÃO
                                    
seasonal_decompose(df2, model='aditive', freq=12).plot()
seasonal_decompose(df2, model='multiplicative', freq=12).plot() #parece melhor nos resíduos

#Tendência
df2_trend = seasonal_decompose(df2, model='multiplicative', freq=12).trend
df2_trend.plot(style='k').plot()

#Integridade dos dados (são contínuos e não há falta nem repetição de valores)
cross_tab = pd.crosstab(index=df2.index.year, columns=df2.index.month)

#Heatmap
fig = plt.figure()
sns.heatmap(pd.pivot_table(data=df2, index=df2.index.year, columns=df2.index.month),
           square=True,
           cmap='Blues',
           xticklabels=["janeiro", "fevereiro", "março", "abril",
                        "maio", "junho","julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]);

#Gráfico de subséries sazonais
df2["Date"] = pd.to_datetime(df2.index)
df2["mes"] = df2["Date"].dt.month

fig, ax = plt.subplots()

tsaplots.seasonal_plot(df2['Consumo (GWh)'].groupby(df2["mes"]),
                       list(range(0,12)),ax=ax)

fig.set_size_inches(18.5, 10.5)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)


                                #ALISAMENTO EXPONENCIAL:
    
# Train and Test data Splitting
train_data = df2.iloc[:115]
test_data = df2.iloc[115:]



# Holt-Winters Multiplicativo
hwm_model = ExponentialSmoothing(train_data['Consumo (GWh)'],
                                 trend='add', seasonal='multiplicative', seasonal_periods=12).fit()
hwm_test_pred = hwm_model.forecast(29).rename('Previsão do Consumo com Holt-Winters Multiplicativo')
print(hwm_test_pred)


train_data['Consumo (GWh)'].plot(legend=True,label='Treino')
test_data['Consumo (GWh)'].plot(legend=True,label='Teste')
hwm_test_pred.plot(legend=True,label='Previsão do Consumo com Holt-Winters Multiplicativo')

# Medidas dos erros (Holt-Winters Multiplicativo)
rmse_hwm = round(np.sqrt(mean_squared_error(test_data['Consumo (GWh)'], hwm_test_pred)),2)
print("RMSE_HWM is ",rmse_hwm)
mae_hwm = round(mean_absolute_error(test_data['Consumo (GWh)'],hwm_test_pred),2)
print("MAE_HWM is ",mae_hwm)
mape_hwm = round(100*mean_absolute_percentage_error(test_data['Consumo (GWh)'],hwm_test_pred),2)
print("MAPE_HWM is ",mape_hwm,'%')



# Multiplicativo Amortecido
hmd_model = ExponentialSmoothing(train_data['Consumo (GWh)'],
                                 trend='add', damped_trend=True, seasonal='multiplicative', seasonal_periods=12).fit()
hmd_test_pred = hmd_model.forecast(29).rename('Previsão do Consumo com Multiplicativo Amortecido')
print(hmd_test_pred)


train_data['Consumo (GWh)'].plot(legend=True,label='Treino')
test_data['Consumo (GWh)'].plot(legend=True,label='Teste')
hmd_test_pred.plot(legend=True,label='Previsão do Consumo com Multiplicativo Amortecido')

# Medidas dos erros (Holt-winters Multiplicativo Amortecido)
rmse_hmd = round(np.sqrt(mean_squared_error(test_data['Consumo (GWh)'], hmd_test_pred)),2)
print("RMSE_HMD is ",rmse_hmd)
mae_hmd = round(mean_absolute_error(test_data['Consumo (GWh)'],hmd_test_pred),2)
print("MAE_HMD is ",mae_hmd)
mape_hmd = round(100*mean_absolute_percentage_error(test_data['Consumo (GWh)'],hmd_test_pred),2)
print("MAPE_HMD is ",mape_hmd,'%')


                                 #MODELOS SARMA/SARIMA:
                                     
#Cálculo de FAC e de FACP para os dados originais   
fig0 = plt.figure(figsize=(12,8))
ax1 = fig0.add_subplot(211)
fig0 = sm.graphics.tsa.plot_acf(df2['Consumo (GWh)'], lags=36, ax=ax1)
ax2 = fig0.add_subplot(212)
fig0 = sm.graphics.tsa.plot_pacf(df2['Consumo (GWh)'], lags=36, ax=ax2)    

#Teste de raízes unitárias para o período de treino
def test_stationarity(timeseries): 
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)   
    
test_stationarity(train_data['Consumo (GWh)']) 

#Transformações para estacionarizar:
    
#Logarítmo
train_data_log= train_data['Consumo (GWh)'].apply(lambda x: np.log(x))  
test_stationarity(train_data_log)
train_data_log.plot(figsize=(12,8), title= 'Logarítmo', fontsize=14)

#Primeira diferença
train_data['first_difference'] = train_data['Consumo (GWh)'] - train_data['Consumo (GWh)'].shift(1)  
test_stationarity(train_data.first_difference.dropna(inplace=False))
train_data['first_difference'].plot(figsize=(12,8), 
                            title= 'Primeira diferença', fontsize=14)

#Segunda diferença
train_data['second_difference'] =train_data['first_difference']-train_data['first_difference'].shift(1)
test_stationarity(train_data.second_difference.dropna(inplace=False))
train_data['second_difference'].plot(figsize=(12,8), 
                            title= 'Segunda diferença', fontsize=14)

#Diferença sazonal -> estacionária a 10%
train_data['seasonal_difference'] = train_data['Consumo (GWh)'] - train_data['Consumo (GWh)'].shift(12)  
test_stationarity(train_data.seasonal_difference.dropna(inplace=False))
train_data['seasonal_difference'].plot(figsize=(12,8), 
                               title= 'Seasonal difference', fontsize=14)

#Diferença simples e sazonal -> estacionária a 1%
train_data['seasonal_first_difference'] = train_data.first_difference - train_data.first_difference.shift(12)  
test_stationarity(train_data.seasonal_first_difference.dropna(inplace=False))
train_data['seasonal_first_difference'].plot(figsize=(12,8), 
                                     title= 'Ordinary and seasonal differences', fontsize=14)

#FAC e FACP da série estacionarizada
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_data.seasonal_first_difference.iloc[13:],lags=32, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_data.seasonal_first_difference.iloc[13:], lags=32, ax=ax2)

#SARIMA

#Melhores parâmetros do SARIMA para minmizar AIC:
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q 
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
params=[]
seasonal=[]
aic=[]

import statsmodels.api as sm

for param in pdq:
    for param_seasonal in seasonal_pdq:
        mod = sm.tsa.statespace.SARIMAX(train_data['Consumo (GWh)'],order=param,seasonal_order=param_seasonal, enforce_stationarity=True, enforce_invertibility=True)

        results = mod.fit()
        params.append(param)
        seasonal.append(param_seasonal)
        aic.append(results.aic)
        
parameter_options=pd.DataFrame({'params':params,'seasonal_params':seasonal,'AIC':aic})

print(parameter_options.sort_values(by='AIC'))    

#Modelo 1
model1 = sm.tsa.statespace.SARIMAX(train_data['Consumo (GWh)'], trend='n', 
                                order=(1,1,1), 
                                seasonal_order=(1,1,1,12))
results1 = model1.fit()
print(results1.summary())

resid1 = pd.DataFrame(results1.resid)
resid1.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pd.DataFrame(results1.resid), lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pd.DataFrame(results1.resid),lags=14, ax=ax2)

df2['forecast com SARIMA (1,1,1)(1,1,1,12)'] = results1.predict(start = 115,
                                   end= 144, dynamic= True)  
df2[['Consumo (GWh)', 'forecast com SARIMA (1,1,1)(1,1,1,12)']].iloc[13:].plot(figsize=(12, 8))

# Medidas dos erros
rmse_model1 = round(np.sqrt(mean_squared_error(test_data['Consumo (GWh)'], df2['forecast com SARIMA (1,1,1)(1,1,1,12)'].iloc[115:])),2)
print("RMSE_MODEL1 is ",rmse_model1)

mae_model1 = round(mean_absolute_error(test_data['Consumo (GWh)'],df2['forecast com SARIMA (1,1,1)(1,1,1,12)'].iloc[115:]),2)
print("MAE_MODEL1 is ",mae_model1)

mape_model1 = round(100*mean_absolute_percentage_error(test_data['Consumo (GWh)'],df2['forecast com SARIMA (1,1,1)(1,1,1,12)'].iloc[115:]),2)
print("MAPE_MODEL1 is ",mape_model1,'%')



#Modelo 2 (escolha através do correlograma)
model2 = sm.tsa.statespace.SARIMAX(train_data['Consumo (GWh)'], trend='n', 
                                order=(1,1,3), 
                                seasonal_order=(1,1,1,12))
results2 = model2.fit()
print(results2.summary())

resid2 = pd.DataFrame(results2.resid)
resid2.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pd.DataFrame(results2.resid), lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pd.DataFrame(results2.resid),lags=14, ax=ax2)


df2['forecast com SARIMA (1,1,3)(1,1,1,12)'] = results2.predict(start = 115,
                                   end= 144, dynamic= True)  
df2[['Consumo (GWh)', 'forecast com SARIMA (1,1,3)(1,1,1,12)']].iloc[13:].plot(figsize=(12, 8))

# Medidas dos erros
rmse_model2 = round(np.sqrt(mean_squared_error(test_data['Consumo (GWh)'], df2['forecast com SARIMA (1,1,3)(1,1,1,12)'].iloc[115:])),2)
print("RMSE_MODEL2 is ",rmse_model2)

mae_model2 = round(mean_absolute_error(test_data['Consumo (GWh)'],df2['forecast com SARIMA (1,1,3)(1,1,1,12)'].iloc[115:]),2)
print("MAE_MODEL2 is ",mae_model2)

mape_model2 = round(100*mean_absolute_percentage_error(test_data['Consumo (GWh)'],df2['forecast com SARIMA (1,1,3)(1,1,1,12)'].iloc[115:]),2)
print("MAPE_MODEL2 is ",mape_model2,'%')



#Modelo 3
model3 = sm.tsa.statespace.SARIMAX(train_data['Consumo (GWh)'], trend='n', 
                                order=(1,1,1), 
                                seasonal_order=(0,1,1,12))
results3 = model3.fit()
print(results3.summary())

resid3 = pd.DataFrame(results3.resid)
resid3.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pd.DataFrame(results3.resid), lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pd.DataFrame(results3.resid),lags=14, ax=ax2)

df2['forecast com SARIMA (1,1,1)(0,1,1,12)'] = results3.predict(start = 115,
                                   end= 144, dynamic= True)  
df2[['Consumo (GWh)', 'forecast com SARIMA (1,1,1)(0,1,1,12)']].iloc[13:].plot(figsize=(12, 8))

# Medidas dos erros
rmse_model3 = round(np.sqrt(mean_squared_error(test_data['Consumo (GWh)'], df2['forecast com SARIMA (1,1,1)(0,1,1,12)'].iloc[115:])),2)
print("RMSE_MODEL3 is ",rmse_model3)

mae_model3 = round(mean_absolute_error(test_data['Consumo (GWh)'],df2['forecast com SARIMA (1,1,1)(0,1,1,12)'].iloc[115:]),2)
print("MAE_MODEL3 is ",mae_model3)

mape_model3 = round(100*mean_absolute_percentage_error(test_data['Consumo (GWh)'],df2['forecast com SARIMA (1,1,1)(0,1,1,12)'].iloc[115:]),2)
print("MAPE_MODEL3 is ",mape_model3,'%')


                                          #PREVISÃO:

#Previsão futura (12 meses) com função predict
future_dates = pd.date_range(start="2022-01-1", end="2022-12-01", freq = "MS")

pred = results3.predict(start=len(df2)+1, end=len(df2)+12)

pred.index = future_dates

for i in range(len(pred)):
    new_row = {'Previsao_futura_predict': pred.iloc[i]}
    df2 = df2.append(new_row, ignore_index=True)

df2[['Consumo (GWh)','Previsao_futura_predict']].plot(figsize=(12, 8))

#Previsão futura (12 meses) recorrendo à biblioteca scalecast (para obter intervalos de confiança)- usar pip intall SCALECAST
from scalecast.Forecaster import Forecaster

dates = pd.date_range(start=pd.datetime(2010, 1, 1), periods=144, freq='MS')

f = Forecaster(y=df2['Consumo (GWh)'],current_dates= dates)

f.generate_future_dates(12) 
f.set_test_length(.2) #20% período de teste
f.set_estimator('arima')

#Definir os parametros do modelo 3 (aquele que possui menores erros de previsão)
f.manual_forecast(order=(1,1,1),seasonal_order=(0,1,1,12),call_me='SARIMA')

f.plot_test_set(ci=True,models='SARIMA') #Previsão para o período de teste
plt.title('SARIMA (1,1,1)(0,1,1,12)',size=14)
plt.show()

f.plot(ci=True,models='SARIMA') #Previsão para o futuro
plt.title('SARIMA (1,1,1) (0,1,1,12)',size=14)
plt.show()

df_export1 = f.export_forecasts_with_cis('SARIMA')
print(df_export1)


                            #PREVISÃO (sem influência de covid-19):

df2["ts_withoutCovid"] = df2['Consumo (GWh)']

df2["ts_withoutCovid"].iloc[123:] = df2['forecast com SARIMA (1,1,1)(0,1,1,12)'].iloc[123:]

dates = pd.date_range(start=pd.datetime(2010, 1, 1), periods=144, freq='MS')

f = Forecaster(y=df2['ts_withoutCovid'],current_dates= dates)

f.generate_future_dates(12) 
f.set_test_length(.2) #20% período de teste
f.set_estimator('arima') 

f.manual_forecast(order=(1,1,1),seasonal_order=(0,1,1,12),call_me='SARIMA')

f.plot(ci=True,models='SARIMA') #Previsão para o futuro
plt.title('SARIMA (1,1,1) (0,1,1,12) Forecast Performance without Covid',size=14)
plt.show()

df_export1 = f.export_forecasts_with_cis('SARIMA')
print(df_export1)



#Combinação de previsões futuras (12 meses) com função predict e com os 3 melhores modelos
#(model3, hmd_model  e hwm_model) ponderados igualmente

#Definir df2 com a configuração inicial anterior 
df2.drop(["ts_withoutCovid"],axis=1, inplace=True)
df2.drop(["Previsao_futura_predict"],axis=1, inplace=True)
df2.drop(df2.tail(len(pred)).index,inplace=True)

future_dates = pd.date_range(start="2022-01-1", end="2022-12-01", freq = "MS")

#model3
pred_1=results3.predict(start=len(df2)+1, end=len(df2)+12)
pred_1.index = future_dates

#hmd_model
pred_2 = hmd_model.predict(start=len(df2)+1, end=len(df2)+12)
pred_2.index = future_dates

#hwm_model
pred_3= hwm_model.predict(start=len(df2)+1, end=len(df2)+12)
pred_3.index = future_dates

comb_modelo_1 = (pred_1 + pred_2 + pred_3) /3
print(comb_modelo_1)

for i in range(len(comb_modelo_1)):
    new_row = {'Previsao_futura_combModel1': comb_modelo_1.iloc[i]}
    df2 = df2.append(new_row, ignore_index=True)

df2[['Consumo (GWh)','Previsao_futura_combModel1']].plot(figsize=(12, 8))

#Combinação de previsões futuras (12 meses) com função predict e com os 3 melhores modelos
#(model3, hmd_model  e hwm_model) por ordem de desempenho (EAM)            

#Definir df1 com a configuração inicial anterior 
df2.drop(["Previsao_futura_combModel1"],axis=1, inplace=True)
df2.drop(df2.tail(len(comb_modelo_1)).index,inplace=True)

#M corresponde à soma do EAM dos 3 melhores modelos selecionados
M = 401.95
EAM = [124.43,135.24,142.28] #EAM dos 3 modelos referidos

future_dates = pd.date_range(start="2022-01-1", end="2022-12-01", freq = "MS")

#model3
pred_1=results3.predict(start=len(df2)+1, end=len(df2)+12)
pred_1.index = future_dates
pred_1 = pred_1.apply(lambda x: x*(M-EAM[0]))

#hmd_model
pred_2 = hmd_model.predict(start=len(df2)+1, end=len(df2)+12)
pred_2.index = future_dates
pred_2 = pred_2.apply(lambda x: x*(M-EAM[1]))

#hwm_model
pred_3= hwm_model.predict(start=len(df2)+1, end=len(df2)+12)
pred_3.index = future_dates
pred_3 = pred_3.apply(lambda x: x*(M-EAM[2]))

comb_modelo_2 = (pred_1 + pred_2 + pred_3) /((3-1)*M)
print(comb_modelo_2)

for i in range(len(comb_modelo_2)):
    new_row = {'Previsao_futura_combModel2': comb_modelo_2.iloc[i]}
    df2 = df2.append(new_row, ignore_index=True)

df2[['Consumo (GWh)','Previsao_futura_combModel2']].plot(figsize=(12, 8))



#Obtenção das previsões para o período de teste, utilizando combinação de previsões poderadas igualmente
df2.drop(["Previsao_futura_combModel2"],axis=1, inplace=True)
df2.drop(df2.tail(len(comb_modelo_2)).index,inplace=True)

#model3
pred_1 = results3.predict(start = 115, end=143)

#hmd_model
pred_2 = hmd_model.predict(start=115, end=143)

#hwm_model
pred_3 = hwm_model.predict(start=115, end=143)

comb_modelo_1_amostra = (pred_1 + pred_2 + pred_3)/3
print(comb_modelo_1_amostra)

df2['comb_modelo_1_amostra'] = np.nan
df2['comb_modelo_1_amostra'].iloc[115:144] = comb_modelo_1_amostra

df2[['Consumo (GWh)','comb_modelo_1_amostra']].plot(figsize=(12, 8))

#Medidas dos erros
rmse_model = round(np.sqrt(mean_squared_error(test_data['Consumo (GWh)'], df2['comb_modelo_1_amostra'].iloc[115:])),2)
print("RMSE_MODEL is ",rmse_model)

mae_model = round(mean_absolute_error(test_data['Consumo (GWh)'],df2['comb_modelo_1_amostra'].iloc[115:]),2)
print("MAE_MODEL is ",mae_model)

mape_model = round(100*mean_absolute_percentage_error(test_data['Consumo (GWh)'],df2['comb_modelo_1_amostra'].iloc[115:]),2)
print("MAPE_MODEL is ",mape_model,'%')



#Obtenção das previsões para o período de teste, utilizando combinação de previsões poderadas por ordem de desempenho (EAM)
df2.drop(["comb_modelo_1_amostra"],axis=1, inplace=True)

#model3
pred_1 = results3.predict(start = 115, end=143)
pred_1 = pred_1.apply(lambda x: x*(M-EAM[0]))

#hmd_model
pred_2 = hmd_model.predict(start=115, end=143)
pred_2 = pred_2.apply(lambda x: x*(M-EAM[0]))

#hwm_model
pred_3 = hwm_model.predict(start=115, end=143)
pred_3 = pred_3.apply(lambda x: x*(M-EAM[0]))

comb_modelo_2_amostra = (pred_1 + pred_2 + pred_3)/((3-1)*M)
print(comb_modelo_2_amostra)

df2['comb_modelo_2_amostra'] = np.nan
df2['comb_modelo_2_amostra'].iloc[115:144] = comb_modelo_2_amostra

df2[['Consumo (GWh)','comb_modelo_2_amostra']].plot(figsize=(12, 8))

#Medidas dos erros
rmse_model = round(np.sqrt(mean_squared_error(test_data['Consumo (GWh)'], df2['comb_modelo_2_amostra'].iloc[115:])),2)
print("RMSE_MODEL is ",rmse_model)

mae_model = round(mean_absolute_error(test_data['Consumo (GWh)'],df2['comb_modelo_2_amostra'].iloc[115:]),2)
print("MAE_MODEL is ",mae_model)

mape_model = round(100*mean_absolute_percentage_error(test_data['Consumo (GWh)'],df2['comb_modelo_2_amostra'].iloc[115:]),2)
print("MAPE_MODEL is ",mape_model,'%')


                                              