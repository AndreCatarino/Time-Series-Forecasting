# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:08:56 2022

"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.holtwinters import Holt

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm  
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_absolute_error

#CARNES BRANCAS
df1=pd.read_excel('C:/Users/saral/Desktop/Trabalhos/Trabalho MP/Carnes brancas/Carnes brancas.xlsx', header=0, index_col=0, parse_dates=True)

print(df1.head())
print(df1.info())

df1.plot()

                                        #DECOMPOSIÇÃO

seasonal_decompose(df1, model='aditive').plot()
seasonal_decompose(df1, model='multiplicative').plot() 

#Integridade dos dados (são contínuos e não há falta nem repetição de valores)
cross_tab = pd.crosstab(index=df1.index.year, columns=df1.index.month)


                                #ALISAMENTO EXPONENCIAL:
    
# Train and Test data
train_data = df1.iloc[:32]
test_data = df1.iloc[32:]



# Holt
h_model = Holt(train_data['Consumo (kg)']).fit()
h_test_pred = h_model.forecast(8).rename('Previsão do Consumo com Holt')
print(h_test_pred)


train_data['Consumo (kg)'].plot(legend=True,label='Treino')
test_data['Consumo (kg)'].plot(legend=True,label='Teste')
h_test_pred.plot(legend=True,label='Previsão do Consumo com Holt')

# Medidas dos erros
rmse_h = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], h_test_pred)),2)
print("RMSE_H is ",rmse_h)
mae_h = round(mean_absolute_error(test_data['Consumo (kg)'],h_test_pred),2)
print("MAE_H is ",mae_h)
mape_h = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],h_test_pred),2)
print("MAPE_H is ",mape_h,'%')



# Amortecido
d_model = Holt(train_data['Consumo (kg)'], damped_trend=True).fit()
d_test_pred = d_model.forecast(8).rename('Previsão do Consumo com Tendência Amortecida')
print(d_test_pred)


train_data['Consumo (kg)'].plot(legend=True,label='Treino')
test_data['Consumo (kg)'].plot(legend=True,label='Teste')
d_test_pred.plot(legend=True,label='Previsão do Consumo com Tendência Amortecida')

# Medidas dos erros
rmse_d = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], d_test_pred)),2)
print("RMSE_D is ",rmse_d)
mae_d = round(mean_absolute_error(test_data['Consumo (kg)'],d_test_pred),2)
print("MAE_D is ",mae_d)
mape_d = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],d_test_pred),2)
print("MAPE_D is ",mape_d,'%')



# Exponencial
e_model = Holt(train_data['Consumo (kg)'], exponential=True).fit()
e_test_pred = e_model.forecast(8).rename('Previsão do Consumo com Tendência Exponencial')
print(e_test_pred)

train_data['Consumo (kg)'].plot(legend=True,label='Treino')
test_data['Consumo (kg)'].plot(legend=True,label='Teste')
e_test_pred.plot(legend=True,label='Previsão do Consumo com Tendência Exponencial')

# Medidas dos erros
rmse_e = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], e_test_pred)),2)
print("RMSE_E is ",rmse_e)
mae_e = round(mean_absolute_error(test_data['Consumo (kg)'],e_test_pred),2)
print("MAE_E is ",mae_e)
mape_e = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],e_test_pred),2)
print("MAPE_E is ",mape_e,'%')



                                #MODELOS ARMA/ARIMA:
                                            
#Cálculo de FAC e de FACP para os dados originais
fig0 = plt.figure(figsize=(12,8))
ax1 = fig0.add_subplot(211)
fig0 = sm.graphics.tsa.plot_acf(df1, lags=19, ax=ax1)
ax2 = fig0.add_subplot(212)
fig0 = sm.graphics.tsa.plot_pacf(df1, lags=19, ax=ax2)

#Teste de raízes unitárias para o período de treino
def test_stationarity(timeseries): 
    #Dickey-Fuller test:
    print('Resultados do teste Dickey-Fuller:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)   

test_stationarity(train_data)

#Transformações para estacionarizar:
    
#Logarítmo
train_data_log= train_data.apply(lambda x: np.log(x))  
test_stationarity(train_data_log)
train_data_log.plot(figsize=(12,8), title= 'Logarítmo ', fontsize=14)

#Primeira diferença -> estacionária a 5%
train_data['first_difference'] = train_data - train_data.shift(1)  
test_stationarity(train_data.first_difference.dropna(inplace=False))
train_data['first_difference'].plot(figsize=(12,8), title= 'Primeira diferença', fontsize=14)

#Segunda diferença -> estacionária a 1%
train_data['second_difference'] =train_data['first_difference']-train_data['first_difference'].shift(1)
test_stationarity(train_data.second_difference.dropna(inplace=False))
train_data['second_difference'].plot(figsize=(12,8), title= 'Segunda diferença', fontsize=14)

#FAC e FACP da série estacionarizada

#Primeira diferença
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_data.first_difference.iloc[1:], lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_data.first_difference.iloc[1:],lags=14, ax=ax2)

#Segunda diferença
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_data.second_difference.iloc[2:], lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_data.second_difference.iloc[2:],lags=14, ax=ax2)


#ARIMA

#Melhores parâmetros do ARIMA para minmizar AIC:
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
params=[]
aic=[]

import statsmodels.api as sm
for param in pdq:
    mod = ARIMA(train_data['Consumo (kg)'] ,order=param, enforce_stationarity=True, enforce_invertibility=True)
    results = mod.fit()
    params.append(param)
    aic.append(results.aic)
        
parameter_options=pd.DataFrame({'params':params,'AIC':aic})

print(parameter_options.sort_values(by='AIC'))

#Modelo 1
model1 = ARIMA(train_data['Consumo (kg)'], order=(0,2,1))
results1 = model1.fit()
print(results1.summary())

resid1 = pd.DataFrame(results1.resid)
resid1.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pd.DataFrame(results1.resid).iloc[1:], lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pd.DataFrame(results1.resid).iloc[1:],lags=14, ax=ax2)

df1['forecast com ARIMA (0,2,1)'] = results1.predict(start = 32,
                                   end= 40, dynamic= True)  
df1[['Consumo (kg)', 'forecast com ARIMA (0,2,1)']].iloc[1:].plot(figsize=(12, 8))

# Medidas dos erros
rmse_model1 = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], df1['forecast com ARIMA (0,2,1)'].iloc[32:])),2)
print("RMSE_MODEL1 is ",rmse_model1)

mae_model1 = round(mean_absolute_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (0,2,1)'].iloc[32:]),2)
print("MAE_MODEL1 is ",mae_model1)

mape_model1 = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (0,2,1)'].iloc[32:]),2)
print("MAPE_MODEL1 is ",mape_model1,'%')



#Modelo 2
model2 = ARIMA(train_data['Consumo (kg)'], order=(1,2,1))
results2 = model2.fit()
print(results2.summary())

resid2 = pd.DataFrame(results2.resid)
resid2.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pd.DataFrame(results2.resid).iloc[1:], lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pd.DataFrame(results2.resid).iloc[1:],lags=14, ax=ax2)

df1['forecast com ARIMA (1,2,1)'] = results2.predict(start = 32,
                                   end= 40, dynamic= True)  
df1[['Consumo (kg)', 'forecast com ARIMA (1,2,1)']].iloc[1:].plot(figsize=(12, 8))

# Medidas dos erros
rmse_model2 = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], df1['forecast com ARIMA (1,2,1)'].iloc[32:])),2)
print("RMSE_MODEL2 is ",rmse_model2)

mae_model2 = round(mean_absolute_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (1,2,1)'].iloc[32:]),2)
print("MAE_MODEL2 is ",mae_model2)

mape_model2 = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (1,2,1)'].iloc[32:]),2)
print("MAPE_MODEL2 is ",mape_model2,'%')



#Modelo 3:
model3 = ARIMA(train_data['Consumo (kg)'], order=(0,2,2))
results3 = model3.fit()
print(results3.summary())

resid3 = pd.DataFrame(results3.resid)
resid3.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pd.DataFrame(results3.resid).iloc[1:], lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pd.DataFrame(results3.resid).iloc[1:],lags=14, ax=ax2)

df1['forecast com ARIMA (0,2,2)'] = results3.predict(start = 32,
                                   end= 40, dynamic= True)  
df1[['Consumo (kg)', 'forecast com ARIMA (0,2,2)']].iloc[1:].plot(figsize=(12, 8))

# Medidas dos erros
rmse_model3 = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], df1['forecast com ARIMA (0,2,2)'].iloc[32:])),2)
print("RMSE_MODEL3 is ",rmse_model3)

mae_model3 = round(mean_absolute_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (0,2,2)'].iloc[32:]),2)
print("MAE_MODEL3 is ",mae_model3)

mape_model3 = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (0,2,2)'].iloc[32:]),2)
print("MAPE_MODEL3 is ",mape_model3,'%')



#Modelo 4:
model4 = ARIMA(train_data['Consumo (kg)'], order=(1,2,2))
results4 = model4.fit()
print(results4.summary())

resid4 = pd.DataFrame(results4.resid)
resid4.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pd.DataFrame(results4.resid).iloc[1:], lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pd.DataFrame(results4.resid).iloc[1:],lags=14, ax=ax2)

df1['forecast com ARIMA (1,2,2)'] = results4.predict(start = 32,
                                   end= 40, dynamic= True)  
df1[['Consumo (kg)', 'forecast com ARIMA (1,2,2)']].iloc[1:].plot(figsize=(12, 8))

# Medidas dos erros
rmse_model4 = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], df1['forecast com ARIMA (1,2,2)'].iloc[32:])),2)
print("RMSE_MODEL4 is ",rmse_model4)

mae_model4 = round(mean_absolute_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (1,2,2)'].iloc[32:]),2)
print("MAE_MODEL4 is ",mae_model4)

mape_model4 = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (1,2,2)'].iloc[32:]),2)
print("MAPE_MODEL4 is ",mape_model4,'%')



#Modelo 5: (escolha através do correlograma)
model5 = ARIMA(train_data['Consumo (kg)'], order=(2,2,1))
results5 = model5.fit()
print(results5.summary())

resid5 = pd.DataFrame(results5.resid)
resid5.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pd.DataFrame(results5.resid).iloc[2:], lags=14, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pd.DataFrame(results5.resid).iloc[2:],lags=14, ax=ax2)

df1['forecast com ARIMA (2,2,1)'] = results5.predict(start = 32,
                                   end= 40, dynamic= True)  
df1[['Consumo (kg)', 'forecast com ARIMA (2,2,1)']].iloc[2:].plot(figsize=(12, 8))

# Medidas dos erros
rmse_model5 = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], df1['forecast com ARIMA (2,2,1)'].iloc[32:])),2)
print("RMSE_MODEL5 is ",rmse_model5)

mae_model5 = round(mean_absolute_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (2,2,1)'].iloc[32:]),2)
print("MAE_MODEL5 is ",mae_model5)

mape_model5 = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],df1['forecast com ARIMA (2,2,1)'].iloc[32:]),2)
print("MAPE_MODEL5 is ",mape_model5,'%')


                                          #PREVISÃO:

#Previsão futura (6 anos) com função predict e com o melhor modelo:
future_dates = pd.date_range(start="2021-01-01", end="2026-01-01", freq = "YS")

pred = e_model.predict(start=len(df1)+1, end=len(df1)+6) 

pred.index = future_dates

for i in range(len(pred)):
    new_row = {'Previsao_futura_predict': pred.iloc[i]}
    df1 = df1.append(new_row, ignore_index=True)

df1[['Consumo (kg)','Previsao_futura_predict']].plot(figsize=(12, 8))



#Combinação de previsões futuras (6 anos) com função predict e com os 3 melhores modelos
#(e_model, model4  e model1) ponderados igualmente 

#Definir df1 com a configuração inicial anterior 
df1.drop(["Previsao_futura_predict"],axis=1, inplace=True)
df1.drop(df1.tail(len(pred)).index,inplace=True)

future_dates = pd.date_range(start="2021-01-1", end="2026-12-01", freq = "YS")

#e_model
pred_1 = e_model.predict(start=len(df1)+1, end=len(df1)+6)
pred_1.index = future_dates

#model4
pred_2 = results4.predict(start=len(df1)+1, end=len(df1)+6)
pred_2.index = future_dates

#model1
pred_3 = results1.predict(start=len(df1)+1, end=len(df1)+6)
pred_3.index = future_dates

comb_modelo_1 = (pred_1 + pred_2 + pred_3) /3
print(comb_modelo_1)

for i in range(len(comb_modelo_1)):
    new_row = {'Previsao_futura_combModel1': comb_modelo_1.iloc[i]}
    df1 = df1.append(new_row, ignore_index=True)

df1[['Consumo (kg)','Previsao_futura_combModel1']].plot(figsize=(12, 8))

#Combinação de previsões futuras (6 anos) com função predict e com os 3 melhores modelos
#(e_model, model4  e model1) por ordem de desempenho (EAM))            

#Definir df1 com a configuração inicial anterior 
df1.drop(["Previsao_futura_combModel1"],axis=1, inplace=True)
df1.drop(df1.tail(len(comb_modelo_1)).index,inplace=True)

#M corresponde à soma do EAM dos 3 melhores modelos selecionados
M = 6.42
EAM = [1.41, 2.50, 2.51] #EAM dos 3 modelos referidos

future_dates = pd.date_range(start="2021-01-1", end="2026-12-01", freq = "YS")

#e_model
pred_1 = e_model.predict(start=len(df1)+1, end=len(df1)+6)
pred_1.index = future_dates
pred_1 = pred_1.apply(lambda x: x*(M-EAM[0]))

#model4
pred_2 = results4.predict(start=len(df1)+1, end=len(df1)+6)
pred_2.index = future_dates
pred_2 = pred_2.apply(lambda x: x*(M-EAM[1]))

#model1
pred_3 = results1.predict(start=len(df1)+1, end=len(df1)+6)
pred_3.index = future_dates
pred_3 = pred_3.apply(lambda x: x*(M-EAM[2]))

comb_modelo_2 = (pred_1 + pred_2 + pred_3) /((3-1)*M)
print(comb_modelo_2)

for i in range(len(comb_modelo_2)):
    new_row = {'Previsao_futura_combModel2': comb_modelo_2.iloc[i]}
    df1 = df1.append(new_row, ignore_index=True)

df1[['Consumo (kg)','Previsao_futura_combModel2']].plot(figsize=(12, 8))


    
#Obtenção das previsões para o período de teste, utilizando combinação de previsões poderadas igualmente
df1.drop(["Previsao_futura_combModel2"],axis=1, inplace=True)
df1.drop(df1.tail(len(comb_modelo_2)).index,inplace=True)

#e_model
pred_1 = e_model.predict(start = 32, end=39)
#model4
pred_2 = results4.predict(start=32, end=39)

#model1
pred_3 = results1.predict(start=32, end=39)

comb_modelo_1_amostra = (pred_1 + pred_2 + pred_3)/3
print(comb_modelo_1_amostra)

df1['comb_modelo_1_amostra'] = np.nan
df1['comb_modelo_1_amostra'].iloc[32:40] = comb_modelo_1_amostra

df1[['Consumo (kg)','comb_modelo_1_amostra']].plot(figsize=(12, 8))

#Medidas dos erros
rmse_model = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], df1['comb_modelo_1_amostra'].iloc[32:])),2)
print("RMSE_MODEL is ",rmse_model)

mae_model = round(mean_absolute_error(test_data['Consumo (kg)'],df1['comb_modelo_1_amostra'].iloc[32:]),2)
print("MAE_MODEL is ",mae_model)

mape_model = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],df1['comb_modelo_1_amostra'].iloc[32:]),2)
print("MAPE_MODEL is ",mape_model,'%')
   
    
   
#Obtenção das previsões para o período de teste, utilizando combinação de previsões poderadas por ordem de desempenho (EAM)
df1.drop(["comb_modelo_1_amostra"],axis=1, inplace=True)

#M corresponde à soma do EAM dos 3 melhores modelos selecionados
M = 6.42
EAM = [1.41, 2.50, 2.51] #EAM dos 3 modelos referidos   

#e_model
pred_1 = e_model.predict(start = 32, end=39)
pred_1 = pred_1.apply(lambda x: x*(M-EAM[0]))

#model4
pred_2 = results4.predict(start=32, end=39)
pred_2 = pred_2.apply(lambda x: x*(M-EAM[1]))

#model1
pred_3 = results1.predict(start=32, end=39)
pred_3 = pred_3.apply(lambda x: x*(M-EAM[2]))

comb_modelo_2_amostra = (pred_1 + pred_2 + pred_3)/((3-1)*M)
print(comb_modelo_2_amostra)

df1['comb_modelo_2_amostra'] = np.nan
df1['comb_modelo_2_amostra'].iloc[32:40] = comb_modelo_2_amostra

df1[['Consumo (kg)','comb_modelo_2_amostra']].plot(figsize=(12, 8))

#Medidas dos erros
rmse_model = round(np.sqrt(mean_squared_error(test_data['Consumo (kg)'], df1['comb_modelo_2_amostra'].iloc[32:])),2)
print("RMSE_MODEL is ",rmse_model)

mae_model = round(mean_absolute_error(test_data['Consumo (kg)'],df1['comb_modelo_2_amostra'].iloc[32:]),2)
print("MAE_MODEL is ",mae_model)

mape_model = round(100*mean_absolute_percentage_error(test_data['Consumo (kg)'],df1['comb_modelo_2_amostra'].iloc[32:]),2)
print("MAPE_MODEL is ",mape_model,'%')    
    
    
    