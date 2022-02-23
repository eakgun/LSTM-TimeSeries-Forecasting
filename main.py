import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import TSMethods


df = pd.read_csv('Datasets\date_time_temp_filtered_combined.csv', parse_dates=[['Date', 'Time']], dayfirst=True, na_filter=True)


df.index = pd.to_datetime(df['Date_Time'])
df= df[df['temp'] != 0]

temp = df['temp']

temp = temp['2018':'2019']
temp = temp.dropna()
# temp = temp[::6]

if TSMethods.stationarity_test(temp, return_p=True, print_res = False) > 0.05:
    print("P Value is high. Consider Differencing: " + str(TSMethods.stationarity_test(temp, return_p = True, print_res = False)))
else:
    TSMethods.stationarity_test(temp)




def df_to_X_y(df, window_size, forecast_length):
    df_as_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_as_np) - window_size - forecast_length):
        row_x = [[a] for a in df_as_np[i:i+window_size]]
        
        X.append(row_x)
        # row_y = df_as_np[i+window_size]
        row_y = [b for b in df_as_np[i+window_size:i+window_size+forecast_length]]
        
        y.append(row_y)
        
    return np.array(X), np.array(y)




WINDOW_WIDTH = 47
FORECAST_LENGTH = 5
X, y = df_to_X_y(temp, WINDOW_WIDTH, FORECAST_LENGTH)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

X_max = X_train.max()
X_min = X_train.min()
# X_train[:,:,0] = normalize(X_train[:,:,0])






batchSize = 16
from keras.models import Sequential, save_model, load_model
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError





normalizer = Normalization()
normalizer.adapt(X_train)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)

model = Sequential()
model.add(LSTM(input_shape = X_train.shape[1:], units=86, return_sequences=True))
# model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=86))



# model.add(LSTM(units=128, return_sequences=True))
# model.add(LSTM(units=100, batch_size = batchSize, return_sequences=True))
# model.add(LSTM(units=100, batch_size = batchSize, return_sequences=True))
# model.add(Dense(units=100,  activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(units=FORECAST_LENGTH, activation='linear'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()] , loss=MeanSquaredError())

history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=50, callbacks=[rlrop])

score = model.evaluate(X_test, y_test, verbose = 1)

plt.figure(2)
plt.plot(history.history['loss'], color = 'red', label='Loss')
plt.plot(history.history['val_loss'], color = 'green', label = 'Validation Loss')
plt.legend(['Loss', 'Val_loss'], loc='upper right')



#modeli kaydet 

filePath = './saved_models'
save_model(model, filePath)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


#load ze modél 

model = load_model(filePath, compile=True)     

train_pred = model.predict(X_train).flatten()

test_pred = model.predict(X_test).flatten()




#X[len(X)//2:].flatten()

train_results = pd.DataFrame(data={'Actual':y_train.flatten() , 'Prediction':train_pred})
test_results = pd.DataFrame(data={'Actual':y_test.flatten() , 'Prediction':test_pred})

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
fig.suptitle('Actuals vs Predicted for Training and Test Data')
axes[0].set_title('Training')
sns.lineplot(data=train_results[0:100], ax=axes[0])
axes[1].set_title('Test')
sns.lineplot(data=test_results[0:100], ax=axes[1])

# plt.figure(1)
# train_results[:168].plot()
# test_results[:168].plot()
# plt.show()
# #Future Forecast
 
predictHorizon = 47


future = []
currentStep = X[-4:-3,:,:] #last step from the previous prediction
realPredictHorizon = predictHorizon//FORECAST_LENGTH


for i in range(realPredictHorizon):
    print('Calculated predict horizon: ', realPredictHorizon)
    prediction = model.predict(currentStep) 
    future.append(prediction) 
    prediction = prediction.reshape(1,FORECAST_LENGTH,1)
    currentStep = np.delete(currentStep,np.arange(FORECAST_LENGTH),axis=1)  
    currentStep = np.append(currentStep,prediction,1) 
    


future = np.array(future)
future = future.flatten()

forecast_results =  pd.DataFrame(data={'Actual':X[-3:-2,:,:].flatten()[:len(future)], 'Forecast':future})


# fig, axes = plt.plot(1, 1, sharex=True, figsize=(15,5))
fig2 = plt.figure(2)
fig2.suptitle('FORECAST')
sns.lineplot(data=forecast_results)

# future = np.array(future)
# plt.figure(2)
# plt.plot(future.flatten())
# plt.show()


#after processing a sequence, reset the states for safety
# model.reset_states()


