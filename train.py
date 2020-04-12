import warnings
warnings.filterwarnings('always')
warnings.filterwarnings("ignore")

import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dropout


def main():
    rn.seed(321)
    np.random.seed(321)

    df = pd.read_csv('household_power_consumption.txt', sep=';', 
                    parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                    low_memory=False, na_values=['nan','?'], index_col='dt')


    ## finding all columns that have nan:
    droping_list_all=[]
    for j in range(0,7):
        if not df.iloc[:, j].notnull().all():
            droping_list_all.append(j)        
            #print(df.iloc[:,j].unique())

    # filling nan with mean in any columns
    for j in range(0,7):        
            df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

    ## resampling of data over hour
    df_resample = df.resample('h').mean()

    ## If you would like to train based on the resampled data (over hour), then used below
    values = df_resample.values

    ## full data without resampling
    #values = df.values

    # integer encode direction
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)

    # split into train and test sets
    values = reframed.values

    n_train_time = 365*24
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    ##test = values[n_train_time:n_test_time, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

    model = create_model(train_X)
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=20, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], 7))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def create_model(train_X):
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    #    model.add(LSTM(70))
    #    model.add(Dropout(0.3))
    model.add(Dense(1))

    return model

if __name__ == '__main__':
    main()