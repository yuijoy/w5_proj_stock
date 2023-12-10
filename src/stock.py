from flask import Flask, render_template, send_file, request
import numpy as np
import pandas as pd
import os
import math
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from array import array
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import RNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import yfinance as yf

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('form.html')

@app.route('/data', methods=['POST'])
def web_head():

    stock_name = request.form['Name']
    tr = request.form['Tr_Set']
    d = request.form['His_Days']
    fea = request.form['Fea_Days']

    tr = int(tr)
    fea = int(fea)
    d = int(d)
    stock = yf.Ticker(stock_name)

    hist = stock.history(period="3y")

    #Training dataset is 70% of the total data and the remaining 30% will be predicted
    df=hist
    n=int(hist.shape[0]*0.7)
    training_set = df.iloc[:n, 1:2].values
    test_set = df.iloc[n:, 1:2].values


    #Scale and reshape the data

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(d, n-fea):
        X_train.append(training_set_scaled[i-d:i, 0])
        y_train.append(training_set_scaled[i+fea, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


    model = Sequential()

    model.add(LSTM(256, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(5))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
    model.fit(X_train, y_train, batch_size = 36, epochs = tr )

    dataset_train = df.iloc[:n, 1:2]
    dataset_test = df.iloc[n:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - d:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(d, inputs.shape[0]):
        X_test.append(inputs[i-d:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    #Predict with the model on the test data"""

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    df['Date']=df.index
    df=df.reset_index(drop=True)

    #Plot the actual and predicted data"""

    plt.plot(df.loc[n:, 'Date'],dataset_test.values, color = 'red', label = 'Actual Price')
    plt.plot(df.loc[n:, 'Date'],predicted_stock_price, color = 'blue', label = 'Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=90)
    STOCK = BytesIO()
    plt.savefig(STOCK, format="png")

    #Send the plot to plot.html

    STOCK.seek(0)
    plot_url = base64.b64encode(STOCK.getvalue()).decode('ascii')
    return render_template("plot.html", plot_url=plot_url)
