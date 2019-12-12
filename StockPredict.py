import math
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import datetime

style.use('ggplot')
df = pd.read_csv('AAPL.csv', index_col='Date')
print(len(df))
df = df[['Open','Close','High','Low','Adj Close']]
df['HL_PCT'] = (df['High']-df['Low'])/df['High']

forecast_column = 'Adj Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_column].shift(-forecast_out)
df.dropna(inplace=True)
print(df['label'].head())
print(df['label'].tail())
print(len(df['label']))
print('forecast_out',forecast_out)
X = np.array((df.drop(['label'],1)))
X = preprocessing.scale(X)

y = np.array(df['label'])
y = df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train,y_train)

X_lately = X[-forecast_out:] # last forecast_out columns are predicted

forecast_set = clf.predict(X_lately)
print(forecast_set,clf.score(X_test,y_test))
df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_date_time = datetime.datetime.strptime(last_date,'%Y-%m-%d').timestamp()
one_day = 86400
next_unix = last_date_time+one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]




df['Adj Close'].plot()
df['forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()
