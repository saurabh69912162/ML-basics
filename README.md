# ML-basics
Basic ML and DL Concepts "that every one does!"
Stock Market Prediction – Linear Regression Example

Stock – Apple (AAPL)

________________________________________________________________________

# Read csv and make the Date column the index
df = read_csv(‘AAPL.csv’, index_col=’Date’)
print(len(df))

+++++++++++++++++++++++++++OUTPUT+++++++++++++++++++++++++++++++++++++++

1259 # length of df

+++++++++++++++++++++++++++OUTPUT+++++++++++++++++++++++++++++++++++++++


# Selecting few parameters that will be used as features for model
df = df[['Open','Close','High','Low','Adj Close']]

# New feature 
df['HL_PCT'] = (df['High']-df['Low'])/df['High']

# Label or forecast column
forecast_column = 'Adj Close'


df.fillna(-99999, inplace=True)

 calculating forecast_out -> which is int value 
 and creating a new label column with ‘Adj Close’ as Label feature
 calculate forecast_out and clear the number of columns from label column
 and then drop them


forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_column].shift(-forecast_out)

#droped the last columns with NA, i.e. shifted Columns
df.dropna(inplace=True)

print(df['label'].head())
print(df['label'].tail())
print(len(df['label']))

+++++++++++++++++++++++++++OUTPUT+++++++++++++++++++++++++++++++++++++++
Date
2014-12-11    101.419060
2014-12-12    100.454300
2014-12-15     97.624336
2014-12-16     97.633545
2014-12-17     99.002556
Name: label, dtype: float64

Date
2019-11-15    265.579987
2019-11-18    270.709991
2019-11-19    266.920013
2019-11-20    268.480011
2019-11-21    270.769989
Name: label, dtype: float64

1246 # len of label

Original Length of DF = 1259
forecast_out = 13
Length of Label = 1259 -13 = 1246

# Which means the label contains only 1246 values, which will be divided in test and train data, and the remaining 13 values will be predicted by the classifier.

+++++++++++++++++++++++++++OUTPUT+++++++++++++++++++++++++++++++++++++++


# X is every column except LABEL
X = np.array((df.drop(['label'],1)))
X = preprocessing.scale(X)

# Y is LABEL
y = np.array(df['label'])
y = df.dropna(inplace=True)
y = np.array(df['label'])



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train,y_train)


# from Last 13 Columns to End
# 1246 -> to -> 1259 = X_lately

X_lately = X[-forecast_out:]
forecast_set = clf.predict(X_lately)
print(forecast_set,clf.score(X_test,y_test))


+++++++++++++++++++++++++++OUTPUT+++++++++++++++++++++++++++++++++++++++

[263.33227266 262.87541045 266.63360613 264.98051493 265.52986886
 269.80232595 268.67017476 269.52003695 271.70631647 272.55340568
 273.48752096 265.75046097 268.91167739] 

0.9653110201242423
Accuracy ~ 96.5

+++++++++++++++++++++++++++OUTPUT+++++++++++++++++++++++++++++++++++++++


# new column for the saving the prediction of last 13 rows, currently NAN

df['forecast'] = np.nan


# Index column is DATE, which has a particular format - ‘%Y-%m-%d’,taking the index column and parsing the name, then converting the name to datetime obj and then finally to timestamp

last_date = df.iloc[-1].name
last_date_time = datetime.datetime.strptime(last_date,'%Y-%m-%d').timestamp()
one_day = 86400
next_unix = last_date_time+one_day

# Here next unix will be last date i.e. 2019-11-21 -> timestamp, we are doing this because later we will reconvert the timestamp and place the corresponding values in Forecast column

# forecast_set has 13 predicted values

for i in forecast_set:
    # calculating next date and putting value in the forecast column
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # next_Date = index column value
    # [i] is the predicted value
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj Close'].plot()
df['forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()
