from sklearn.svm import SVR
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd, numpy as np

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	See: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

steps = 6

df = pd.read_excel('data.xlsx', sheet_name='Training Data')
df = series_to_supervised(list(df['Data'].values), steps)
x = df.iloc[:, [a for a in range(steps - 2)]].values
y = df.iloc[:, [steps - 1]].values.ravel()

df = pd.read_excel('data.xlsx', sheet_name='Testing Data')
df = series_to_supervised(list(df['Data'].values), steps)
xtest = df.iloc[:, [a for a in range(steps - 2)]].values
ytest = df.iloc[:, [steps - 1]].values.ravel()

regressor = SVR(kernel='linear', epsilon=1.0)
regressor.fit(x, y)
ypred = regressor.predict(xtest)
mse = mean_squared_error(ytest, ypred)
rmse = sqrt(mse)
print(rmse)