import pandas as pd
import numpy as np
import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt   # Import matplotlib
from sklearn.svm import SVR






dates = []
prices = []

# We will look at stock prices 
start = datetime.datetime(2014,1,1)
end = datetime.date.today()
 
# Let's get Apple stock data; Apple's ticker symbol is AAPL
# The source ("yahoo" for Yahoo! Finance)
#apple = pdr.get_data_yahoo('AAPL', start, end)

# The source ("google" for Google Finance)
apple = pdr.DataReader("AAPL", 'google', start, end)

#print (apple.to_string(index=False))

#microsoft = pdr.DataReader("MSFT", 'google', start, end)

#google = pdr.DataReader("GOOG", 'google', start, end)
 
# Below I create a DataFrame consisting of the adjusted closing price of these stocks
stocks = pd.DataFrame({"AAPL": apple["Close"]})
					#,"MSFT": microsoft["Close"]})
					#,"GOOG": google["Close"]})


stock_return = stocks.apply(lambda x: x / x[0])

dates = np.reshape(apple.index.values,(len(apple.index.values), 1)) # converting to matrix of n X 1
prices = apple["Close"]



# plt.figure('Actual price 2')
# plt.scatter(dates, prices, color= 'blue', label= 'Data') # plotting the initial datapoints 
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()




stock_return.plot(grid = True).axhline(y = 1, color = "red", lw = 2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price stock daily compargining date')
plt.legend()



#----05/14------
#ret = prices.pct_change()

#lagged return features
X=pd.concat([stock_return.shift(1),stock_return.shift(2),stock_return.shift(3),stock_return.shift(4)],axis=1)

#tomorrows return
Y=stock_return.shift(-1)


#combining featrues and target
col=pd.concat([X,Y],axis=1)


#removing NaN values
col=col.dropna()

#creating separte features and target variable
feat=col.iloc[:, [i for i in range(col.shape[1]) if i != 4]]
target=col.iloc[:,4]

#training features and target for machine learnging
trainfeat=feat[1:200]
traintrgt=target[1:200]


#testing features and target for machine learnging
testfeat=feat[201:]
testtrgt=target[201:]


#Support vector regression

svr_lin = SVR(kernel= 'linear', C= 1e3)
svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models

svr_rbf.fit(trainfeat, traintrgt) # fitting the data points in the models
svr_lin.fit(trainfeat, traintrgt) 
svr_poly.fit(trainfeat, traintrgt) 



testnp = testtrgt.as_matrix()
plt.figure('AAPL price')
plt.plot(testtrgt.index.values,testnp, color= 'black', label= 'Actual price')

# plotting the line made by the RBF kernel
plt.plot(testfeat.index.values, svr_rbf.predict(testfeat), color= 'red', label= 'Predicted price with RBF model') 

# plotting the line made by linear kernel
plt.plot(testfeat.index.values,svr_lin.predict(testfeat), color= 'green', label= 'Predicted price with linear model') 

# plotting the line made by polynomial kernel
plt.plot(testfeat.index.values,svr_poly.predict(testfeat), color= 'blue', label= 'Predicted price with Polynomial model') 
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression')
plt.legend()



#---------------

plt.show()




