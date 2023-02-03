# First of all we have to import the libraries that we are going to use

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from scipy.stats import norm

yf.pdr_override()

# Then we have to download the data

startdate = '2012-01-01'
enddate = '2023-02-02'
data = pd.DataFrame()

data = pdr.get_data_yahoo('AMZN', start=startdate, end=enddate)['Adj Close']

# Variables

S = data.iloc[-1]  # We fix the last value to determine the underlying asset price
K = 120         # Strike
r = 0.0352       # Risk free rate
T = 1         # Time to Maturity


# Values

log_returns = np.log(1+data.pct_change())
Vol = log_returns.std()*252**0.5 # Historical volatily annualized (**0.5 due to the fact that the volatility grows like this over the time)

# Model

# d1 and d2 is how closer the options are 'at the money'

def d1(S, K, r, Vol, T):               
    return (np.log(S/K) + ((r + (Vol**2)/2) * T)) / (Vol * np.sqrt(T))

def d2(S, K, r, Vol, T):
    return (np.log(S/K) + ((r - (Vol**2)/2) * T)) / (Vol * np.sqrt(T))

# Finding out the option premiums

def BSMCall(S, K, r, Vol, T):
    d_one = d1(S, K, r, Vol, T)
    d_two = d2(S, K, r, Vol, T)
    return (S*norm.cdf(d_one)) - (K*np.exp(-r*T)*norm.cdf(d_two))

def BSMPut(S, K, r, Vol, T):
    d_one = d1(S, K, r, Vol, T)
    d_two = d2(S, K, r, Vol, T)
    return (K*np.exp(-r*T)*norm.cdf(-d_two)) - (S*norm.cdf(-d_one))

print("\nCall Price: \n", BSMCall(S, K, r, Vol, T), "\n")
print("\nPut Price: \n", BSMPut(S, K, r, Vol, T), "\n")