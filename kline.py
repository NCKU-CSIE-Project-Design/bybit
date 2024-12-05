from keys import api, secret
import requests
from requests import get
from pybit.unified_trading import HTTP
import pandas as pd
import ta
import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
session = HTTP(
    testnet=False,
    api_key=api,
    api_secret=secret
)

# Config:
tp = 0.012  # Take Profit +1.2%
sl = 0.009  # Stop Loss -0.9%
mode = 1  # 1 - Isolated, 0 - Cross
leverage = 10
qty = 50    # Amount of USDT for one order


def get_balance():
    try:
        resp = session.get_wallet_balance(accountType="CONTRACT", coin="USDT")['result']['list'][0]['coin'][0]['walletBalance']
        resp = float(resp)
        return resp
    except Exception as err:
        print(err)

print(f'Your money: {get_balance()} USDT')




def klines(symbol,yearbegin,monthbegin,daybegin,yearend,monthend,dayend,limit,timeframe):
    start_time = int(datetime.datetime(yearbegin, monthbegin,daybegin).timestamp() * 1000)  # 轉換為毫秒
    end_time = int(datetime.datetime(yearend,monthend,dayend).timestamp() * 1000)    # 轉換為毫秒
    try:
        resp = session.get_kline(
            category='linear',
            symbol=symbol,
            interval=timeframe,
            limit=limit,
            start=start_time,
            end=end_time
        )['result']['list']
        resp = pd.DataFrame(resp)
        resp.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
        resp['Time'] = pd.to_datetime(resp['Time'], unit='ms')
        resp['Time'] = resp['Time'].dt.tz_localize('UTC') #change time
        resp['Time'] = resp['Time'].dt.tz_convert('Asia/Singapore')
        resp = resp.set_index('Time')
        resp = resp.astype(float)
        resp = resp[::-1]
        return resp
    except Exception as err:
        print(err)
start_time = int(datetime.datetime(2024, 7, 18).timestamp() * 1000)  # 轉換為毫秒
end_time = int(datetime.datetime(2024, 8, 3).timestamp() * 1000)    # 轉換為毫秒

data = klines('BTCUSDT',2024,7,18,2024,8,3,200,'D') #print plot
mpf.plot(data, type='candle', style='charles', title='BTC/USDT 4 Hour Candle', ylabel='Price (USDT)', figsize=(8, 5))