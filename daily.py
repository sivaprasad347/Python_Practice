from __future__ import division

import json
import re
import time

from pandas import DataFrame, isnull, notnull, to_datetime

from pandas_datareader._utils import RemoteDataError
from pandas_datareader.base import _DailyBaseReader
from pandas_datareader.yahoo.headers import DEFAULT_HEADERS


class YahooDailyReader(_DailyBaseReader):
    """
    Returns DataFrame of with historical over date range,
    start to end.
    To avoid being penalized by Yahoo! Finance servers, pauses between
    downloading 'chunks' of symbols can be specified.

    Parameters
    ----------
    symbols : string, array-like object (list, tuple, Series), or DataFrame
        Single stock symbol (ticker), array-like object of symbols or
        DataFrame with index containing stock symbols.
    start : string, int, date, datetime, Timestamp
        Starting date. Parses many different kind of date
        representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980'). Defaults to
        5 years before current date.
    end : string, int, date, datetime, Timestamp
        Ending date
    retry_count : int, default 3
        Number of times to retry query request.
    pause : int, default 0.1
        Time, in seconds, to pause between consecutive queries of chunks. If
        single value given for symbol, represents the pause between retries.
    session : Session, default None
        requests.sessions.Session instance to be used. Passing a session
        is an advanced usage and you must set any required
        headers in the session directly.
    adjust_price : bool, default False
        If True, adjusts all prices in hist_data ('Open', 'High', 'Low',
        'Close') based on 'Adj Close' price. Adds 'Adj_Ratio' column and drops
        'Adj Close'.
    ret_index : bool, default False
        If True, includes a simple return index 'Ret_Index' in hist_data.
    chunksize : int, default 25
        Number of symbols to download consecutively before intiating pause.
    interval : string, default 'd'
        Time interval code, valid values are 'd' for daily, 'w' for weekly,
        'm' for monthly.
    get_actions : bool, default False
        If True, adds Dividend and Split columns to dataframe.
    adjust_dividends: bool, default true
        If True, adjusts dividends for splits.
    """

    def __init__(
        self,
        symbols=None,
        start=None,
        end=None,
        retry_count=3,
        pause=0.1,
        session=None,
        adjust_price=False,
        ret_index=False,
        chunksize=1,
        interval="d",
        get_actions=False,
        adjust_dividends=True,
    ):
        super().__init__(
            symbols=symbols,
            start=start,
            end=end,
            retry_count=retry_count,
            pause=pause,
            session=session,
            chunksize=chunksize,
        )

        # Ladder up the wait time between subsequent requests to improve
        # probability of a successful retry
        self.pause_multiplier = 2.5
        if session is None:
            self.headers = DEFAULT_HEADERS
        else:
            self.headers = session.headers

        self.adjust_price = adjust_price
        self.ret_index = ret_index
        self.interval = interval
        self._get_actions = get_actions

        if self.interval not in ["d", "wk", "mo", "m", "w"]:
            raise ValueError(
                "Invalid interval: valid values are  'd', 'wk' and 'mo'. 'm' and 'w' "
                "have been implemented for backward compatibility. 'v' has been moved "
                "to the yahoo-actions or yahoo-dividends APIs."
            )
        elif self.interval in ["m", "mo"]:
            self.pdinterval = "m"
            self.interval = "mo"
        elif self.interval in ["w", "wk"]:
            self.pdinterval = "w"
            self.interval = "wk"

        self.interval = "1" + self.interval
        self.adjust_dividends = adjust_dividends

    @property
    def get_actions(self):
        return self._get_actions

    @property
    def url(self):
        return "https://finance.yahoo.com/quote/{}/history"

    # Test test_get_data_interval() crashed because of this issue, probably
    # whole yahoo part of package wasn't
    # working properly
    def _get_params(self, symbol):
        # This needed because yahoo returns data shifted by 4 hours ago.
        four_hours_in_seconds = 14400
        unix_start = int(time.mktime(self.start.timetuple()))
        unix_start += four_hours_in_seconds
        day_end = self.end.replace(hour=23, minute=59, second=59)
        unix_end = int(time.mktime(day_end.timetuple()))
        unix_end += four_hours_in_seconds

        params = {
            "period1": unix_start,
            "period2": unix_end,
            "interval": self.interval,
            "frequency": self.interval,
            "filter": "history",
            "symbol": symbol,
        }
        return params

    def _read_one_data(self, url, params):
        """read one data from specified symbol"""

        symbol = params["symbol"]
        del params["symbol"]
        url = url.format(symbol)

        resp = self._get_response(url, params=params, headers=self.headers)
        ptrn = r"root\.App\.main = (.*?);\n}\(this\)\);"
        try:
            j = json.loads(re.search(ptrn, resp.text, re.DOTALL).group(1))
            data = j["context"]["dispatcher"]["stores"]["HistoricalPriceStore"]
        except KeyError:
            msg = "No data fetched for symbol {} using {}"
            raise RemoteDataError(msg.format(symbol, self.__class__.__name__))

        # price data
        prices = DataFrame(data["prices"])
        prices.columns = [col.capitalize() for col in prices.columns]
        prices["Date"] = to_datetime(to_datetime(prices["Date"], unit="s").dt.date)

        if "Data" in prices.columns:
            prices = prices[prices["Data"].isnull()]
        prices = prices[["Date", "High", "Low", "Open", "Close", "Volume", "Adjclose"]]
        prices = prices.rename(columns={"Adjclose": "Adj Close"})

        prices = prices.set_index("Date")
        prices = prices.sort_index().dropna(how="all")

        if self.ret_index:
            prices["Ret_Index"] = _calc_return_index(prices["Adj Close"])
        if self.adjust_price:
            prices = _adjust_prices(prices)

        # dividends & splits data
        if self.get_actions and data["eventsData"]:

            actions = DataFrame(data["eventsData"])
            actions.columns = [col.capitalize() for col in actions.columns]
            actions["Date"] = to_datetime(
                to_datetime(actions["Date"], unit="s").dt.date
            )

            types = actions["Type"].unique()
            if "DIVIDEND" in types:
                divs = actions[actions.Type == "DIVIDEND"].copy()
                divs = divs[["Date", "Amount"]].reset_index(drop=True)
                divs = divs.set_index("Date")
                divs = divs.rename(columns={"Amount": "Dividends"})
                prices = prices.join(divs, how="outer")

            if "SPLIT" in types:

                def split_ratio(row):
                    if float(row["Numerator"]) > 0:
                        if ":" in row["Splitratio"]:
                            n, m = row["Splitratio"].split(":")
                            return float(m) / float(n)
                        else:
                            return eval(row["Splitratio"])
                    else:
                        return 1

                splits = actions[actions.Type == "SPLIT"].copy()
                splits["SplitRatio"] = splits.apply(split_ratio, axis=1)
                splits = splits.reset_index(drop=True)
                splits = splits.set_index("Date")
                splits["Splits"] = splits["SplitRatio"]
                prices = prices.join(splits["Splits"], how="outer")

                if "DIVIDEND" in types and not self.adjust_dividends:
                    # dividends are adjusted automatically by Yahoo
                    adj = (
                        prices["Splits"].sort_index(ascending=False).fillna(1).cumprod()
                    )
                    prices["Dividends"] = prices["Dividends"] / adj

        return prices


def _adjust_prices(hist_data, price_list=None):
    """
    Return modifed DataFrame with adjusted prices based on
    'Adj Close' price. Adds 'Adj_Ratio' column.
    """
    if price_list is None:
        price_list = "Open", "High", "Low", "Close"
    adj_ratio = hist_data["Adj Close"] / hist_data["Close"]

    data = hist_data.copy()
    for item in price_list:
        data[item] = hist_data[item] * adj_ratio
    data["Adj_Ratio"] = adj_ratio
    del data["Adj Close"]
    return data


def _calc_return_index(price_df):
    """
    Return a returns index from a input price df or series. Initial value
    (typically NaN) is set to 1.
    """
    df = price_df.pct_change().add(1).cumprod()
    mask = notnull(df.iloc[1]) & isnull(df.iloc[0])
    if mask:
        df.loc[df.index[0]] = 1

    # Check for first stock listings after starting date of index in ret_index
    # If True, find first_valid_index and set previous entry to 1.
    if not mask:
        tstamp = df.first_valid_index()
        t_idx = df.index.get_loc(tstamp) - 1
        df.iloc[t_idx] = 1

    return df




import pandas as pd
from datetime import datetime
df = ['iPhone', 'iPad', 'Mac']
df_1 = pd.DataFrame(df)

filename = df.iloc[1]
e = datetime.now()
df_1.to_excel(f"{filename}{e}.xlsx", index=False)




#import the necessary packages
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#download the historical stock data
aapl_df = yf.download("INFY", start="2018-03-24", end="2023-03-24")


"""
  calculate the short period and long period emas 
  the 20 and 50 here should be changed to whatever timeframes you wish to use
  the values are added into the same dataframe
"""
aapl_df['ema_short'] = aapl_df['Close'].ewm(span=20, adjust=False).mean()
aapl_df['ema_long'] = aapl_df['Close'].ewm(span=50, adjust=False).mean()


"""
  New column 'bullish' will hold a value of 1.0 when the ema_short > ema_long, and a value of 0.0 when ema_short < ema_long. 
  'crossover' will tell us when the crossover actually happened - when the ema_short crossed above or below the ema_long.
  'crossover' column will hold 1.0 on a cross above, and -1.0 on a cross below
"""
aapl_df['bullish'] = 0.0
aapl_df['bullish'] = np.where(aapl_df['ema_short'] > aapl_df['ema_long'], 1.0, 0.0)
aapl_df['crossover'] = aapl_df['bullish'].diff()


"""
  Finally, we will plot the chart - showing the Close and the emas, as well as the buy and sell signals using the crossover column
  A cross to the upside can be used as a buy signal.
  A cross to the downside can be used as a sell signal 
"""
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111, ylabel='Price in $')

aapl_df['Close'].plot(ax=ax1, color='b', lw=2.)
aapl_df['ema_short'].plot(ax=ax1, color='r', lw=2.)
aapl_df['ema_long'].plot(ax=ax1, color='g', lw=2.)

ax1.plot(aapl_df.loc[aapl_df.crossover == 1.0].index, 
         aapl_df.Close[aapl_df.crossover == 1.0],
         '^', markersize=10, color='g')
ax1.plot(aapl_df.loc[aapl_df.crossover == -1.0].index, 
         aapl_df.Close[aapl_df.crossover == -1.0],
         'v', markersize=10, color='r')
plt.legend(['Close', 'EMA Short', 'EMA Long', 'Buy', 'Sell'])
plt.title('AAPL EMA Crossover')










import yfinance as yf


# Define the ticker symbol of the stock to analyze
tickerSymbol = 'INFY'


# Set the start and end date of the historical data
startDate = '2019-01-01'
endDate = '2022-01-01'


# Retrieve the historical data for the stock
stockData = yf.Ticker(tickerSymbol).history(start=startDate, end=endDate)


peRatio = stockData['Close'][-1] / yf.Ticker(tickerSymbol).info['epsTrailingTwelveMonths']





import requests
url = 'https://www.nseindia.com/market-data/new-52-week-high-low-equity-market'
r = requests.get(url, allow_redirects=True)

open('facebook.ico', 'wb').write(r.content)


#####   https://www.nseindia.com/market-data/new-52-week-high-low-equity-market






from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# Setup chrome driver
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# Navigate to the url
driver.get('https://www.nseindia.com/market-data/new-52-week-high-low-equity-market')

# Find element by id
myLink = driver.find_element(By.PARTIAL_LINK_TEXT, 'Download')

# Optional
time.sleep(3)

# Click on link
myLink.click()

# Optional
time.sleep(7)

# Close the browser
driver.quit()



from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# Setup chrome driver
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# Navigate to the url
driver.get('https://www.nseindia.com/market-data/new-52-week-high-low-equity-market')

# Find all link elements
links = driver.find_elements(By.TAG_NAME, 'Nse_Links')

# Iterate over link elements
for link in links:
    print(link.get_attribute('outerHTML'))

# Close the browser
driver.quit() 

