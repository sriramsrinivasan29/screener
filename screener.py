import talib as ta
import yfinance as yf
import datetime
import datetime as dt
import streamlit as st
import base64
import matplotlib.pyplot as plt
import pandas as pd
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import time
import numpy as np
from numpy import *

from pandas_datareader import data as pdr
#from requests_ratelimiter import LimiterSession, RequestRate, Limiter, Duration

#history_rate = RequestRate(1, Duration.SECOND)
#limiter = Limiter(history_rate)
#session = LimiterSession(limiter=limiter)

#session.headers['User-agent'] = 'tickerpicker/1.0'


start = dt.datetime(2017, 12, 1)
now = dt.datetime.now()
indiaFlag=True
#end_date = "2024-01-25"
def color_survived(val):
    color = 'green' if val>0 else 'red'
    return f'background-color: {color}'
def period(days=365):
  '''
  return start and end dates
  '''
  start_date = dt.datetime(2017, 12, 1)
  end_date = datetime.date.today()
  return start_date, end_date 


def Technical_Rank(clsval):
    ma200 = ta.EMA(clsval, timeperiod=200)  # sma(clsval, 200)
    ma50 = ta.EMA(clsval, timeperiod=50)  # sma(clsval, 50)

    longtermma = 0.30 * 100 * (clsval - ma200) / ma200
    longtermroc = 0.30 * ta.ROC(clsval, timeperiod=125)

    midtermma = 0.15 * 100 * (clsval - ma50) / ma50
    midtermroc = 0.15 * ta.ROC(clsval, timeperiod=20)

    ma12 = ta.EMA(clsval, timeperiod=12)  # ema(clsval, 12)
    ma26 = ta.EMA(clsval, timeperiod=26)  # ema(clsval, 26)

    
    ppo_ta = 100 * (ma12 - ma26) / ma26

    sig = ta.EMA(ppo_ta, timeperiod=9)  # ema(ppo, 9)

    ppoHist = ppo_ta - sig
    stRsi = 0.05 * ta.RSI(clsval, timeperiod=14)
    slope = (ppoHist[-1] - ppoHist[-8]) / 3
    
    stPpo = 0.05 * 100 * slope

    trank = longtermma + longtermroc + midtermma + midtermroc + stPpo + stRsi
    trank_1 = 0 if trank[-1] < 0 else trank[-1]

    return 100 if trank_1 > 100 else trank_1


def Return(values):
    return (values[-1] - values[0]) / values[0]


def Volatility(values):
    values = np.array(values)
    returns = (values[1:] - values[:-1]) / values[:-1]
    return np.std(returns)


def efficiency_ratio(window_length, close):
    try:
        lb = window_length

        a = (close[-1] - close[0]) / (close[0])

        b = 0
        i = 0
        
        while i < lb - 1:
            b = b + abs(close[i] - close[i + 1])
            i += 1
        c = (a / b) * 100

        return c
    except Exception as e:
        print("Error")




def eff_ratio(close):
    direction = (close[-1] - close[0]) / (close[0])
    volatility = np.sum(np.absolute(np.diff(close, axis=0)), axis=0)
    c = direction / volatility
    return c


def compute(window_length, assets, close, high, low):
    lb = window_length
    e_r = np.zeros(len(assets), dtype=np.float64)
    
    HL = np.absolute(np.diff(high[1:(lb)] - low[1:(lb)], axis=0))
    where_are_NaNs = isnan(HL)
    HL[where_are_NaNs] = 0
    HL = HL.max()
    
    HC = np.absolute(np.diff(high[1:(lb)] - close[0 : (lb - 1)], axis=0))
    where_are_NaNs = isnan(HC)
    HC[where_are_NaNs] = 0
    HC = HC.max()
    
    LC = np.absolute(np.diff(low[1:(lb)] - close[0 : (lb - 1)], axis=0))
    where_are_NaNs = isnan(LC)
    LC[where_are_NaNs] = 0
    LC = LC.max()
    
    c = HL + HC + LC
    e_r = abs(close[-1] - close[0]) / c

    return e_r


def annualised_sharpe(returns, N=252):
    """
    Calculate the annualised Sharpe ratio of a returns stream
    based on a number of trading periods, N. N defaults to 252,
    which then assumes a stream of daily returns.

    The function assumes that the returns are the excess of
    those compared to a benchmark.
    """
    return np.sqrt(N) * returns.mean() / returns.std()

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Momentum Ranking.csv">Download CSV File</a>'
    return href   

def switch(index_ticker):
    if index_ticker == "Nifty 750":
        return "./dataset/ind_niftytotalmarket_list.csv"
    elif index_ticker == "Nifty Midcap 150":
        return "./dataset/ind_niftymidcap150list.csv"
    elif index_ticker == "Nifty 50":
        return "./dataset/ind_nifty50list.csv"
    elif index_ticker == "Nifty 200":
        return "./dataset/ind_nifty200list.csv"
    elif index_ticker == "Nifty 500":
        return "./dataset/ind_nifty500list.csv"
    elif index_ticker == "Nifty Smallcap 250":
        return "./dataset/ind_niftysmallcap250list.csv"
    elif index_ticker == "Nifty Microcap 250":
        return "./dataset/ind_niftymicrocap250_list.csv"
    elif index_ticker == "Nifty 100":
        return "./dataset/ind_nifty100list.csv"
    elif index_ticker == "Nifty Midcap150 Momentum 50":
        return "./dataset/ind_niftymidcap150momentum50_list.csv"
    elif index_ticker == "Nifty Smallcap250 Momentum Quality 100":
        return "./dataset/ind_niftySmallcap250MomentumQuality100_list.csv"   
    elif index_ticker=='OMXS30':
        return "./dataset/OMXS30.csv"
    elif index_ticker=='OMX Stockholm Large Cap':
        return "./dataset/OMXSLCPI.csv"
    elif index_ticker=='OMX Stockholm Mid Cap':
        return "./dataset/OMXSMCPI.csv"
    elif index_ticker=='OMX Stockholm Small Cap':
        return "./dataset/OMXSSCPI.csv"
    elif index_ticker=='OMX Stockholm All Share Cap':
        return "./dataset/OMXSCAPPI.csv"
    elif index_ticker=='OMX Stockholm 60':
        return "./dataset/OMXS60PI.csv"
        
def stock_screener(index_ticker,end_date,indiaFlag,minerveni_flag):
    print(index_ticker)
    stocklist = pd.read_csv(switch(index_ticker), header=0, index_col=0)
    fullList=stocklist
    print(stocklist)
    st.header(f'Ranking for  {index_ticker} on {end_date}')  
    latest_iteration = st.empty()
    filter_stock = st.empty()
    bar = st.progress(0)
    total = len(stocklist)

    
    if indiaFlag :
       # nifty = yf.download("^NSEI", "2010-1-1", end_date)
        #stocklist = stocklist["Symbol"] + ".NS"
        #Std_Nifty_252 = nifty["Adj Close"].rolling(window=20).std()[-1]
        Std_Nifty_252=1
    else:
        #nifty = yf.download("^OMXS30", "2010-1-1", end_date)
        #stocklist = stocklist["Symbol"]+ ".ST"
        Std_Nifty_252 = 1
    
    
    
    
   
    n = -1
    final = []
    index = []
    exportList = pd.DataFrame(
        columns=[
            "Stock",
            "Company Name",
            "Close",
            "1D",
            "ATH",
            "Score",
            "50 Day MA",
            "150 Day Ma",
            "200 Day MA",
            "52 Week Low",
            "52 week High",
            "Coffecient_Variation",
            "sharpe_ratio",
            "52Max_Drawdown",
            "price_score",
            "BetaEqualized",
            "PCRS",
            "RS_Rating",
            "Relative_Strength",
            "Efficiency_Ratio1",
            "Efficiency_Ratio2",
            "Pretty_Good_Oscillator",
            "Technical_Rank",
            "Price_To_MA",
            "factor1",
            "factor2",
            "factor3",
            "factor4",
            "factor5",
            "Linear_Reg",
            "ROC_20",
            "Position_Score",
            "RSRating_6M",
            "RSL_27",
            "Moving Average Distance",
            "High To Price Momentum",
            "KAMA",
        ]
    )
    try:

        for ticker in stocklist:

            n += 1
            time.sleep(0.10)
            print("\npulling {} with index {}".format(ticker.split(".")[1], n))
            stock_industry = ""
            
            stock = yf.download(ticker, "2017-5-2", end_date)
            stockDetails = yf.Ticker(ticker)
            # get all stock info
            stockName=stockDetails.info['longName']
            if stock.size > 2000:

                moving_average_4 = ta.SMA(stock["Adj Close"], timeperiod=4)[-1]
                moving_average_50 = ta.SMA(stock["Adj Close"], timeperiod=50)[-1]
                moving_average_150 = ta.SMA(stock["Adj Close"], timeperiod=150)[-1]
                moving_average_21 = ta.SMA(stock["Adj Close"], timeperiod=21)[-1]
                moving_average_200 = ta.SMA(stock["Adj Close"], timeperiod=200)[-1]
                moving_average_27 = ta.SMA(stock["Adj Close"], timeperiod=27)[-1]
                average_Close = stock["Adj Close"].rolling(window=21).median()[-1]
                average_Volume = stock["Volume"].rolling(window=21).median()[-1]
                Median_Trading_Value = average_Close * average_Volume
                currentClose = stock["Adj Close"][-1]
                close_252 = stock["Adj Close"][-252]
                close_126 = stock["Adj Close"][-126]
                rsrating = 100 * ((currentClose - close_252) / close_252)
                rsrating_6M = 100 * ((currentClose - close_126) / close_126)
                rsl_27 = currentClose / moving_average_27
                KAMA_27 = ta.KAMA(stock["Adj Close"], timeperiod=27)[-1]
                KAMA_27 = currentClose / KAMA_27
                MAD = moving_average_21 / moving_average_200
                try:
                    moving_average_200_20 = ta.SMA(stock["Adj Close"], timeperiod=200)[-20]

                except Exception:
                    moving_average_200_20 = 0

                ATH = stock["High"].rolling(window=756).max()[-1]
                high_of_52week = stock["High"].rolling(window=252).max()[-1]
                high_of_26week = stock["High"].rolling(window=126).max()[-1]
                low_of_52week = stock["Low"].rolling(window=252).min()[-1]
                price_change_4WK = stock["Adj Close"].pct_change(periods=21)[-1]
                price_change_13WK = stock["Adj Close"].pct_change(periods=63)[-1]
                price_change_10WK = stock["Adj Close"].pct_change(periods=48)[-1]
                price_change_26WK = stock["Adj Close"].pct_change(periods=126)[-1]
                Std_20 = stock["Adj Close"].rolling(window=20).std()[-1]
                Std_stock_252 = stock["Adj Close"].rolling(window=252).std()[-1]
                HTP = high_of_52week / close_252
                #BetaEqualized = Std_stock_252 / Std_Nifty_252
                BetaEqualized=1
                Coffecient_Variation = (Std_stock_252 / moving_average_200) * 100
                price_change_1day = (stock["Adj Close"].pct_change(periods=1)[-1]) * 100

                moving_average_89 = ta.SMA(stock["Adj Close"], timeperiod=89)[-1]
                tr = ta.TRANGE(stock["High"], stock["Low"], stock["Adj Close"])
                ATR = ta.ATR(
                    stock["High"], stock["Low"], stock["Adj Close"], timeperiod=200
                )[-1]
                positionscore = ((10000 / currentClose) * ATR) / 100
                moving_average_89EMA_TR = ta.EMA(tr, timeperiod=89)[-1]
                # print('4')
                pgo = (currentClose - moving_average_89) / moving_average_89EMA_TR

                pricetoMA = ((currentClose - moving_average_200) / moving_average_200) * 100
                efficiency_ratio1 = efficiency_ratio(252, stock["Adj Close"].iloc[-252:])
                efficiency_ratio2 = eff_ratio(stock["Adj Close"].iloc[-252:])
                eff = compute(
                    253,
                    stock["Adj Close"].iloc[-252:],
                    stock["Adj Close"].iloc[-252:],
                    stock["High"].iloc[-252:],
                    stock["Low"].iloc[-252:],
                )

                score = (
                    (moving_average_4 - low_of_52week) / (high_of_52week - low_of_52week)
                ) * 100
                # price cycle RS Model
                factor1 = currentClose / high_of_52week
                factor2 = currentClose / low_of_52week
                factor3 = price_change_4WK
                factor4 = price_change_13WK
                factor5 = price_change_26WK


                leg_momentum = ta.LINEARREG_SLOPE(stock["Adj Close"], timeperiod=90)[-1]
                PCRS = (
                    (currentClose / high_of_52week)
                    + (currentClose / low_of_52week)
                    + price_change_4WK
                    + price_change_13WK
                    + price_change_26WK
                )

                # Use the percentage change method to easily calculate daily returns
                stock["daily_ret"] = stock["Adj Close"].pct_change()

                # Assume an average annual risk-free rate over the period of 5%
                stock["excess_daily_ret"] = stock["daily_ret"] - 0.0647 / 252
                # Return the annualised Sharpe ratio based on the excess daily returns
                sharpe_ratio = annualised_sharpe(stock["excess_daily_ret"])
                # print('6')

                # Calculate the max drawdown in the past window days for each day in the series.
                # Use min_periods=1 if you want to let the first 252 days data have an expanding window
                Roll_Max = stock["High"].rolling(window=252, min_periods=1).max()

                Daily_Drawdown = stock["Adj Close"] / Roll_Max - 1.0

                # Next we calculate the minimum (negative) daily drawdown in that window.
                # Again, use min_periods=1 if you want to allow the expanding window
                Max_Daily_Drawdown = Daily_Drawdown.rolling(window=252, min_periods=1).min()
                Max_Drawdown = Max_Daily_Drawdown[-1] * 100
                price_score = (
                    (40 * currentClose / stock["Adj Close"][-21])
                    + (20 * currentClose / stock["Adj Close"][-63])
                    + (20 * currentClose / stock["Adj Close"][-126])
                    + (20 * currentClose / stock["Adj Close"][-189])
                    + (20 * currentClose / stock["Adj Close"][-252])
                    - 100
                )

                rsrating = 100 * ((currentClose - stock["Adj Close"][-252]) / currentClose)
                # relative strength IBD style
                three_month_rs = 0.4 * (currentClose / stock["Adj Close"][-63])
                six_month_rs = 0.2 * (currentClose / stock["Adj Close"][-126])
                nine_month_rs = 0.2 * (currentClose / stock["Adj Close"][-189])
                twelve_month_rs = 0.2 * (currentClose / stock["Adj Close"][-252])
                rs_rating = (
                    three_month_rs + six_month_rs + nine_month_rs + twelve_month_rs
                ) * 100



                closes = [x for x in stock["Adj Close"].iloc[-726:]]

                highs = [x for x in stock["High"].iloc[-726:]]
                lows = [x for x in stock["Low"].iloc[-726:]]


                shorttermroc = ta.ROC(stock["Adj Close"], timeperiod=20)[-1]



                


                technical_Rank = Technical_Rank(stock["Adj Close"])
                if currentClose > moving_average_50:
                    condition_1 = True
                else:
                    condition_1 = False

                if currentClose > moving_average_150:
                    condition_2 = True
                else:
                    condition_2 = False

                if currentClose > moving_average_200:
                    condition_3 = True
                else:
                    condition_3 = False

                if moving_average_150 > moving_average_200:
                    condition_4 = True
                else:
                    condition_4 = False

                if currentClose >= (1.3 * low_of_52week):
                    condition_5 = True
                else:
                    condition_5 = False

                if currentClose >= (0.75 * high_of_52week):
                    condition_6 = True
                else:
                    condition_6 = False

                if moving_average_200 > moving_average_200_20:
                    condition_7 = True
                else:
                    condition_7 = False

                if moving_average_50 > moving_average_150:
                    condition_8 = True
                else:
                    condition_8 = False

                if Median_Trading_Value > 10000000:
                    condition_9 = True
                else:
                    condition_9 = False

                if rsrating > 0:
                    condition_10 = True
                else:
                    condition_10 = False

                if efficiency_ratio1 > 1:
                    condition_11 = True
                else:
                    condition_11 = False                

                if float(technical_Rank) > 50.0:
                    condition_13 = True
                else:
                    condition_13 = False

                if currentClose >= (0.75 * ATH):
                    condition_14 = True
                else:
                    condition_14 = False
                if minerveni_flag =="Yes" :
                    filter= condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7 and condition_8 and  condition_10 and condition_1
                else :
                    filter=condition_10
                              
                if(filter) :
                    # if(condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7 and condition_8 and condition_9 and condition_10 and condition_11):
                    new_row = {
                        "Stock": ticker.split(".")[0],
                        "Company Name":stockName,
                        "Close": currentClose,
                        "1D": price_change_1day,
                        "ATH": ATH,
                        "Score": score,
                        "50 Day MA": moving_average_50,
                        "150 Day Ma": moving_average_150,
                        "200 Day MA": moving_average_200,
                        "52 Week Low": low_of_52week,
                        "52 week High": high_of_52week,
                        "Coffecient_Variation": Coffecient_Variation,
                        "sharpe_ratio": sharpe_ratio,
                        "52Max_Drawdown": Max_Drawdown,
                        "price_score": price_score,
                        "BetaEqualized": BetaEqualized,
                        "PCRS": PCRS,
                        "RS_Rating": rsrating,
                        "Relative_Strength": rs_rating,
                        "Efficiency_Ratio1": efficiency_ratio1,
                        "Efficiency_Ratio2": efficiency_ratio2,
                        "Pretty_Good_Oscillator": pgo,
                        "Technical_Rank": technical_Rank,
                        "Price_To_MA": pricetoMA,
                        "factor1": factor1,
                        "factor2": factor2,
                        "factor3": factor3,
                        "factor4": factor4,
                        "factor5": factor5,
                        "Linear_Reg": leg_momentum,
                        "ROC_20": shorttermroc,
                        "Position_Score": positionscore,
                        "RSRating_6M": rsrating_6M,
                        "RSL_27": rsl_27,
                        "Moving Average Distance": MAD,
                        "High To Price Momentum": HTP,
                        "KAMA": KAMA_27,
                    }
                    exportList = pd.concat(
                        [exportList, pd.DataFrame([new_row])], ignore_index=True
                    )
                    print(ticker.split(".")[0] + " made the requirements")
                    filter_stock.text(f'Stock passed: {ticker.split(".")[0]}')


            else:
                print(ticker.split(".")[0] + " has no data")
            latest_iteration.text(f'Stocks Processed: {(n+1)}/{total}')
            
            bar.progress((n+1)/total)
    except Exception as e:
        print(e)
    exportList["rs_rank"] = exportList["RS_Rating"].rank(ascending=False)
    exportList["rs6M_rank"] = exportList["RSRating_6M"].rank(ascending=False)
    exportList["Relative_Strength_rank"] = exportList["Relative_Strength"].rank(
        ascending=False
    )
    exportList["pricescore_rank"] = exportList["price_score"].rank(ascending=False)
    exportList["PCRS_rank"] = exportList["PCRS"].rank(ascending=False)
    exportList["PGO_rank"] = exportList["Pretty_Good_Oscillator"].rank(ascending=False)
    exportList["Technical_Rank"] = exportList["Technical_Rank"].rank(ascending=False)
    exportList["Price2MA_Rank"] = exportList["Price_To_MA"].rank(ascending=False)
    exportList["Score_Rank"] = exportList["Score"].rank(ascending=False)
    exportList["Efficiency_Ratio_Rank"] = exportList["Efficiency_Ratio1"].rank(
        ascending=False
    )
    exportList["ROC_20_Rank"] = exportList["ROC_20"].rank(ascending=False)


    exportList["Total_Rank"] = (
        exportList["rs_rank"]
        + exportList["pricescore_rank"]
        + exportList["PCRS_rank"]
        + exportList["PGO_rank"]
        + exportList["Technical_Rank"]
        + exportList["Price2MA_Rank"]
        + exportList["Score_Rank"]
        + exportList["Relative_Strength_rank"]
        + exportList["Efficiency_Ratio_Rank"]
        + exportList["ROC_20_Rank"]
    )

    exportList["Total_Rank_Index"] = exportList["Total_Rank"].rank(ascending=True)


    exportList["factor1_rank"] = exportList["factor1"].rank(ascending=False)
    exportList["factor2_rank"] = exportList["factor2"].rank(ascending=False)
    exportList["factor3_rank"] = exportList["factor3"].rank(ascending=False)
    exportList["factor4_rank"] = exportList["factor4"].rank(ascending=False)
    exportList["factor5_rank"] = exportList["factor5"].rank(ascending=False)

    exportList["Factor_Total_Rank"] = (
        exportList["factor1_rank"]
        + exportList["factor2_rank"]
        + exportList["factor3_rank"]
        + exportList["factor4_rank"]
        + exportList["factor5_rank"]
    )


    exportList["Factor_Total_Rank_Index"] = exportList["Factor_Total_Rank"].rank(
        ascending=True
    )


    exportList = exportList.sort_values(by=["rs6M_rank"], ascending=True)
    filter_stock.text(f'Total Stock(s) passed: { len(exportList.index)}')
    
    print(exportList)
    return exportList
    
    #writer = ExcelWriter("Ranking_ind_niftymidcap150list.xlsx")
    #exportList.to_excel(writer, "Sheet1")
    #writer.close()
#Settings
st.sidebar.header('Settings')
index_ticker = st.sidebar.selectbox('Index', ('Nifty 750', 'Nifty Midcap 150', 'Nifty 50','Nifty 100','Nifty 500','Nifty 200','Nifty Smallcap 250','Nifty Microcap 250','Nifty Midcap150 Momentum 50','Nifty Smallcap250 Momentum Quality 100','OMXS30','OMX Stockholm Large Cap','OMX Stockholm Mid Cap','OMX Stockholm Small Cap','OMX Stockholm All Share Cap','OMX Stockholm 60'))  #'OMX Stockholm Large Cap'
minerveni_flag = st.sidebar.selectbox('Mark Minervini Filter', ('Yes','No'))  


end_date=st.sidebar.date_input('End date', value="today", min_value=dt.datetime(2017, 12, 1), max_value=datetime.date.today(), key=None, help=None, on_change=None, args=None, kwargs=None,  format="YYYY-MM-DD", disabled=False, label_visibility="visible")
#min_volume = st.sidebar.text_input("Minimum Volume", 1e6)
#min_price = st.sidebar.slider('Minimum Price ($)', 0,5000, 0)
#days = st.sidebar.slider('Max Period (days)', 14, 730, 365)
#min_rs_rating = st.sidebar.slider('Minimum Relative Strange Rating', 1, 100, 70)






with st.container():
    st.title('Momentum Ranking')
    if st.button('Start screening'):
       
            
           
            
        if  index_ticker.__contains__('OMX'):    
            indiaFlag=False
        final_df = stock_screener(index_ticker,end_date,indiaFlag,minerveni_flag)
            
        st.dataframe(final_df.style.applymap(color_survived, subset=['1D']))

        st.markdown(filedownload(final_df), unsafe_allow_html=True)
        #st.set_option('deprecation.showPyplotGlobalUse', False)
