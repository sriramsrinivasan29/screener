import datetime as dt
import streamlit as st
import base64
import pandas as pd
import time
import numpy as np
from numpy import isnan
import yfinance as yf

# resilient NSE import
equity_history = None
try:
    from nsepython import equity_history  # provided by nsepythonserver
    nse_module_used = "nsepython"
except ImportError:
    try:
        from nsepythonserver import equity_history
        nse_module_used = "nsepythonserver"
    except ImportError:
        equity_history = None
        nse_module_used = None

from nsetools import Nse
from requests_ratelimiter import LimiterSession, RequestRate, Limiter, Duration

# Rate limiter stub (used earlier for yfinance-ish patterns, optional)
history_rate = RequestRate(1, Duration.SECOND)
limiter = Limiter(history_rate)
session = LimiterSession(limiter=limiter)
session.headers['User-agent'] = 'tickerpicker/1.0'

nse_obj = Nse()

# ----------------- Indicator replacements (no ta-lib) -----------------

def sma(series, period):
    return series.rolling(window=period, min_periods=1).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def roc(series, period):
    return (series / series.shift(period) - 1) * 100

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    atr_vals = pd.Series(index=tr.index, dtype=float)
    if len(tr) >= period:
        atr_vals.iloc[period - 1] = tr.iloc[:period].mean()
        for i in range(period, len(tr)):
            prev = atr_vals.iloc[i - 1]
            atr_vals.iloc[i] = (prev * (period - 1) + tr.iloc[i]) / period
    else:
        atr_vals[:] = np.nan
    return atr_vals

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=period, min_periods=period).mean()
    roll_down = down.rolling(window=period, min_periods=period).mean()
    rs = roll_up / roll_down
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series

def linearreg_slope(series, period):
    idx = np.arange(period)
    def slope(x):
        if len(x) < period:
            return np.nan
        m, _ = np.polyfit(idx, x, 1)
        return m
    return series.rolling(window=period).apply(slope, raw=True)

def kama(series, period=10, fast=2, slow=30):
    change = series.diff(period)
    volatility = series.diff().abs().rolling(window=period).sum()
    er = change.abs() / volatility
    er = er.replace([np.inf, -np.inf], 0).fillna(0)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama_series = pd.Series(index=series.index, dtype=float)
    if len(series) >= period:
        kama_series.iloc[period - 1] = series.iloc[:period].mean()
        for i in range(period, len(series)):
            kama_series.iloc[i] = kama_series.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama_series.iloc[i - 1])
    return kama_series

# ----------------- Supporting analytics functions -----------------

def Technical_Rank(clsval):
    ma200 = ema(clsval, 200)
    ma50 = ema(clsval, 50)

    longtermma = 0.30 * 100 * (clsval - ma200) / ma200
    longtermroc = 0.30 * roc(clsval, 125)

    midtermma = 0.15 * 100 * (clsval - ma50) / ma50
    midtermroc = 0.15 * roc(clsval, 20)

    ma12 = ema(clsval, 12)
    ma26 = ema(clsval, 26)

    ppo_ta = 100 * (ma12 - ma26) / ma26
    sig = ema(ppo_ta, 9)
    ppoHist = ppo_ta - sig
    stRsi = 0.05 * rsi(clsval, 14)
    slope_val = (ppoHist.iloc[-1] - ppoHist.iloc[-8]) / 3 if len(ppoHist.dropna()) >= 8 else 0
    stPpo = 0.05 * 100 * slope_val

    trank = longtermma + longtermroc + midtermma + midtermroc + stPpo + stRsi
    trank_last = trank.iloc[-1] if not trank.empty else 0
    trank_1 = 0 if trank_last < 0 else trank_last
    return 100 if trank_1 > 100 else trank_1

def efficiency_ratio(window_length, close):
    try:
        lb = window_length
        a = (close[-1] - close[0]) / (close[0]) if close[0] != 0 else 0
        b = 0
        for i in range(lb - 1):
            b += abs(close[i] - close[i + 1])
        c = (a / b) * 100 if b != 0 else 0
        return c
    except Exception:
        return 0

def eff_ratio(close):
    direction = (close[-1] - close[0]) / (close[0]) if close[0] != 0 else 0
    volatility = np.sum(np.absolute(np.diff(close, axis=0)), axis=0)
    return direction / volatility if volatility != 0 else 0

def compute(window_length, assets, close, high, low):
    lb = window_length
    HL = np.absolute(np.diff(high[1:(lb)] - low[1:(lb)], axis=0))
    HL[np.isnan(HL)] = 0
    HL_val = HL.max() if HL.size else 0

    HC = np.absolute(np.diff(high[1:(lb)] - close[0 : (lb - 1)], axis=0))
    HC[np.isnan(HC)] = 0
    HC_val = HC.max() if HC.size else 0

    LC = np.absolute(np.diff(low[1:(lb)] - close[0 : (lb - 1)], axis=0))
    LC[np.isnan(LC)] = 0
    LC_val = LC.max() if LC.size else 0

    c = HL_val + HC_val + LC_val
    return abs(close[-1] - close[0]) / c if c != 0 else 0

def annualised_sharpe(returns, N=252):
    return np.sqrt(N) * returns.mean() / returns.std() if returns.std() != 0 else 0

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="Momentum Ranking.csv">Download CSV File</a>'

def color_survived(val):
    color = "green" if val > 0 else "red"
    return f"background-color: {color}"

# ----------------- Data fetch / caching -----------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nse_history(ticker: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    if equity_history is None:
        raise RuntimeError("NSE data library not available.")
    symbol = ticker.split(".")[0].upper()
    sd_str = start_date.strftime("%d-%m-%Y")
    ed_str = end_date.strftime("%d-%m-%Y")
    raw = equity_history(symbol, series="EQ", start_date=sd_str, end_date=ed_str)
    df = pd.DataFrame(raw)
    if df.empty:
        raise ValueError(f"No NSE data for {symbol} from {sd_str} to {ed_str}")
    df = df.rename(columns={"Close": "Adj Close"})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df.set_index("Date", inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    for col in ["High", "Low", "Volume", "Adj Close"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_index()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_history(ticker: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    df = df.rename(columns={"Close": "Adj Close"})
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_name_nse(symbol: str) -> str:
    try:
        quote = nse_obj.get_quote(symbol.upper())
        return quote.get("companyName", symbol)
    except Exception:
        return symbol

def get_stock_and_name(ticker: str, start_date: dt.datetime, end_date: dt.datetime):
    if ticker.endswith(".NS"):
        try:
            stock = fetch_nse_history(ticker, start_date, end_date)
            name = get_company_name_nse(ticker.split(".")[0])
            return stock, name, nse_module_used or "nsepython"
        except Exception as e:
            st.warning(f"NSE fetch failed for {ticker}: {e}. Falling back to yfinance.")
    stock = fetch_yfinance_history(ticker, start=start_date, end=end_date) if False else fetch_yfinance_history(ticker, start_date, end_date)
    try:
        info_name = yf.Ticker(ticker).info.get("longName") or ticker.split(".")[0]
    except Exception:
        info_name = ticker.split(".")[0]
    return stock, info_name, "yfinance"

# ----------------- Index mapping -----------------

def switch(index_ticker):
    mapping = {
        "Nifty 750": "./dataset/ind_niftytotalmarket_list.csv",
        "Nifty Midcap 150": "./dataset/ind_niftymidcap150list.csv",
        "Nifty 50": "./dataset/ind_nifty50list.csv",
        "Nifty 200": "./dataset/ind_nifty200list.csv",
        "Nifty 500": "./dataset/ind_nifty500list.csv",
        "Nifty Smallcap 250": "./dataset/ind_niftysmallcap250list.csv",
        "Nifty Microcap 250": "./dataset/ind_niftymicrocap250_list.csv",
        "Nifty 100": "./dataset/ind_nifty100list.csv",
        "Nifty Midcap150 Momentum 50": "./dataset/ind_niftymidcap150momentum50_list.csv",
        "Nifty Smallcap250 Momentum Quality 100": "./dataset/ind_niftySmallcap250MomentumQuality100_list.csv",
        "OMXS30": "./dataset/OMXS30.csv",
        "OMX Stockholm Large Cap": "./dataset/OMXSLCPI.csv",
        "OMX Stockholm Mid Cap": "./dataset/OMXSMCPI.csv",
        "OMX Stockholm Small Cap": "./dataset/OMXSSCPI.csv",
        "OMX Stockholm All Share Cap": "./dataset/OMXSCAPPI.csv",
        "OMX Stockholm 60": "./dataset/OMXS60PI.csv",
    }
    return mapping.get(index_ticker)

# ----------------- Screener -----------------

def stock_screener(index_ticker, end_date, indiaFlag, minerveni_flag):
    # Load list and build tickers
    df_list = pd.read_csv(switch(index_ticker), header=0)
    if "Symbol" not in df_list.columns:
        st.error(f"CSV for {index_ticker} missing 'Symbol' column.")
        return pd.DataFrame()

    # Prepare tickers (add .NS for Indian)
    if indiaFlag:
        tickers = df_list["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS")
    else:
        tickers = df_list["Symbol"].astype(str).str.strip()

    # Company name mapping from CSV (uppercase symbol key)
    symbol_to_name = dict(
        zip(df_list["Symbol"].astype(str).str.upper(), df_list.get("Company Name", df_list["Symbol"]).astype(str))
    )

    st.header(f"Ranking for {index_ticker} on {end_date}")
    latest_iteration = st.empty()
    filter_stock = st.empty()
    bar = st.progress(0)
    total = len(tickers)

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

    start_dt = dt.datetime(2017, 5, 2)
    n = -1

    try:
        for ticker in tickers:
            n += 1
            time.sleep(0.10)
            base_symbol = ticker.split(".")[0].upper()
            st.text(f"Pulling {base_symbol} ({n+1}/{total})")

            try:
                stock, fetched_name, source = get_stock_and_name(ticker, start_dt, end_date)
            except Exception as e:
                st.warning(f"Skipping {base_symbol}: fetch error {e}")
                latest_iteration.text(f"Stocks Processed: {n+1}/{total}")
                bar.progress((n + 1) / total)
                continue

            stockName = symbol_to_name.get(base_symbol, fetched_name)

            if stock.shape[0] < 200:
                st.info(f"{base_symbol} has insufficient data from {source}")
                latest_iteration.text(f"Stocks Processed: {n+1}/{total}")
                bar.progress((n + 1) / total)
                continue

            try:
                currentClose = stock["Adj Close"].iloc[-1]
                close_252 = stock["Adj Close"].iloc[-252]
                close_126 = stock["Adj Close"].iloc[-126]
            except Exception:
                latest_iteration.text(f"Stocks Processed: {n+1}/{total}")
                bar.progress((n + 1) / total)
                continue

            moving_average_4 = sma(stock["Adj Close"], 4).iloc[-1]
            moving_average_50 = sma(stock["Adj Close"], 50).iloc[-1]
            moving_average_150 = sma(stock["Adj Close"], 150).iloc[-1]
            moving_average_21 = sma(stock["Adj Close"], 21).iloc[-1]
            moving_average_200 = sma(stock["Adj Close"], 200).iloc[-1]
            moving_average_27 = sma(stock["Adj Close"], 27).iloc[-1]
            average_Close = stock["Adj Close"].rolling(window=21).median().iloc[-1]
            average_Volume = stock["Volume"].rolling(window=21).median().iloc[-1]
            Median_Trading_Value = average_Close * average_Volume

            rsrating = 100 * ((currentClose - close_252) / close_252)
            rsrating_6M = 100 * ((currentClose - close_126) / close_126)
            rsl_27 = currentClose / moving_average_27
            KAMA_27_raw = kama(stock["Adj Close"], period=27).iloc[-1]
            KAMA_27 = currentClose / KAMA_27_raw if KAMA_27_raw not in (0, np.nan) else np.nan
            MAD = moving_average_21 / moving_average_200
            try:
                moving_average_200_20 = sma(stock["Adj Close"], 200).iloc[-20]
            except Exception:
                moving_average_200_20 = 0

            ATH = stock["High"].rolling(window=756).max().iloc[-1]
            high_of_52week = stock["High"].rolling(window=252).max().iloc[-1]
            high_of_26week = stock["High"].rolling(window=126).max().iloc[-1]
            low_of_52week = stock["Low"].rolling(window=252).min().iloc[-1]
            price_change_4WK = stock["Adj Close"].pct_change(periods=21).iloc[-1]
            price_change_13WK = stock["Adj Close"].pct_change(periods=63).iloc[-1]
            price_change_10WK = stock["Adj Close"].pct_change(periods=48).iloc[-1]
            price_change_26WK = stock["Adj Close"].pct_change(periods=126).iloc[-1]
            Std_stock_252 = stock["Adj Close"].rolling(window=252).std().iloc[-1]
            HTP = high_of_52week / close_252
            BetaEqualized = 1
            Coffecient_Variation = (Std_stock_252 / moving_average_200) * 100
            price_change_1day = stock["Adj Close"].pct_change(periods=1).iloc[-1] * 100

            moving_average_89 = sma(stock["Adj Close"], 89).iloc[-1]
            tr = true_range(stock["High"], stock["Low"], stock["Adj Close"])
            ATR = atr(stock["High"], stock["Low"], stock["Adj Close"], period=200).iloc[-1]
            positionscore = ((10000 / currentClose) * ATR) / 100
            moving_average_89EMA_TR = ema(tr, 89).iloc[-1]
            pgo = (currentClose - moving_average_89) / moving_average_89EMA_TR if moving_average_89EMA_TR != 0 else 0
            pricetoMA = ((currentClose - moving_average_200) / moving_average_200) * 100
            efficiency_ratio1 = efficiency_ratio(252, stock["Adj Close"].iloc[-252:].values)
            efficiency_ratio2 = eff_ratio(stock["Adj Close"].iloc[-252:].values)
            eff = compute(
                253,
                stock["Adj Close"].iloc[-252:].values,
                stock["Adj Close"].iloc[-252:].values,
                stock["High"].iloc[-252:].values,
                stock["Low"].iloc[-252:].values,
            )

            score = ((moving_average_4 - low_of_52week) / (high_of_52week - low_of_52week)) * 100
            factor1 = currentClose / high_of_52week
            factor2 = currentClose / low_of_52week
            factor3 = price_change_4WK
            factor4 = price_change_13WK
            factor5 = price_change_26WK

            leg_momentum = linearreg_slope(stock["Adj Close"], 90).iloc[-1]
            PCRS = (
                (currentClose / high_of_52week)
                + (currentClose / low_of_52week)
                + price_change_4WK
                + price_change_13WK
                + price_change_26WK
            )

            stock["daily_ret"] = stock["Adj Close"].pct_change()
            stock["excess_daily_ret"] = stock["daily_ret"] - 0.0647 / 252
            sharpe_ratio = annualised_sharpe(stock["excess_daily_ret"])

            Roll_Max = stock["High"].rolling(window=252, min_periods=1).max()
            Daily_Drawdown = stock["Adj Close"] / Roll_Max - 1.0
            Max_Daily_Drawdown = Daily_Drawdown.rolling(window=252, min_periods=1).min()
            Max_Drawdown = Max_Daily_Drawdown.iloc[-1] * 100

            price_score = (
                (40 * currentClose / stock["Adj Close"].iloc[-21])
                + (20 * currentClose / stock["Adj Close"].iloc[-63])
                + (20 * currentClose / stock["Adj Close"].iloc[-126])
                + (20 * currentClose / stock["Adj Close"].iloc[-189])
                + (20 * currentClose / stock["Adj Close"].iloc[-252"])
                - 100
            )

            rsrating = 100 * ((currentClose - stock["Adj Close"].iloc[-252]) / currentClose)
            three_month_rs = 0.4 * (currentClose / stock["Adj Close"].iloc[-63])
            six_month_rs = 0.2 * (currentClose / stock["Adj Close"].iloc[-126])
            nine_month_rs = 0.2 * (currentClose / stock["Adj Close"].iloc[-189])
            twelve_month_rs = 0.2 * (currentClose / stock["Adj Close"].iloc[-252])
            rs_rating = (three_month_rs + six_month_rs + nine_month_rs + twelve_month_rs) * 100

            shorttermroc = roc(stock["Adj Close"], 20).iloc[-1]
            technical_Rank = Technical_Rank(stock["Adj Close"])

            # Conditions
            condition_1 = currentClose > moving_average_50
            condition_2 = currentClose > moving_average_150
            condition_3 = currentClose > moving_average_200
            condition_4 = moving_average_150 > moving_average_200
            condition_5 = currentClose >= (1.3 * low_of_52week)
            condition_6 = currentClose >= (0.75 * high_of_52week)
            condition_7 = moving_average_200 > moving_average_200_20
            condition_8 = moving_average_50 > moving_average_150
            condition_9 = Median_Trading_Value > 10000000
            condition_10 = rsrating > 0
            condition_11 = efficiency_ratio1 > 1
            condition_13 = float(technical_Rank) > 50.0
            condition_14 = currentClose >= (0.75 * ATH)

            if minerveni_flag == "Yes":
                passed = (
                    condition_1
                    and condition_2
                    and condition_3
                    and condition_4
                    and condition_5
                    and condition_6
                    and condition_7
                    and condition_8
                    and condition_10
                    and condition_1
                )
            else:
                passed = condition_10

            if passed:
                new_row = {
                    "Stock": base_symbol,
                    "Company Name": stockName,
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
                exportList = pd.concat([exportList, pd.DataFrame([new_row])], ignore_index=True)
                filter_stock.text(f"Stock passed: {base_symbol}")

            latest_iteration.text(f"Stocks Processed: {n+1}/{total}")
            bar.progress((n + 1) / total)

    except Exception as e:
        st.error(f"Unexpected error during screening: {e}")

    # Ranking logic
    if not exportList.empty:
        exportList["rs_rank"] = exportList["RS_Rating"].rank(ascending=False)
        exportList["rs6M_rank"] = exportList["RSRating_6M"].rank(ascending=False)
        exportList["Relative_Strength_rank"] = exportList["Relative_Strength"].rank(ascending=False)
        exportList["pricescore_rank"] = exportList["price_score"].rank(ascending=False)
        exportList["PCRS_rank"] = exportList["PCRS"].rank(ascending=False)
        exportList["PGO_rank"] = exportList["Pretty_Good_Oscillator"].rank(ascending=False)
        exportList["Technical_Rank"] = exportList["Technical_Rank"].rank(ascending=False)
        exportList["Price2MA_Rank"] = exportList["Price_To_MA"].rank(ascending=False)
        exportList["Score_Rank"] = exportList["Score"].rank(ascending=False)
        exportList["Efficiency_Ratio_Rank"] = exportList["Efficiency_Ratio1"].rank(ascending=False)
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

        for i in range(1, 6):
            exportList[f"factor{i}_rank"] = exportList[f"factor{i}"].rank(ascending=False)

        exportList["Factor_Total_Rank"] = sum(
            exportList[f"factor{i}_rank"] for i in range(1, 6)
        )
        exportList["Factor_Total_Rank_Index"] = exportList["Factor_Total_Rank"].rank(ascending=True)

        exportList = exportList.sort_values(by=["rs6M_rank"], ascending=True)

    filter_stock.text(f"Total Stock(s) passed: {len(exportList.index)}")
    return exportList

# ----------------- Streamlit UI -----------------

st.sidebar.header("Settings")
index_ticker = st.sidebar.selectbox(
    "Index",
    (
        "Nifty 750",
        "Nifty Midcap 150",
        "Nifty 50",
        "Nifty 100",
        "Nifty 500",
        "Nifty 200",
        "Nifty Smallcap 250",
        "Nifty Microcap 250",
        "Nifty Midcap150 Momentum 50",
        "Nifty Smallcap250 Momentum Quality 100",
        "OMXS30",
        "OMX Stockholm Large Cap",
        "OMX Stockholm Mid Cap",
        "OMX Stockholm Small Cap",
        "OMX Stockholm All Share Cap",
        "OMX Stockholm 60",
    ),
)
minerveni_flag = st.sidebar.selectbox("Mark Minervini Filter", ("Yes", "No"))
end_date = st.sidebar.date_input(
    "End date",
    value=dt.date.today(),
    min_value=dt.datetime(2017, 12, 1),
    max_value=dt.date.today(),
    format="YYYY-MM-DD",
)

indiaFlag = not index_ticker.__contains__("OMX")

with st.container():
    st.title("Momentum Ranking")
    if st.button("Start screening"):
        final_df = stock_screener(index_ticker, end_date, indiaFlag, minerveni_flag)
        if not final_df.empty:
            st.dataframe(final_df.style.applymap(color_survived, subset=["1D"]))
            st.markdown(filedownload(final_df), unsafe_allow_html=True)
        else:
            st.info("No stocks passed the filters / no data available.")
