import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

# ปิด Warning
warnings.filterwarnings('ignore')

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Smart Money Analysis", layout="wide")

# ==========================================
# CSS Styles
# ==========================================
st.markdown("""
<style>
    .metric-label { font-size: 14px; color: #666; }
    .metric-value { font-size: 32px; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stock-title { font-size: 36px; font-weight: 800; color: #1E1E1E; margin-bottom: 0px; }
    .stock-subtitle { font-size: 18px; color: #666; margin-top: -5px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = ['^GSPC', 'BTC-USD', '^VIX', '^NDX', 'GC=F']

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = '^GSPC'

def add_ticker():
    new_ticker = st.session_state.new_ticker_input.upper().strip()
    if new_ticker and new_ticker not in st.session_state.portfolio:
        st.session_state.portfolio.append(new_ticker)
        st.session_state.new_ticker_input = ""

def delete_ticker(ticker):
    if ticker in st.session_state.portfolio:
        st.session_state.portfolio.remove(ticker)

def select_ticker(ticker):
    st.session_state.selected_ticker = ticker

# ==========================================
# CLASS: Logic Core (No Volume Profile)
# ==========================================
class SmartMoneyAnalyzer:
    def __init__(self, symbol, period='2y', timeframe='1d', n_states=4):
        self.symbol = symbol
        self.period = period
        self.interval = timeframe
        self.n_states = n_states
        self.data = None
        self.model = None
        self.asset_name = symbol

    def fetch_data(self):
        try:
            ticker = yf.Ticker(self.symbol)
            try:
                info = ticker.info
                self.asset_name = info.get('longName') or info.get('shortName') or info.get('name') or self.symbol
            except:
                self.asset_name = self.symbol

            df = ticker.history(period=self.period, interval=self.interval)
            
            if df.empty: return False

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # --- Indicators ---
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['bb_mean'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_mean'] + (2 * df['bb_std'])
            df['bb_lower'] = df['bb_mean'] - (2 * df['bb_std'])

            # Trend & Location
            df['ema200'] = df['close'].rolling(window=200).mean()
            df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']

            df['min_52'] = df['close'].rolling(window=252).min()
            df['max_52'] = df['close'].rolling(window=252).max()
            denom = df['max_52'] - df['min_52']
            df['price_pos'] = np.where(denom == 0, 0, (df['close'] - df['min_52']) / denom)

            # Volatility
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                             abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(window=14).mean()
            df['atr_pct'] = df['atr'] / df['close']

            # Volume
            df['vol_ma'] = df['volume'].rolling(window=20).mean()
            df['rel_vol'] = np.where(df['vol_ma'] == 0, 0, df['volume'] / df['vol_ma'])

            # VWAP
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['cum_vol_price'] = (df['tp'] * df['volume']).cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            df['vwap'] = np.where(df['cum_vol'] == 0, df['tp'], df['cum_vol_price'] / df['cum_vol'])

            # Clean Data
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            if len(df) < 50:
                st.error(f"ข้อมูลไม่เพียงพอ ({len(df)} วัน) กรุณาใช้ Period '2y' หรือ '5y'")
                return False

            self.data = df
            return True
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")
            return False

    def
