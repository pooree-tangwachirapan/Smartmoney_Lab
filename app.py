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
# CLASS: Logic Core
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

    def train_hmm(self):
        if self.data is None or self.data.empty: return

        feature_cols = ['rsi', 'dist_ema200', 'atr_pct', 'rel_vol']
        X_data = self.data[feature_cols].copy()
        
        if X_data.isnull().values.any() or np.isinf(X_data.values).any():
            X_data = X_data.replace([np.inf, -np.inf], np.nan).dropna()
            self.data = self.data.loc[X_data.index]
        
        if X_data.empty: return

        X = X_data.values
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=1000, random_state=42)
            self.model.fit(X_scaled)
            hidden_states = self.model.predict(X_scaled)
            self.data['state'] = hidden_states
            self.map_smart_money_labels()
        except Exception as e:
            st.error(f"Training Error: {e}")

    def map_smart_money_labels(self):
        state_stats = {}
        for state in range(self.n_states):
            mask = self.data['state'] == state
            if mask.sum() == 0: continue
            
            state_stats[state] = {
                'return': self.data.loc[mask, 'log_ret'].mean(),
                'dist_ema200': self.data.loc[mask, 'dist_ema200'].mean(),
                'id': state
            }

        stats_list = list(state_stats.values())
        if not stats_list: return

        labels = {}
        
        markdown_state = min(stats_list, key=lambda x: x['return'])
        labels[markdown_state['id']] = 'Markdown (ขาลง)'
        stats_list.remove(markdown_state)

        if stats_list:
            markup_state = max(stats_list, key=lambda x: x['return'])
            labels[markup_state['id']] = 'Markup (ขาขึ้น)'
            stats_list.remove(markup_state)

        if stats_list:
            sorted_by_loc = sorted(stats_list, key=lambda x: x['dist_ema200'])
            labels[sorted_by_loc[0]['id']] = 'Accumulation (เก็บของ)'
            if len(sorted_by_loc) > 1:
                labels[sorted_by_loc[1]['id']] = 'Distribution (ระบายของ)'

        self.data['phase'] = self.data['state'].map(labels).fillna('Uncertain')

    def get_stats(self):
        if self.data is None: return None
        current_price = self.data['close'].iloc[-1]
        current_phase = self.data['phase'].iloc[-1]
        
        total_days = len(self.data)
        accum_days = len(self.data[self.data['phase'] == 'Accumulation (เก็บของ)'])
        accum_pct = (accum_days / total_days) * 100

        acc_mask = self.data['phase'] == 'Accumulation (เก็บของ)'
        if acc_mask.any():
            self.data['group'] = (self.data['phase
