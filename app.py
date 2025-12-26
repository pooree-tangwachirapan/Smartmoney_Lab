import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

# ปิด Warning ที่ไม่จำเป็น
warnings.filterwarnings('ignore')

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Smart Money Analysis", layout="wide")

# ==========================================
# CSS Styles (ปรับแต่งความสวยงาม UI)
# ==========================================
st.markdown("""
<style>
    .metric-label { font-size: 14px; color: #666; }
    .metric-value { font-size: 32px; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    
    /* ปรับหัวข้อชื่อหุ้น */
    .stock-title { font-size: 36px; font-weight: 800; color: #1E1E1E; margin-bottom: 0px; }
    .stock-subtitle { font-size: 18px; color: #666; margin-top: -5px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE (ระบบจำค่า Portfolio)
# ==========================================
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = ['BTC-USD', 'TSLA', 'NVDA', 'AMD', 'GC=F']

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'BTC-USD'

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
# CLASS: Logic Core (Version 5.0: Fixed VWAP & Ranking)
# ==========================================
class SmartMoneyAnalyzer:
    def __init__(self, symbol, period='2y', timeframe='1d', n_states=4):
        self.symbol = symbol
        self.period = period
        self.interval = timeframe
        self.n_states = n_states
        self.data = None
        self.model = None
        self.asset_name = symbol # Default เป็นชื่อย่อก่อน

    def fetch_data(self):
        try:
            ticker = yf.Ticker(self.symbol)
            
            # --- ดึงชื่อเต็มสินทรัพย์ ---
            try:
                info = ticker.info
                self.asset_name = info.get('longName') or info.get('shortName') or info.get('name') or self.symbol
            except:
                self.asset_name = self.symbol

            df = ticker.history(period=self.period, interval=self.interval)
            
            if df.empty: return False

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # --- Indicators Calculation ---
            # 1. Log Return
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
            # 2. RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # 3. Bollinger Bands
            df['bb_mean'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_mean'] + (2 * df['bb_std'])
            df['bb_lower'] = df['bb_mean'] - (2 * df['bb_std'])

            # 4. Trend & Location (EMA200 & Price Position)
            df['ema200'] = df['close'].rolling(window=200).mean()
            df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']

            df['min_52'] = df['close'].rolling(window=252).min()
            df['max_52'] = df['close'].rolling(window=252).max()
            denom = df['max_52'] - df['min_52']
            df['price_pos'] = np.where(denom == 0, 0, (df['close'] - df['min_52']) / denom)

            # 5. Volatility (ATR%)
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                             abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(window=14).mean()
            df['atr_pct'] = df['atr'] / df['close']

            # 6. Relative Volume
            df['vol_ma'] = df['volume'].rolling(window=20).mean()
            df['rel_vol'] = np.where(df['vol_ma'] == 0, 0, df['volume'] / df['vol_ma'])

            # 7. VWAP (Market VWAP)
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['cum_vol_price'] = (df['tp'] * df['volume']).cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            df['vwap'] = np.where(df['cum_vol'] == 0, df['tp'], df['cum_vol_price'] / df['cum_vol'])

            # --- Data Cleaning ---
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

        # Features ที่คัดมาแล้วว่าดีที่สุด
        feature_cols = ['rsi', 'dist_ema200', 'atr_pct', 'rel_vol']
        X_data = self.data[feature_cols].copy()
        
        # Double Check NaN/Inf
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
        # 1. คำนวณสถิติของแต่ละ State
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
        
        # 2. Ranking Logic (จัดอันดับแทนการกำหนดค่าตายตัว)
        
        # หา Markdown (Return ต่ำสุด)
        markdown_state = min(stats_list, key=lambda x: x['return'])
        labels[markdown_state['id']] = 'Markdown (ขาลง)'
        stats_list.remove(markdown_state)

        # หา Markup (Return สูงสุด)
        if stats_list:
            markup_state = max(stats_list, key=lambda x: x['return'])
            labels[markup_state['id']] = 'Markup (ขาขึ้น)'
            stats_list.remove(markup_state)

        # แยก Sideway (Accumulation vs Distribution) ด้วย Location
        if stats_list:
            sorted_by_loc = sorted(stats_list, key=lambda x: x['dist_ema200'])
            
            # ตัวที่อยู่ต่ำกว่า = Accumulation
            labels[sorted_by_loc[0]['id']] = 'Accumulation (เก็บของ)'
            
            if len(sorted_by_loc) > 1:
                labels[sorted_by_loc[1]['id']] = 'Distribution (ระบายของ)'

        self.data['phase'] = self.data['state'].map(labels).fillna('Uncertain')

    def get_stats(self):
        if self.data is None: return None
        current_price = self.data['close'].iloc[-1]
        current_phase = self.data['phase'].iloc[-1]
        
        # 1. คำนวณ % Accumulation
        total_days = len(self.data)
        accum_days = len(self.data[self.data['phase'] == 'Accumulation (เก็บของ)'])
        accum_pct = (accum_days / total_days) * 100 if total_days > 0 else 0

        # 2. คำนวณ VWAP (Fix: คำนวณจากทุกช่วงที่เป็น Accumulation ไม่ใช่แค่ก้อนล่าสุด)
        acc_data = self.data[self.data['phase'] == 'Accumulation (เก็บของ)']
        
        sm_vwap = None
        if not acc_data.empty:
            # สูตร VWAP = sum(Price * Volume) / sum(Volume)
            total_vol = acc_data['volume'].sum()
            total_vol_price = (acc_data['close'] * acc_data['volume']).sum()
            
            if total_vol > 0:
                sm_vwap = total_vol_price / total_vol

        return current_price, current_phase, sm_vwap, accum_pct

# ==========================================
# UI: SIDEBAR PORTFOLIO
# ==========================================
with st.sidebar:
    st.title("💼 Portfolio")
    
    with st.expander("➕ เพิ่มหุ้น", expanded=True):
        st.text_input("ชื่อหุ้น (เช่น TSLA)", key="new_ticker_input", on_change=add_ticker)

    st.markdown("---")
    
    for ticker in st.session_state.portfolio:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"📊 {ticker}", key=f"btn_{ticker}", use_container_width=True):
                select_ticker(ticker)
        with col2:
            if st.button("❌", key=f"del_{ticker}"):
                delete_ticker(ticker)
                st.rerun()

# ==========================================
# UI: MAIN CONTENT
# ==========================================
# Input Bar
with st.container():
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        ticker_input = st.text_input("ค้นหาหุ้น", value=st.session_state.selected_ticker)
    with c2:
        period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=1)
    with c3:
        timeframe = st.selectbox("Timeframe", ["1d", "1wk"], index=0)
    with c4:
        st.write("") 
        run_btn = st.button("Analyze", type="primary", use_container_width=True)

if run_btn or ticker_input != st.session_state.get('last_run_ticker', ''):
    st.session_state.last_run_ticker = ticker_input
    st.session_state.selected_ticker = ticker_input

    analyzer = SmartMoneyAnalyzer(ticker_input, period, timeframe)
    
    with st.spinner('กำลังวิเคราะห์ข้อมูลเจ้ามือ...'):
        if analyzer.fetch_data():
            analyzer.train_hmm()
            df = analyzer.data
            
            # เรียกใช้ฟังก์ชัน get_stats ที่แก้บั๊กแล้ว (รับค่า 4 ตัวแปร)
            price, phase, sm_vwap, accum_pct = analyzer.get_stats()

            # --- HEADER ---
            st.markdown(f'<p class="stock-title">{analyzer.asset_name}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="stock-subtitle">Symbol: {ticker_input.upper()} • Timeframe: {timeframe}</p>', unsafe_allow_html=True)

            # --- METRICS ---
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("ราคาตลาด", f"${price:,.2f}")
            with m2:
                if sm_vwap:
                    diff_pct = ((price - sm_vwap) / sm_vwap) * 100
                    st.metric("ต้นทุนเจ้ามือ (VWAP)", f"${sm_vwap:,.2f}", f"{diff_pct:.2f}%")
                else:
                    st.metric("ต้นทุนเจ้ามือ", "N/A", "ไม่พบข้อมูล")
            with m3:
                st.metric("% เวลาเก็บของ", f"{accum_pct:.1f}%", help="เปอร์เซ็นต์ของช่วงเวลาทั้งหมดที่อยู่ในสถานะ Accumulation")
            with m4:
                color_map = {
                    'Accumulation (เก็บของ)': '#00C805', 
                    'Markup (ขาขึ้น)': '#0066FF',
                    'Distribution (ระบายของ)': '#FF9900', 
                    'Markdown (ขาลง)': '#FF3333'
                }
                phase_color = color_map.get(phase, 'black')
                st.markdown(f"""
                <div style="font-size: 14px; color: #666;">สถานะตลาด:</div>
                <div style="font-size: 20px; font-weight: bold; color: {phase_color};">{phase}</div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # --- CHART ---
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05
            )
