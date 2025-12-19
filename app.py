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
# CSS Styles (ปรับแต่ง UI)
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
# CLASS: Logic Core (Version 4.0: Name + Ranking)
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
            
            # --- ส่วนที่เพิ่ม: ดึงชื่อเต็มสินทรัพย์ ---
            try:
                # พยายามดึงชื่อจาก info (อาจจะช้านิดนึงในบางครั้ง)
                info = ticker.info
                self.asset_name = info.get('longName') or info.get('shortName') or info.get('name') or self.symbol
            except:
                self.asset_name = self.symbol
            # -------------------------------------

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
        # ใช้ Logic Ranking System (เปรียบเทียบในตัวเอง)
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
        
        # 1. Markdown (Return ต่ำสุด)
        markdown_state = min(stats_list, key=lambda x: x['return'])
        labels[markdown_state['id']] = 'Markdown (ขาลง)'
        stats_list.remove(markdown_state)

        # 2. Markup (Return สูงสุด)
        if stats_list:
            markup_state = max(stats_list, key=lambda x: x['return'])
            labels[markup_state['id']] = 'Markup (ขาขึ้น)'
            stats_list.remove(markup_state)

        # 3. Accumulation vs Distribution (Sideway ที่เหลือ)
        if stats_list:
            # เรียงตาม Location (EMA200)
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
        
        acc_mask = self.data['phase'] == 'Accumulation (เก็บของ)'
        if acc_mask.any():
            self.data['group'] = (self.data['phase'] != self.data['phase'].shift()).cumsum()
            recent_groups = self.data[acc_mask]['group'].unique()
            last_group = recent_groups[-1]
            last_acc_data = self.data[self.data['group'] == last_group]
            
            sm_vwap = (last_acc_data['close'] * last_acc_data['volume']).sum() / last_acc_data['volume'].sum()
        else:
            sm_vwap = None

        return current_price, current_phase, sm_vwap

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
    
    with st.spinner('กำลังโหลดข้อมูลและวิเคราะห์...'):
        if analyzer.fetch_data():
            analyzer.train_hmm()
            df = analyzer.data
            price, phase, sm_vwap = analyzer.get_stats()

            # --- HEADER: แสดงชื่อหุ้น ---
            st.markdown(f'<p class="stock-title">{analyzer.asset_name}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="stock-subtitle">Symbol: {ticker_input.upper()} • Timeframe: {timeframe}</p>', unsafe_allow_html=True)

            # --- METRICS ---
            m1, m2, m3 = st.columns([1, 1.5, 1.5])
            with m1:
                st.metric("ราคาตลาด", f"${price:,.2f}")
            with m2:
                if sm_vwap:
                    diff_pct = ((price - sm_vwap) / sm_vwap) * 100
                    st.metric("ต้นทุนเจ้ามือ (Accum VWAP)", f"${sm_vwap:,.2f}", f"{diff_pct:.2f}% vs Market")
                else:
                    st.metric("ต้นทุนเจ้ามือ", "N/A", "ไม่พบข้อมูลเก็บของ")
            with m3:
                color_map = {
                    'Accumulation (เก็บของ)': '#00C805', 
                    'Markup (ขาขึ้น)': '#0066FF',
                    'Distribution (ระบายของ)': '#FF9900', 
                    'Markdown (ขาลง)': '#FF3333'
                }
                phase_color = color_map.get(phase, 'black')
                st.markdown(f"""
                <div style="font-size: 14px; color: #666;">สถานะตลาด:</div>
                <div style="font-size: 24px; font-weight: bold; color: {phase_color};">{phase}</div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # --- CHART ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])

            fig.add_trace(go.Scatter(
                x=df.index, y=df['close'], mode='lines',
                line=dict(color='lightgray', width=1), name='Price'
            ), row=1, col=1)

            phases_order = ['Accumulation (เก็บของ)', 'Markup (ขาขึ้น)', 'Distribution (ระบายของ)', 'Markdown (ขาลง)']
            colors_list = ['#00C805', '#0066FF', '#FF9900', '#FF3333']
            
            for p_name, p_color in zip(phases_order, colors_list):
                subset = df[df['phase'] == p_name]
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset.index, y=subset['close'],
                        mode='markers', marker=dict(color=p_color, size=4),
                        name=p_name
                    ), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#9370DB', width=1.5), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

            fig.update_layout(height=600, template='plotly_white', margin=dict(l=20, r=20, t=10, b=20),
                              hovermode="x unified", title_text="")
            
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"ไม่พบข้อมูลสำหรับ {ticker_input}")
