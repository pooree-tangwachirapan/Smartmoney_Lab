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

# ตั้งค่าหน้าเว็บ (Wide Layout)
st.set_page_config(page_title="AI Smart Money Analysis", layout="wide")

# ==========================================
# CSS Styles (ปรับแต่งให้เหมือนรูป)
# ==========================================
st.markdown("""
<style>
    /* ปรับ font และ spacing */
    .metric-label { font-size: 14px; color: #666; }
    .metric-value { font-size: 32px; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; }
    /* ซ่อน Decoration ด้านบนของ Streamlit */
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CLASS: Logic Core (เหมือนเดิมแต่ปรับจูน)
# ==========================================
class SmartMoneyAnalyzer:
    def __init__(self, symbol, period='1y', timeframe='1d', n_states=4):
        self.symbol = symbol
        self.period = period
        self.interval = timeframe
        self.n_states = n_states
        self.data = None
        self.model = None

    def fetch_data(self):
        try:
            # แปลง Period/Interval ให้ตรงกับ format ของ yfinance
            # ticker.history รองรับ interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=self.period, interval=self.interval)
            
            if df.empty: return False

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # --- Indicators ---
            # 1. VWAP
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['cum_vol_price'] = (df['tp'] * df['volume']).cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            df['vwap'] = df['cum_vol_price'] / df['cum_vol']

            # 2. RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # 3. Features for HMM
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['vol_ma'] = df['volume'].rolling(window=20).mean()
            df['rel_vol'] = df['volume'] / df['vol_ma']
            df['dist_vwap'] = (df['close'] - df['vwap']) / df['vwap']

            df.dropna(inplace=True)
            self.data = df
            return True
        except Exception as e:
            return False

    def train_hmm(self):
        if self.data is None: return

        # Features: RSI, Relative Volume, Log Return
        X = self.data[['rsi', 'rel_vol', 'log_ret']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=1000, random_state=42)
        self.model.fit(X_scaled)
        hidden_states = self.model.predict(X_scaled)
        self.data['state'] = hidden_states
        self.map_thai_labels()

    def map_thai_labels(self):
        # Logic การ Map State (Simplified)
        state_stats = {}
        for state in range(self.n_states):
            mask = self.data['state'] == state
            if mask.sum() == 0: continue
            state_stats[state] = {
                'return': self.data.loc[mask, 'log_ret'].mean(),
                'rsi': self.data.loc[mask, 'rsi'].mean()
            }
        
        # Sort by Return to guess phases
        sorted_states = sorted(state_stats.items(), key=lambda x: x[1]['return'])
        
        # Mapping logic
        labels = {}
        # Return ต่ำสุด = Markdown (ขาลง)
        labels[sorted_states[0][0]] = 'Markdown (ขาลง)'
        # Return สูงสุด = Markup (ขาขึ้น)
        labels[sorted_states[-1][0]] = 'Markup (ขาขึ้น)'
        
        # เหลือตรงกลาง
        middle = sorted_states[1:-1]
        if len(middle) >= 2:
            # RSI ต่ำกว่า = Accumulation (เก็บของ)
            if middle[0][1]['rsi'] < middle[1][1]['rsi']:
                labels[middle[0][0]] = 'Accumulation (เก็บของ)'
                labels[middle[1][0]] = 'Distribution (ระบายของ)'
            else:
                labels[middle[0][0]] = 'Distribution (ระบายของ)'
                labels[middle[1][0]] = 'Accumulation (เก็บของ)'
        elif len(middle) == 1:
            labels[middle[0][0]] = 'Accumulation (เก็บของ)'

        self.data['phase'] = self.data['state'].map(labels)

    def get_stats(self):
        if self.data is None: return None
        
        current_price = self.data['close'].iloc[-1]
        current_phase = self.data['phase'].iloc[-1]
        
        # หา VWAP ของช่วง Accumulation ล่าสุด
        acc_mask = self.data['phase'] == 'Accumulation (เก็บของ)'
        if acc_mask.any():
            # กรองเอาเฉพาะช่วง Accumulation ท้ายสุด
            self.data['group'] = (self.data['phase'] != self.data['phase'].shift()).cumsum()
            last_group = self.data[acc_mask]['group'].iloc[-1]
            last_acc_data = self.data[self.data['group'] == last_group]
            
            # คำนวณ VWAP ในช่วงนั้น
            sm_vwap = (last_acc_data['close'] * last_acc_data['volume']).sum() / last_acc_data['volume'].sum()
        else:
            sm_vwap = None

        return current_price, current_phase, sm_vwap

# ==========================================
# UI SECTION
# ==========================================

st.subheader("🤖 AI Smart Money Analysis")

# 1. INPUT SECTION (แนวนอนด้านบน)
with st.container():
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        ticker = st.text_input("ค้นหาหุ้นหรือคริปโต (เช่น BTC-USD, AAPL)", value="BTC-USD")
    with c2:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "max"], index=1)
    with c3:
        timeframe = st.selectbox("Timeframe", ["1d", "1wk"], index=0)
    with c4:
        st.write("") # Spacer
        run_btn = st.button("Analyze", type="primary", use_container_width=True)

if run_btn or ticker:
    analyzer = SmartMoneyAnalyzer(ticker, period, timeframe)
    with st.spinner('กำลังวิเคราะห์ข้อมูลเจ้ามือ...'):
        if analyzer.fetch_data():
            analyzer.train_hmm()
            df = analyzer.data
            price, phase, sm_vwap = analyzer.get_stats()

            # 2. METRICS SECTION (แสดงผลตัวเลขใหญ่ๆ)
            m1, m2, m3 = st.columns([1, 1.5, 1.5])
            
            with m1:
                st.metric("ราคาตลาด", f"${price:,.2f}")
            
            with m2:
                if sm_vwap:
                    diff_pct = ((price - sm_vwap) / sm_vwap) * 100
                    st.metric("ต้นทุนเจ้ามือ (Accum VWAP)", f"${sm_vwap:,.2f}", f"{diff_pct:.2f}% vs Market", delta_color="normal")
                else:
                    st.metric("ต้นทุนเจ้ามือ (Accum VWAP)", "N/A", "ไม่พบช่วงเก็บของล่าสุด")

            with m3:
                # Custom HTML สำหรับ Phase เพื่อใส่สีตัวอักษร
                color_map = {
                    'Accumulation (เก็บของ)': '#00C805', # เขียว
                    'Markup (ขาขึ้น)': '#0066FF',        # น้ำเงิน
                    'Distribution (ระบายของ)': '#FF9900', # ส้ม
                    'Markdown (ขาลง)': '#FF3333'         # แดง
                }
                phase_color = color_map.get(phase, 'black')
                st.markdown(f"""
                <div style="font-size: 14px; color: #666; margin-bottom: 4px;">สถานะ:</div>
                <div style="font-size: 24px; font-weight: bold; color: {phase_color};">
                    {phase}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 3. CHART SECTION (กราฟเส้น + จุดสี)
            
            # สร้าง Subplot (บน=ราคา, ล่าง=RSI)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3],
                                specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

            # -- กราฟราคา (เส้นสีเทาจางๆ เป็นพื้นหลัง) --
            fig.add_trace(go.Scatter(
                x=df.index, y=df['close'],
                mode='lines',
                line=dict(color='lightgray', width=1),
                name='Price',
                showlegend=True
            ), row=1, col=1)

            # -- จุดสีตาม Phase (Overlay) --
            # เราจะ Loop สร้าง Trace แยกแต่ละ Phase เพื่อให้ Legend ขึ้นแยกสีชัดเจน
            phases_order = ['Accumulation (เก็บของ)', 'Markup (ขาขึ้น)', 'Distribution (ระบายของ)', 'Markdown (ขาลง)']
            colors_list = ['#00C805', '#0066FF', '#FF9900', '#FF3333'] # เขียว, น้ำเงิน, ส้ม, แดง
            
            for p_name, p_color in zip(phases_order, colors_list):
                subset = df[df['phase'] == p_name]
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset.index, y=subset['close'],
                        mode='markers',
                        marker=dict(color=p_color, size=4),
                        name=p_name
                    ), row=1, col=1)

            # -- RSI Chart (ด้านล่าง) --
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi'],
                mode='lines', line=dict(color='#9370DB', width=1.5), # สีม่วงอ่อน
                name='RSI'
            ), row=2, col=1)
            
            # เส้น RSI Levels (70, 30)
            fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

            # -- Layout Styling --
            fig.update_layout(
                height=600,
                template='plotly_white', # พื้นหลังขาวตามรูป
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            # ปรับแกน Y
            fig.update_yaxes(title_text="", showgrid=True, gridcolor='#f0f0f0', row=1, col=1)
            fig.update_yaxes(title_text="", range=[0, 100], showgrid=True, gridcolor='#f0f0f0', row=2, col=1)
            fig.update_xaxes(showgrid=False)

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"ไม่พบข้อมูลสำหรับ {ticker} กรุณาตรวจสอบชื่อย่อหุ้น")
