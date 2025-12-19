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
    def __init__(self, symbol, period='2y', timeframe='1d', n_states=4):
        self.symbol = symbol
        self.period = period
        self.interval = timeframe
        self.n_states = n_states
        self.data = None
        self.model = None

   def fetch_data(self):
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=self.period, interval=self.interval)
            
            if df.empty: return False

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # --- 1. Basic Indicators ---
            # ใช้ np.log1p แทน log ปกติ หรือจัดการ 0 ก่อน เพื่อป้องกัน Infinity
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # --- 2. Trend & Location Features ---
            # EMA 200
            df['ema200'] = df['close'].rolling(window=200).mean()
            df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']

            # 52-week Range (252 days)
            df['min_52'] = df['close'].rolling(window=252).min()
            df['max_52'] = df['close'].rolling(window=252).max()
            
            # ป้องกันการหารด้วย 0 กรณี High = Low
            denom = df['max_52'] - df['min_52']
            df['price_pos'] = np.where(denom == 0, 0, (df['close'] - df['min_52']) / denom)

            # --- 3. Volatility ---
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                             abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(window=14).mean()
            df['atr_pct'] = df['atr'] / df['close']

            # --- 4. Volume ---
            df['vol_ma'] = df['volume'].rolling(window=20).mean()
            # ป้องกันหารด้วย 0
            df['rel_vol'] = np.where(df['vol_ma'] == 0, 0, df['volume'] / df['vol_ma'])

            # VWAP
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['cum_vol_price'] = (df['tp'] * df['volume']).cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            # ป้องกันหารด้วย 0
            df['vwap'] = np.where(df['cum_vol'] == 0, df['tp'], df['cum_vol_price'] / df['cum_vol'])

            # === CLEANING DATA STEP (สำคัญมาก) ===
            # 1. แทนค่า Infinity ด้วย NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 2. ลบแถวที่มี NaN (ข้อมูลช่วงต้นจะหายไป 200-252 วัน เพราะเราใช้ EMA200/MinMax252)
            df.dropna(inplace=True)

            # 3. เช็คว่าเหลือข้อมูลพอไหมหลังจากลบ
            if len(df) < 50:
                st.error(f"ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์ (เหลือ {len(df)} วัน) กรุณาเพิ่ม Period เป็น '2y' หรือ '5y' เพราะต้องใช้ข้อมูล 252 วันย้อนหลังในการคำนวณ")
                return False

            self.data = df
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

   def train_hmm(self):
        if self.data is None or self.data.empty: 
            return

        feature_cols = ['rsi', 'dist_ema200', 'atr_pct', 'rel_vol']
        
        # Double Check: ให้แน่ใจว่าไม่มี NaN หรือ Inf หลงเหลือใน Column ที่จะใช้
        X_data = self.data[feature_cols].copy()
        
        # ถ้ายังมี NaN/Inf ให้ลบทิ้งเฉพาะจุด
        if X_data.isnull().values.any() or np.isinf(X_data.values).any():
            X_data = X_data.replace([np.inf, -np.inf], np.nan).dropna()
            # update self.data ให้ index ตรงกัน
            self.data = self.data.loc[X_data.index]
        
        if X_data.empty:
            st.error("ไม่เหลือข้อมูลสำหรับ Train Model (Data Cleaned resulted in empty set)")
            return

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
            st.error(f"Error in HMM Training: {e}")
    def map_smart_money_labels(self):
        # Logic การ Map ใหม่ที่ "ฉลาดขึ้น"
        state_stats = {}
        for state in range(self.n_states):
            mask = self.data['state'] == state
            if mask.sum() == 0: continue
            
            state_stats[state] = {
                'rsi': self.data.loc[mask, 'rsi'].mean(),
                'dist_ema200': self.data.loc[mask, 'dist_ema200'].mean(), # ค่าลบ = ใต้เส้น 200 (ถูก), ค่าบวก = เหนือเส้น (แพง)
                'atr': self.data.loc[mask, 'atr_pct'].mean(), # ต่ำ = นิ่ง, สูง = เหวี่ยง
                'return': self.data.loc[mask, 'log_ret'].mean()
            }

        # การให้คะแนนเพื่อระบุ Phase (Scoring System)
        labels = {}
        for state, stats in state_stats.items():
            # กฎเหล็ก: Accumulation ต้องอยู่โซนล่าง และ นิ่ง
            if stats['dist_ema200'] < 0.05 and stats['atr'] < stats['atr'] * 1.2: 
                # ถ้าราคาอยู่ใต้ EMA200 หรือเหนือกว่านิดหน่อย และไม่ผันผวนมาก
                if stats['rsi'] < 55:
                    labels[state] = 'Accumulation (เก็บของ)'
                else:
                    labels[state] = 'Re-Accumulation / Base (พักตัว)'
            
            # กฎเหล็ก: Distribution มักอยู่โซนบน หรือ ผันผวนจัดๆ
            elif stats['dist_ema200'] > 0.10: # อยู่เหนือเส้น 200 เกิน 10%
                if stats['rsi'] > 60:
                    labels[state] = 'Distribution (ระบายของ)'
                else:
                    labels[state] = 'Markup (ขาขึ้น)'
            
            # ถ้าไม่เข้าพวก ดู Return
            else:
                if stats['return'] < -0.001:
                    labels[state] = 'Markdown (ขาลง)'
                else:
                    labels[state] = 'Markup (ขาขึ้น)'

        self.data['phase'] = self.data['state'].map(labels).fillna('Uncertain')

    def get_stats(self):
        # (คงเดิม หรือปรับตามต้องการ)
        if self.data is None: return None
        current_price = self.data['close'].iloc[-1]
        current_phase = self.data['phase'].iloc[-1]
        
        # หา VWAP ของกลุ่ม Accumulation ล่าสุด
        acc_mask = self.data['phase'] == 'Accumulation (เก็บของ)'
        if acc_mask.any():
            self.data['group'] = (self.data['phase'] != self.data['phase'].shift()).cumsum()
            # กรองเฉพาะช่วงที่พึ่งเกิดเร็วๆนี้ (ไม่เอาเก่าเกิน 1 ปี)
            recent_groups = self.data[acc_mask]['group'].unique()
            last_group = recent_groups[-1]
            last_acc_data = self.data[self.data['group'] == last_group]
            
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


