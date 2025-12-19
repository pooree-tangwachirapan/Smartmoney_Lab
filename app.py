import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="HMM Smart Money Tracker", layout="wide")

st.title("🕵️‍♂️ HMM Smart Money Tracker (US Stock)")
st.markdown("""
โปรแกรมนี้ใช้ **Hidden Markov Model (HMM)** เพื่อค้นหา **Market Regimes (Hidden States)** โดยใช้ข้อมูล Price Action, VWAP, RSI, BB, Stochastic เป็น Observable States
""")

# --- Sidebar Inputs ---
st.sidebar.header("User Settings")
ticker = st.sidebar.text_input("Stock Ticker (e.g., NVDA, TSLA, AAPL)", "NVDA")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
n_states = st.sidebar.slider("Number of Hidden States (Regimes)", 2, 5, 3)
st.sidebar.markdown("---")
st.sidebar.info("Note: 'Smart Money Avg' is calculated based on the VWAP of the current active Hidden State.")

# --- Functions ---

@st.cache_data
def get_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return None
        # Flat MultiIndex columns if exists (fix for new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_indicators(df):
    data = df.copy()
    
    # 1. VWAP
    # ใช้ try-except หรือตรวจสอบว่ามี column สร้างขึ้นหรือไม่ เพื่อความชัวร์
    try:
        data.ta.vwap(append=True)
        # หา column ที่ชื่อมีคำว่า VWAP (เพราะชื่ออาจเปลี่ยนไปตาม library version)
        vwap_col = [c for c in data.columns if 'VWAP' in c]
        if vwap_col:
            data['VWAP'] = data[vwap_col[0]]
        else:
            # Fallback ถ้าคำนวณไม่ได้ ให้ใช้ Typical Price แทนชั่วคราว
            data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3
    except:
         data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    # 2. RSI
    data.ta.rsi(length=14, append=True)
    
    # 3. Bollinger Bands (แก้ไขจุดที่ Error) -----------------------------
    # รับค่าใส่ตัวแปร bb_df ก่อน เพื่อดูชื่อ column ที่แท้จริง
    bb_df = data.ta.bbands(length=20, std=2)
    
    # รวม column เข้าไปใน data หลัก
    data = pd.concat([data, bb_df], axis=1)
    
    # pandas_ta จะคำนวณ Bandwidth มาให้แล้ว มักจะขึ้นต้นด้วย 'BBB'
    # เราจะหา column นั้นมาใช้เลย โดยไม่ต้องระบุชื่อตายตัวแบบ 'BBU_20_2.0'
    width_col = [c for c in bb_df.columns if c.startswith('BBB')]
    
    if width_col:
        data['BB_Width'] = data[width_col[0]]
    else:
        # กรณีหาไม่เจอจริงๆ ค่อยคำนวณมือจาก column แรก (Lower) และ column ที่ 3 (Upper)
        # bb_df columns: [Lower, Mid, Upper, Bandwidth, Percent]
        upper = bb_df.iloc[:, 2]
        lower = bb_df.iloc[:, 0]
        mid = bb_df.iloc[:, 1]
        data['BB_Width'] = (upper - lower) / mid
    # -------------------------------------------------------------------
    
    # 4. Stochastic
    data.ta.stoch(append=True)
    
    # 5. Volume Logic
    data['Rel_Vol'] = data['Volume'] / data['Volume'].rolling(20).mean()
    
    # 6. Price Action (Log Returns)
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Drop NaN (บรรทัดแรกๆ ที่อินดิเคเตอร์ยังคำนวณไม่ได้)
    data.dropna(inplace=True)
    
    return data

def train_hmm(data, n_states):
    # เลือก Feature ที่เป็น Observable States
    # เราใช้ Returns, RSI, BB Width, และ Relative Volume เพื่อหา Pattern
    features = ['Log_Ret', 'RSI_14', 'BB_Width', 'Rel_Vol']
    
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train HMM
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_scaled)
    
    # Predict Hidden States
    hidden_states = model.predict(X_scaled)
    
    return hidden_states, model

# --- Main Logic ---

df = get_data(ticker, start_date, end_date)

if df is not None:
    data = calculate_indicators(df)
    
    if len(data) > 50:
        # Run HMM
        hidden_states, model = train_hmm(data, n_states)
        data['Hidden_State'] = hidden_states
        
        # --- Calculate Smart Money Avg Price (Contextual VWAP) ---
        # สมมติฐาน: ราคาเฉลี่ยของ Smart Money คือ VWAP ของแท่งเทียนที่อยู่ใน State เดียวกัน
        # เราจะคำนวณ VWAP แยกตาม State
        
        data['Smart_Money_Proxy'] = np.nan
        for state in range(n_states):
            # หา VWAP สะสมเฉพาะวันที่เกิด State นั้นๆ
            mask = data['Hidden_State'] == state
            # คำนวณ VWAP แบบง่ายสำหรับกลุ่ม State (Price * Vol / Sum Vol)
            # ในที่นี้ใช้ Rolling หรือ Cumulative ของ State นั้นๆ
            state_data = data[mask]
            
            # Simple average of price in that regime (หรือจะใช้ VWAP สูตรเต็มก็ได้)
            # เพื่อความง่ายในการแสดงผล เราจะใช้ Typical Price ((H+L+C)/3) เฉลี่ยของ State นั้น
            tp = (state_data['High'] + state_data['Low'] + state_data['Close']) / 3
            weighted_price = (tp * state_data['Volume']).cumsum() / state_data['Volume'].cumsum()
            
            data.loc[mask, 'Smart_Money_Proxy'] = weighted_price

        # แสดงผลล่าสุด
        last_state = data['Hidden_State'].iloc[-1]
        st.subheader(f"Current Market Regime (Hidden State): {last_state}")
        
        # --- Visualization with Plotly ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=('Price & Smart Money Levels', 'Volume & Regimes'), 
                            row_width=[0.2, 0.7])

        # Candlestick
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'], high=data['High'],
                                     low=data['Low'], close=data['Close'],
                                     name='Price'), row=1, col=1)

        # Smart Money Proxy (Hidden State Avg)
        # Plot แยกสีตาม State
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        for state in range(n_states):
            mask = data['Hidden_State'] == state
            state_df = data[mask]
            fig.add_trace(go.Scatter(x=state_df.index, y=state_df['Smart_Money_Proxy'],
                                     mode='markers', marker=dict(color=colors[state], size=4),
                                     name=f'Smart Money (State {state})'), row=1, col=1)

        # VWAP Line
        fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], 
                                 line=dict(color='yellow', width=1, dash='dot'),
                                 name='Standard VWAP'), row=1, col=1)

        # Volume colored by State
        for state in range(n_states):
            mask = data['Hidden_State'] == state
            state_df = data[mask]
            fig.add_trace(go.Bar(x=state_df.index, y=state_df['Volume'],
                                 marker_color=colors[state],
                                 name=f'Vol State {state}'), row=2, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # --- Statistics Table ---
        st.write("### Statistics by Hidden State (Smart Money Behavior)")
        stat_df = data.groupby('Hidden_State').agg({
            'Close': ['count', 'mean'],
            'Volume': 'mean',
            'RSI_14': 'mean',
            'BB_Width': 'mean',
            'Log_Ret': 'mean'
        })
        stat_df.columns = ['Days', 'Avg Price', 'Avg Volume', 'Avg RSI', 'Avg Volatility', 'Avg Return']
        st.dataframe(stat_df)
        
    else:
        st.warning("Not enough data to train HMM. Please extend the date range.")
else:

    st.info("Please verify the ticker symbol.")
