import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 設置 Streamlit 頁面標題
st.set_page_config(
    page_title="互動式線性迴歸演示",
    layout="wide"  # 設置寬佈局，讓左右兩邊有足夠空間
)

st.title("線性迴歸互動式演示")
st.write("---")

# CRISP-DM 步驟的簡單說明 (在應用程式中體現)
st.sidebar.header("CRISP-DM 步驟 (應用程式體現)")
st.sidebar.markdown("""
- **業務理解**: 探索數據生成參數對線性迴歸的影響。
- **數據理解**: 透過圖表視覺化數據點分佈。
- **數據準備**: 程式自動生成帶有噪音的線性數據。
- **建模**: 使用 Scikit-learn 訓練線性迴歸模型。
- **評估**: 透過迴歸線與數據點的擬合程度進行視覺化評估。
- **部署**: 此 Streamlit 應用程式即為部署形式。
""")
st.sidebar.write("---")


# 左側佈局：使用者輸入控制項
st.sidebar.header("數據與模型參數設定")

# 1. 輸入 a (斜率)
a = st.sidebar.slider(
    "輸入斜率 (a)",
    min_value=-10.0,
    max_value=10.0,
    value=2.0,  # 預設值
    step=0.1
)

# 2. 輸入 var (噪音方差)
var = st.sidebar.slider(
    "輸入噪音方差 (var)",
    min_value=0.0,
    max_value=100.0,
    value=25.0,  # 預設值
    step=1.0
)

# 3. 輸入數據點數量
num_points = st.sidebar.slider(
    "輸入數據點數量",
    min_value=100,
    max_value=1000,
    value=300,  # 預設值
    step=50
)

# 右側佈局：圖形顯示
st.header("數據點與線性迴歸線")

# 生成數據
# x 值從 0 到 100 之間均勻分佈
x = np.linspace(0, 100, num_points).reshape(-1, 1)

# 生成噪音 N(0, var)
noise = np.random.normal(0, np.sqrt(var), num_points).reshape(-1, 1)

# 生成 y 值: y = ax + 30 + N(0, var)
y = a * x + 30 + noise

# 訓練線性迴歸模型
model = LinearRegression()
model.fit(x, y)

# 預測 y 值 (用於繪製迴歸線)
y_pred = model.predict(x)

# 繪製圖形
fig, ax = plt.subplots(figsize=(10, 6)) # 調整圖形大小以匹配佈局

# 繪製數據點
ax.scatter(x, y, color='blue', label='生成的數據點', alpha=0.6)

# 繪製線性迴歸線
ax.plot(x, y_pred, color='red', label='線性迴歸線', linewidth=3)

# 顯示模型學到的係數和截距
st.markdown(f"**模型學到的斜率 (a):** `{model.coef_[0][0]:.2f}`")
st.markdown(f"**模型學到的截距 (b):** `{model.intercept_[0]:.2f}`")

# 添加標題和標籤
ax.set_title("線性迴歸擬合結果", fontsize=16)
ax.set_xlabel("X 值", fontsize=12)
ax.set_ylabel("Y 值", fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

# 顯示圖形
st.pyplot(fig)

st.write("---")
st.markdown("""
**說明：**
- 左側滑桿可以調整數據的生成方式：
    - **斜率 (a)**：控制數據點的傾斜程度。
    - **噪音方差 (var)**：控制數據點的散佈程度（越大越混亂）。
    - **數據點數量**：控制生成數據點的數量。
- 右側圖形會即時更新，顯示生成的數據點和 Scikit-learn 訓練出的紅色線性迴歸線。
- 模型會嘗試學習出最能代表數據趨勢的斜率和截距，並顯示在圖形下方。
""")
