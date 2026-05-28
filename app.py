# ==========================================
# AI DEMAND FORECASTING DASHBOARD
# ==========================================

# Run using:
# streamlit run app.py

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: white;
}

[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333333;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("📈 AI Demand Forecasting & Inventory Optimization")

st.markdown("""
Interactive AI-powered dashboard for restaurant demand forecasting,
inventory optimization, and sales analytics.
""")

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/raw/train.csv")

    return df

df = load_data()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Dashboard Filters")

selected_meal = st.sidebar.selectbox(
    "Select Meal ID",
    sorted(df['meal_id'].unique())
)

selected_center = st.sidebar.selectbox(
    "Select Fulfillment Center",
    sorted(df['center_id'].unique())
)

# ==========================================
# FILTER DATA
# ==========================================

filtered_df = df[
    (df['meal_id'] == selected_meal) &
    (df['center_id'] == selected_center)
]

# ==========================================
# KPI METRICS
# ==========================================

st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Orders",
        int(filtered_df['num_orders'].sum())
    )

with col2:
    st.metric(
        "Average Orders",
        round(filtered_df['num_orders'].mean(), 2)
    )

with col3:
    st.metric(
        "Maximum Orders",
        int(filtered_df['num_orders'].max())
    )

with col4:
    st.metric(
        "Average Checkout Price",
        round(filtered_df['checkout_price'].mean(), 2)
    )

# ==========================================
# WEEKLY DEMAND TREND
# ==========================================

st.subheader("📊 Weekly Demand Trend")

weekly_orders = filtered_df.groupby('week')['num_orders'].sum().reset_index()

fig1 = px.line(
    weekly_orders,
    x='week',
    y='num_orders',
    markers=True,
    title='Weekly Food Demand Trend'
)

fig1.update_layout(
    template='plotly_dark',
    xaxis_title='Week',
    yaxis_title='Orders'
)

st.plotly_chart(fig1, use_container_width=True)

# ==========================================
# TOP 10 MEALS
# ==========================================

st.subheader("🍽 Top 10 Meals")

top_meals = df.groupby('meal_id')['num_orders'] \
              .sum() \
              .sort_values(ascending=False) \
              .head(10) \
              .reset_index()

fig2 = px.bar(
    top_meals,
    x='meal_id',
    y='num_orders',
    color='num_orders',
    title='Top 10 Meals by Orders'
)

fig2.update_layout(
    template='plotly_dark'
)

st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# PRICE VS ORDERS
# ==========================================

st.subheader("💰 Checkout Price vs Orders")

fig3 = px.scatter(
    filtered_df,
    x='checkout_price',
    y='num_orders',
    color='homepage_featured',
    size='base_price',
    hover_data=['week'],
    title='Checkout Price vs Orders'
)

fig3.update_layout(
    template='plotly_dark'
)

st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

filtered_df = filtered_df.sort_values(by='week')

filtered_df['lag_1'] = filtered_df['num_orders'].shift(1)

filtered_df['lag_2'] = filtered_df['num_orders'].shift(2)

filtered_df['rolling_mean_3'] = filtered_df['num_orders'] \
                                .shift(1) \
                                .rolling(3) \
                                .mean()

filtered_df.dropna(inplace=True)

# ==========================================
# MODEL TRAINING
# ==========================================

features = [
    'checkout_price',
    'base_price',
    'emailer_for_promotion',
    'homepage_featured',
    'lag_1',
    'lag_2',
    'rolling_mean_3'
]

X = filtered_df[features]

y = filtered_df['num_orders']

split_index = int(len(filtered_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ==========================================
# XGBOOST MODEL
# ==========================================

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

st.subheader("🤖 Model Performance")

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

metric1, metric2 = st.columns(2)

with metric1:
    st.metric("MAE", round(mae, 2))

with metric2:
    st.metric("RMSE", round(rmse, 2))

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

st.subheader("📉 Actual vs Predicted Orders")

forecast_df = pd.DataFrame({
    'Actual Orders': y_test.values,
    'Predicted Orders': predictions
})

fig4 = go.Figure()

fig4.add_trace(
    go.Scatter(
        y=forecast_df['Actual Orders'],
        mode='lines',
        name='Actual Orders'
    )
)

fig4.add_trace(
    go.Scatter(
        y=forecast_df['Predicted Orders'],
        mode='lines',
        name='Predicted Orders'
    )
)

fig4.update_layout(
    title='Actual vs Predicted Food Demand',
    template='plotly_dark',
    xaxis_title='Samples',
    yaxis_title='Orders'
)

st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("⭐ Feature Importance")

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

fig5 = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    title='Feature Importance Analysis'
)

fig5.update_layout(
    template='plotly_dark'
)

st.plotly_chart(fig5, use_container_width=True)

# ==========================================
# DATA PREVIEW
# ==========================================

st.subheader("📄 Dataset Preview")

st.dataframe(filtered_df.head(20))

# ==========================================
# DOWNLOAD BUTTON
# ==========================================

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇ Download Filtered Dataset",
    data=csv,
    file_name='filtered_food_demand.csv',
    mime='text/csv'
)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
### 🚀 Technologies Used
- Python
- Streamlit
- Plotly
- XGBoost
- Pandas
- Scikit-Learn

Made with ❤️ for AI Demand Forecasting & Inventory Optimization
""")