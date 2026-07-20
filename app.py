import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---- Load model and historical data ----
model = joblib.load("sales_forecast_model.pkl")
historical = pd.read_csv("historical_sales.csv")
historical['ds'] = pd.to_datetime(historical['ds'])

# ---- Page setup ----
st.set_page_config(page_title="Sales Forecast", page_icon="📈")
st.title("📈 Sales Forecasting")
st.write(
    "This app forecasts future monthly sales using a Prophet time series "
    "model trained on historical order data, capturing both the overall "
    "growth trend and yearly seasonal patterns."
)

st.divider()

# ---- User controls ----
months_ahead = st.slider("How many months ahead to forecast?", 1, 12, 6)

future = model.make_future_dataframe(periods=months_ahead, freq='ME')
forecast = model.predict(future)

# ---- Forecast chart ----
st.subheader("Forecast")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(historical['ds'], historical['y'], 'k.', label='Actual Sales')
ax.plot(forecast['ds'], forecast['yhat'], color='tab:blue', label='Forecast')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                 color='tab:blue', alpha=0.2, label='Confidence Range')
ax.set_xlabel("Date")
ax.set_ylabel("Sales ($)")
ax.legend()
st.pyplot(fig)

# ---- Forecast table ----
st.subheader(f"Next {months_ahead} Month(s) Forecast")
future_only = forecast.tail(months_ahead)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
future_only.columns = ['Month', 'Predicted Sales', 'Low Estimate', 'High Estimate']
future_only['Month'] = future_only['Month'].dt.strftime('%B %Y')
st.dataframe(
    future_only.set_index('Month').style.format("${:,.0f}"),
    use_container_width=True
)

st.divider()

with st.expander("ℹ️ About this model & its limitations"):
    st.write(
        "This model is a Prophet time series forecast trained on aggregated "
        "monthly sales data. It captures a clear upward growth trend and a "
        "recurring yearly seasonal pattern (typically a mid-year dip and a "
        "holiday-season peak)."
    )
    st.write(
        "**Historical accuracy:** average error of about 20% (MAPE) on past "
        "months, with an average miss of roughly $7,350 per month. This "
        "makes the forecast useful for **budgeting and planning ranges**, "
        "but it should not be treated as a precise financial guarantee — "
        "sharp one-off spikes (promotions, bulk orders) are harder for the "
        "model to predict exactly."
    )
    st.write(
        "Unlike collaborative or purchase-history-based approaches, this "
        "kind of forecasting model can be built directly from a business's "
        "own historical sales records — no external data required."
    )

st.caption("Built with Prophet & Streamlit • Sixth ML project 🚀")
