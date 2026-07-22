import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# ---- Page setup ----
st.set_page_config(page_title="Sales Forecast", page_icon="📈")
st.title("📈 Sales Forecasting")
st.write(
    "This app forecasts future monthly sales using a Prophet time series "
    "model, capturing both the overall growth trend and yearly seasonal "
    "patterns."
)

st.divider()

# ---- Data source choice ----
data_source = st.radio(
    "Choose a data source",
    ["Try the demo (Superstore sample data)", "Upload my own sales data"]
)

if data_source == "Upload my own sales data":
    st.info(
        "Upload a CSV with two columns: a **date** column and a **sales "
        "amount** column (any column names are fine — you'll map them below)."
    )

    sample = pd.DataFrame({'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                            'Sales': [120.50, 89.00, 210.75]})
    st.download_button(
        "📄 Download a sample CSV format",
        sample.to_csv(index=False),
        "sample_format.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Preview of your data:")
        st.dataframe(raw_df.head())

        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Which column is the date?", raw_df.columns)
        with col2:
            sales_col = st.selectbox("Which column is the sales amount?", raw_df.columns)

        # Clean sales column: strip currency symbols, commas, spaces before converting to numbers
        raw_df[sales_col] = (
            raw_df[sales_col]
            .astype(str)
            .str.replace(r'[^0-9.\-]', '', regex=True)
        )
        raw_df[sales_col] = pd.to_numeric(raw_df[sales_col], errors='coerce')

        # Prepare and aggregate the uploaded data
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')

        if raw_df[date_col].isna().all():
            st.error(
                "Couldn't read any valid dates from that column. Please "
                "check the date format and try again."
            )
            st.stop()

        raw_df = raw_df.dropna(subset=[date_col, sales_col])

        historical = (
            raw_df.groupby(pd.Grouper(key=date_col, freq='ME'))[sales_col]
            .sum()
            .reset_index()
        )
        historical.columns = ['ds', 'y']

        if len(historical) < 3:
            st.error(
                "Not enough historical data to forecast reliably. Please "
                "upload at least 3 months of sales history."
            )
            st.stop()
        elif len(historical) < 8:
            st.warning(
                "This dataset has very few months of history. Prophet "
                "generally needs at least ~12-24 months of data for a "
                "reliable yearly seasonality estimate — treat this forecast "
                "with extra caution."
            )

        with st.spinner("Training forecast model on your data..."):
            model = Prophet(yearly_seasonality=5)
            model.fit(historical)
    else:
        st.stop()  # wait for a file before continuing

else:
    # ---- Demo mode: use the pre-trained Superstore model ----
    with st.spinner("Loading demo model..."):
        model = joblib.load("sales_forecast_model.pkl")
        historical = pd.read_csv("historical_sales.csv")
        historical['ds'] = pd.to_datetime(historical['ds'])

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

# ---- Headline metrics ----
future_raw = forecast.tail(months_ahead)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
total_forecast = future_raw['yhat'].sum()

metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.metric(f"Total Predicted Sales (next {months_ahead} mo.)", f"${total_forecast:,.0f}")

last_period_actual = historical['y'].tail(months_ahead).sum()
if last_period_actual > 0:
    growth_pct = ((total_forecast - last_period_actual) / last_period_actual) * 100
    with metric_col2:
        st.metric(
            f"vs. Last {months_ahead} Month(s) (${last_period_actual:,.0f})",
            f"{growth_pct:+.1f}%"
        )

# ---- Forecast table ----
st.subheader(f"Next {months_ahead} Month(s) Forecast")
future_only = future_raw.copy()
future_only.columns = ['Month', 'Predicted Sales', 'Low Estimate', 'High Estimate']
future_only['Month'] = future_only['Month'].dt.strftime('%B %Y')
st.dataframe(
    future_only.set_index('Month').style.format("${:,.0f}"),
    use_container_width=True
)

csv_download = future_only.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast as CSV", csv_download, "sales_forecast.csv", "text/csv")

st.divider()

with st.expander("ℹ️ About this model & its limitations"):
    st.write(
        "This app uses Prophet, a time series forecasting library, to "
        "capture both the overall growth trend and recurring yearly "
        "seasonal patterns in sales data."
    )
    if data_source == "Try the demo (Superstore sample data)":
        st.write(
            "**Demo accuracy:** average error of about 20% (MAPE) on past "
            "months for the sample Superstore dataset, with an average "
            "miss of roughly $7,350 per month."
        )
    else:
        st.write(
            "This forecast was trained live on your uploaded data. Accuracy "
            "depends heavily on how much history you provided — at least "
            "12-24 months is recommended for a reliable yearly seasonal "
            "estimate."
        )
    st.write(
        "**General guidance:** forecasts like this are useful for "
        "**budgeting and planning ranges**, not precise financial "
        "guarantees — sharp one-off spikes (promotions, bulk orders) are "
        "harder for any time series model to predict exactly."
    )
    st.write(
        "This kind of forecasting model can be built directly from a "
        "business's own historical sales records — no external market data "
        "or large customer base required."
    )

st.caption("Built with Prophet & Streamlit • Sixth ML project 🚀")
