import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# ---- Page setup ----
st.set_page_config(page_title="Sales Forecast", page_icon="📈")
st.title("📈 Sales Forecasting")
st.write(
    "This app forecasts future sales — daily, weekly, or monthly — using a "
    "Prophet time series model that captures the overall growth trend and "
    "seasonal patterns."
)

st.divider()

FREQ_MAP = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
UNIT_LABEL = {"Daily": "day", "Weekly": "week", "Monthly": "month"}
DEFAULT_HORIZON = {"Daily": 14, "Weekly": 8, "Monthly": 6}
MAX_HORIZON = {"Daily": 90, "Weekly": 26, "Monthly": 12}

# ---- Data source choice ----
data_source = st.radio(
    "Choose a data source",
    ["Try the demo (Superstore sample data, monthly)", "Upload my own sales data"]
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

        granularity = st.selectbox(
            "Forecast granularity — how do you want outcomes broken down?",
            ["Daily", "Weekly", "Monthly"],
            index=2,
            help="Daily needs the most history to be reliable; Monthly needs the least."
        )
        freq = FREQ_MAP[granularity]
        unit = UNIT_LABEL[granularity]

        # Clean sales column: strip currency symbols, commas, spaces before converting to numbers
        raw_df[sales_col] = (
            raw_df[sales_col]
            .astype(str)
            .str.replace(r'[^0-9.\-]', '', regex=True)
        )
        raw_df[sales_col] = pd.to_numeric(raw_df[sales_col], errors='coerce')

        # Prepare dates
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')

        if raw_df[date_col].isna().all():
            st.error(
                "Couldn't read any valid dates from that column. Please "
                "check the date format and try again."
            )
            st.stop()

        raw_df = raw_df.dropna(subset=[date_col, sales_col])

        historical = (
            raw_df.groupby(pd.Grouper(key=date_col, freq=freq))[sales_col]
            .sum()
            .reset_index()
        )
        historical.columns = ['ds', 'y']

        if len(historical) < 3:
            st.error(
                f"Not enough historical data to forecast reliably. Please "
                f"upload at least 3 {unit}s of sales history."
            )
            st.stop()

        # Decide seasonality settings based on the actual calendar span covered,
        # not just the number of aggregated rows — this matters a lot for daily/weekly data
        span_days = (historical['ds'].max() - historical['ds'].min()).days

        use_yearly = span_days >= 730  # need ~2 years to trust a yearly cycle
        use_weekly = (freq == "D") and (span_days >= 60)  # daily data, at least ~2 months

        if not use_yearly:
            st.warning(
                f"This dataset spans about {span_days // 30} month(s), which "
                "isn't enough to reliably detect a *yearly* repeating pattern "
                "(e.g. holiday spikes). To avoid inventing a false seasonal "
                "pattern, this forecast will model the overall **trend** "
                f"{'and weekly pattern ' if use_weekly else ''}only — treat "
                "longer-range predictions with extra caution."
            )

        # Cap the forecast horizon to a reasonable fraction of available history
        max_reasonable_horizon = max(1, len(historical) // 2)

        with st.spinner("Training forecast model on your data..."):
            model = Prophet(
                yearly_seasonality=use_yearly,
                weekly_seasonality=use_weekly,
                daily_seasonality=False
            )
            model.fit(historical)
    else:
        st.stop()  # wait for a file before continuing

else:
    # ---- Demo mode: use the pre-trained Superstore model (monthly only) ----
    with st.spinner("Loading demo model..."):
        model = joblib.load("sales_forecast_model.pkl")
        historical = pd.read_csv("historical_sales.csv")
        historical['ds'] = pd.to_datetime(historical['ds'])
    granularity = "Monthly"
    unit = "month"
    freq = "ME"
    max_reasonable_horizon = 12

st.divider()

# ---- User controls ----
slider_max = min(MAX_HORIZON[granularity], max_reasonable_horizon)
if slider_max < 1:
    slider_max = 1
default_periods = min(DEFAULT_HORIZON[granularity], slider_max)
periods_ahead = st.slider(
    f"How many {unit}s ahead to forecast?", 1, slider_max, default_periods
)

if slider_max < MAX_HORIZON[granularity]:
    st.caption(
        f"⚠️ Forecast horizon is capped at {slider_max} {unit}(s) because "
        "predicting much further than half your available history becomes "
        "unreliable."
    )

future = model.make_future_dataframe(periods=periods_ahead, freq=freq)
forecast = model.predict(future)

# Sales can't realistically be negative — clip any dipped-below-zero predictions for display
for col in ['yhat', 'yhat_lower', 'yhat_upper']:
    forecast[col] = forecast[col].clip(lower=0)

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
future_raw = forecast.tail(periods_ahead)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
total_forecast = future_raw['yhat'].sum()

metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.metric(f"Total Predicted Sales (next {periods_ahead} {unit}(s))", f"${total_forecast:,.0f}")

last_period_actual = historical['y'].tail(periods_ahead).sum()
if last_period_actual > 0:
    growth_pct = ((total_forecast - last_period_actual) / last_period_actual) * 100
    with metric_col2:
        st.metric(
            f"vs. Last {periods_ahead} {unit}(s) (${last_period_actual:,.0f})",
            f"{growth_pct:+.1f}%"
        )

# ---- Forecast table ----
st.subheader(f"Next {periods_ahead} {unit.capitalize()}(s) Forecast")
future_only = future_raw.copy()
future_only.columns = ['Period', 'Predicted Sales', 'Low Estimate', 'High Estimate']
date_fmt = '%B %Y' if granularity == "Monthly" else '%b %d, %Y'
future_only['Period'] = future_only['Period'].dt.strftime(date_fmt)
st.dataframe(
    future_only.set_index('Period').style.format("${:,.0f}"),
    use_container_width=True
)

csv_download = future_only.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast as CSV", csv_download, "sales_forecast.csv", "text/csv")

st.divider()

with st.expander("ℹ️ About this model & its limitations"):
    st.write(
        "This app uses Prophet, a time series forecasting library, to "
        "capture the overall growth trend and recurring seasonal patterns "
        "in sales data — at a **daily, weekly, or monthly** granularity of "
        "your choice."
    )
    if data_source.startswith("Try the demo"):
        st.write(
            "**Demo accuracy:** average error of about 20% (MAPE) on past "
            "months for the sample Superstore dataset, with an average "
            "miss of roughly $7,350 per month."
        )
    else:
        st.write(
            "This forecast was trained live on your uploaded data. Accuracy "
            "depends heavily on how much history you provided, and finer "
            "granularities (daily/weekly) generally need more historical "
            "data than monthly to produce reliable seasonal patterns."
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
