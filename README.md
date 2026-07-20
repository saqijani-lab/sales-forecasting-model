# 📈 Sales Forecasting

A time series forecasting project that predicts future monthly sales using Prophet, capturing both long-term growth trends and recurring yearly seasonality. My sixth ML project and first time-series model.

🔗 Live App: https://sales-forecasting-model-007.streamlit.app/ 

---

## 📌 Overview

Unlike my previous projects, this one predicts values "forward in time" rather than classifying or scoring a single input. The goal: help a retail business anticipate future monthly revenue for budgeting, inventory planning, and staffing decisions — a genuinely common freelance request for small and medium businesses.

📊 Dataset

**Sample Superstore Dataset** (Kaggle) — ~10,000 retail order records spanning 2014-2018, aggregated into monthly sales totals for forecasting.

🧠 Model

- **Algorithm:** Prophet (Meta/Facebook's time series forecasting library)
- **Input:** Monthly aggregated sales (`ds` = date, `y` = total sales)
- **Seasonality:** Yearly seasonality tuned down from Prophet's default (Fourier order 10 → 5) after the default setting produced an implausibly wiggly, overfit seasonal curve given only 4 years of monthly data

📈 What the Model Found

- **Clear upward growth trend**: baseline sales grew from ~$34K to ~$66K/month over the historical period
- **Recurring yearly seasonality**: a dip around March-May, and a peak around November-December — consistent with typical retail holiday-season patterns

🎯 Accuracy

| Metric | Value |
|---|---|
| MAE | $7,352 |
| MAPE | 20.05% |

**Honest interpretation:** A ~20% average error means this model is useful for **budgeting and planning ranges** (e.g., "expect $45K-$65K next month"), not for precise financial commitments. Sharp one-off spikes (promotions, large bulk orders) are inherently harder for any time series model to predict exactly.

🖥️ App Features

- Adjustable forecast horizon (1-12 months ahead)
- Combined chart of actual historical sales + forecast + confidence interval
- Forecast table with predicted, low, and high estimates per month
- Transparent discussion of model accuracy and limitations

🛠️ Tech Stack

- Python, pandas
- Prophet (time series forecasting)
- joblib (model persistence)
- Streamlit (UI & deployment)
- Matplotlib (visualization)

🚀 Run Locally

```bash
git clone https://github.com/saqijani-lab/sales-forecasting.git
cd sales-forecasting
pip install -r requirements.txt
streamlit run app.py
```

⚠️ Limitations

- Trained on only 4 years of monthly data (48 points) — seasonality patterns are estimated from limited history and may not generalize to major business changes (new product lines, market shifts, economic shocks).
- Cannot predict one-off events like promotions, viral products, or supply disruptions.
- Forecasts should be treated as planning ranges, not guaranteed figures.

📁 Project Structure

```
sales-forecasting/
├── app.py                    # Streamlit app
├── sales_forecast_model.pkl   # Trained Prophet model
├── historical_sales.csv       # Aggregated monthly sales used for training
├── requirements.txt            # Python dependencies
└── README.md
```

---

*This is my sixth end-to-end machine learning project, and my first time series forecasting model — completing a portfolio spanning regression, classification, clustering, and forecasting.*
