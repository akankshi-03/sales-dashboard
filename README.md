# 📊 Sales & Revenue Analytics Dashboard

🔗 **Live Demo:** https://sales-dashboard-ggrjqmubqzuwmtumgzgzry.streamlit.app

An end-to-end **Sales Analytics & Forecasting Dashboard** built using **Python and Streamlit**.
This project helps businesses analyze historical sales data, monitor KPIs in real-time, and predict future revenue using machine learning.

---

## 🚀 Key Features

* 📈 **Revenue Trend Analysis**
  Analyze 24 months of historical sales data with moving averages

* 🔮 **Sales Forecasting (ML-based)**
  Predict next 6 months of revenue using Polynomial Regression

* 🌍 **Regional Performance Analysis**
  Compare sales across regions (North, South, East, West)

* 🛍️ **Product Category Insights**
  Visualize revenue distribution using interactive donut charts

* 💹 **Profitability Tracking**
  Track gross margin (%) and return rate trends over time

* 🤖 **Automated Business Insights**
  AI-generated recommendations for better decision-making

* 🌐 **Live Data Integration**
  Real-time USD/INR exchange rates via API

* 🎛️ **Interactive Dashboard Controls**
  Dynamic filters for date range, region, and forecasting toggle

---

## 🛠️ Tech Stack

* **Python 3.12**
* **Streamlit** – Web app framework
* **Plotly** – Interactive data visualizations
* **Pandas & NumPy** – Data processing and analysis
* **Scikit-learn** – Machine learning (forecasting model)
* **Requests** – API integration

---

## 📊 Dashboard Overview

### 🔹 KPI Metrics

* Total Revenue
* Total Orders
* Average Order Value
* Gross Margin (%)
* 6-Month Forecast

### 🔹 Visualizations

* Revenue Trend + Forecast (with confidence band)
* Category-wise Revenue Distribution
* Regional Sales Comparison
* Margin vs Return Rate Trend
* Forecast Projection Table

---

## 🤖 Machine Learning Forecasting

The forecasting module uses **Polynomial Regression (degree 2)**:

* Trained on historical monthly revenue data
* Captures non-linear growth trends
* Predicts future revenue for selected months
* Displays a ±8% confidence interval for reliability

---

## 🌐 Live API Integration

* **Exchange Rate API** → Real-time USD to INR conversion
* **Yahoo Finance** → Gold price (used as external market indicator)

All revenue values are dynamically converted and displayed in INR (₹).

---

## 🧠 Auto-Generated Business Insights

The dashboard provides actionable insights such as:

* Identification of best and worst performing months
* Fastest growing product categories
* High return rate alerts
* Forecasted peak demand periods
* Profit margin improvement suggestions

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/sales-dashboard.git
cd sales-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

Open in browser: http://localhost:8501

---

## 🚀 Deployment

Deployed using **Streamlit Community Cloud**:

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set `app.py` as entry point
4. Deploy instantly

---

## 👩‍💻 About the Project

This project demonstrates:

* Data Cleaning & Analysis
* Business Intelligence & KPI Tracking
* Machine Learning for Forecasting
* Dashboard Development & Deployment
* API Integration & Real-time Data Handling

---

## ⭐ Author

**Akanksha Dubey**
AI/ML & Data Analytics Enthusiast

---

## 📌 Why This Project Matters

This dashboard showcases how raw data can be transformed into **actionable business insights**, enabling organizations to make **data-driven decisions** and plan future strategies effectively.

---

Machine Learning — Regression & Forecasting
Dashboard Development & Cloud Deployment
REST API Integration
Business Intelligence & Insights Generation

