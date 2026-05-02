📊 Sales & Revenue Analytics Dashboard

Live Demo: sales-dashboard-ggrjqmubqzuwmtumgzgzry.streamlit.app

An end-to-end Sales Analytics & Forecasting System built with Python and Streamlit. This dashboard helps businesses analyze past sales trends, track KPIs in real time, and predict future revenue using Machine Learning.

🚀 Features
FeatureDescription📈 Revenue Trend Analysis24-month historical sales with moving average🔮 AI ForecastingPolynomial Regression model predicts next 6 months🌍 Regional BreakdownNorth / South / East / West sales comparison🛍️ Category AnalysisRevenue share by product category (Donut chart)💹 Margin TrackingGross margin % and return rate over time🤖 Auto InsightsAI-generated actionable business recommendations🌐 Live API DataReal-time USD/INR exchange rate integration🎛️ Interactive FiltersDate range slider, region filter, forecast toggle

🛠️ Tech Stack
Python 3.12
├── Streamlit       → Web dashboard UI
├── Plotly          → Interactive charts
├── Pandas          → Data cleaning & manipulation
├── NumPy           → Numerical computations
├── Scikit-learn    → ML forecasting (Polynomial Regression)
└── Requests        → Live API integration (ExchangeRate API)

📁 Project Structure
sales-dashboard/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← Project documentation

⚙️ Run Locally
bash# 1. Clone the repository
git clone https://github.com/yourusername/sales-dashboard.git
cd sales-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python -m streamlit run app.py

# 4. Open in browser
# http://localhost:8501

🤖 ML Forecasting — How It Works
The forecasting module uses Polynomial Regression (degree 2) from Scikit-learn:

Historical monthly revenue data is used as training data
A polynomial feature transformer captures non-linear growth trends
The model predicts revenue for the next N months (configurable via sidebar)
A ±8% confidence interval is calculated and visualized as a band

pythonfrom sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
model = LinearRegression()
model.fit(poly.fit_transform(X_train), y_train)
predictions = model.predict(poly.transform(X_future))

🌐 Live API Integration
The dashboard fetches real-time data from:

ExchangeRate API → Live USD/INR conversion rate (updates hourly)
Yahoo Finance → Gold commodity price (used as market indicator)

All revenue figures are displayed in Indian Rupees (₹) using the live exchange rate.

📊 Dashboard Sections
KPI Metrics Row
Total Revenue · Total Orders · Avg Order Value · Gross Margin · 6-Month Forecast
Charts

Revenue + Forecast → Bar chart (actual) + Line (forecast + confidence band)
Category Donut → Revenue distribution by product category
Region Bar Chart → Filterable regional sales comparison
Margin Trend → Dual-axis: Gross Margin % vs Return Rate %
Forecast Table → Conservative / Expected / Optimistic projections


🤖 Auto Business Insights
The dashboard generates insights like:

Best/worst performing months with inventory recommendations
Fastest growing product categories with ad spend suggestions
Return rate alerts with corrective action tips
Forecast peak month for pre-stocking planning
Margin optimization opportunities


🚀 Deployment
Deployed on Streamlit Community Cloud (free):

Push code to GitHub
Connect repo at share.streamlit.io
Set main file as app.py
Deploy — live in under 2 minutes


👩‍💻 About
Built by Akanksha as part of an AI/ML portfolio project.
Skills demonstrated:

Data Analysis & Visualization
Machine Learning — Regression & Forecasting
Dashboard Development & Cloud Deployment
REST API Integration
Business Intelligence & Insights Generation

