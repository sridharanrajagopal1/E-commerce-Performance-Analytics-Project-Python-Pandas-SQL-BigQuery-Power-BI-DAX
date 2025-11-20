# 🛒 E-commerce Analytics Dashboard (Olist + Google Analytics + Power BI)

### 🔍 End-to-End Data Analytics Project | Python • SQL • BigQuery • Power BI • DAX

This project analyzes an e-commerce business using **Olist Brazilian E-commerce Dataset** and **Google Analytics (BigQuery)** to understand:

- Sales performance  
- Customer behavior  
- Website traffic  
- Conversion funnel  
- Product & category performance  
- Delivery performance  
- Traffic → Sales relationship

An interactive **Power BI dashboard** was built to visualize insights and drive data-based decisions.

---

## 📁 Project Structure

Ecommerce_Analytics_Project/
├── data/
│ ├── raw/ # Original Olist CSV files
│ ├── cleaned/ # Processed datasets (orders_full, daily_sales, merged)
│ ├── bq_exports/ # Data exported from Google BigQuery (GA)
├── notebooks/
│ ├── 01_olist_cleaning.ipynb
│ ├── 02_ga_analysis.ipynb
│ └── 03_merge_olist_ga.ipynb
├── sql/
│ ├── ga_daily.sql
│ ├── ga_funnel.sql
│ └── olist_queries.sql
├── dashboards/
│ └── Ecommerce_Dashboard.pbix
├── reports/
│ └── Final_Insights.pdf
└── README.md

markdown
Copy code

---

## 🗂️ Datasets Used

### **1. Olist E-commerce Dataset (Kaggle)**
Contains:
- Orders  
- Products  
- Customers  
- Sellers  
- Payments  
- Reviews  
- Order Items  

👉 https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

### **2. Google Analytics Sample Dataset (BigQuery)**  
Used for:
- Sessions  
- Traffic sources  
- Device usage  
- Funnel events  
- Product views  

Dataset path:  
bigquery-public-data.google_analytics_sample

pgsql
Copy code

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python (Pandas)** | Data cleaning, merging, RFM, EDA |
| **SQL (BigQuery)** | GA analysis, traffic, funnel, sessions |
| **Power BI** | Final dashboard visualization |
| **DAX** | KPI measures (Revenue, AOV, Conversion rate) |
| **Jupyter/Colab** | Exploratory analysis |
| **Matplotlib** | Visualizations |

---

## 📊 Power BI Dashboard

### 🔹 **Page 1: Sales Overview (Olist)**
- Total Revenue, Orders, AOV  
- Monthly Revenue Trend  
- Revenue by State  
- Sales by Category  

### 🔹 **Page 2: Product & Customer Analysis**
- Top 10 Products  
- RFM Segmentation  
- Customer Lifetime Value  
- Top cities & states  

### 🔹 **Page 3: Website Analytics (GA)**
- Daily Sessions  
- Device Breakdown  
- Traffic Sources  
- Bounce & Engagement Metrics  

### 🔹 **Page 4: Conversion Analysis (Olist + GA)**
- Sessions vs Revenue  
- Conversion Rate Trend  
- Funnel: View → Add to Cart → Checkout → Purchase  

---

## 📈 Key Insights

- Product categories like **bed_bath_table** and **health_beauty** contribute most revenue.  
- Delivery delays strongly affect customer **review_score**, reducing repeat purchase likelihood.  
- Traffic spikes from **Organic Search** and **Referral** channels align with higher order volume.  
- Overall **conversion rate = orders / sessions** reveals days of high traffic but low sales, indicating usability or pricing issues.  

---

## 🧪 SQL Queries (BigQuery)

### Daily Sessions
```sql
SELECT
  date,
  COUNT(*) AS total_sessions
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
GROUP BY date
ORDER BY date;
Funnel Events
sql
Copy code
SELECT
  hits.eventInfo.eventAction AS action,
  COUNT(*) AS events
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
UNNEST(hits) AS hits
WHERE hits.eventInfo.eventAction IN ('view_item','add_to_cart','checkout','purchase')
GROUP BY action
ORDER BY events DESC;
Revenue (Olist)
sql
Copy code
SELECT
  DATE(order_purchase_timestamp) AS date,
  SUM(payment_value) AS revenue
FROM olist_orders_full
GROUP BY date
ORDER BY date;

Python (Merge Olist + GA)
python
Copy code
import pandas as pd
import numpy as np

olist = pd.read_csv("data/cleaned/daily_sales.csv", parse_dates=['order_date'])
ga = pd.read_csv("data/bq_exports/ga_daily.csv", parse_dates=['date'])

df = olist.merge(ga, left_on='order_date', right_on='date', how='left')
df['sessions'] = df['sessions'].fillna(0)
df['conversion_rate'] = np.where(df['sessions'] > 0, df['orders']/df['sessions'], 0)
df.to_csv("data/cleaned/olist_ga_merged.csv", index=False)

Project Workflow
Data Extraction
Download Olist data
Query GA data from BigQuery
Data Cleaning (Python)
Remove duplicates
Standardize date formats
Join orders, customers, payments
Create product performance metrics
RFM segmentation
Google Analytics Analysis (SQL)
Sessions
Device
Traffic sources
Funnel analysis
Sales + Traffic Merge
Merge Olist daily revenue with GA daily sessions
Compute conversion rate
Visualization (Power BI)
Build 4-page dashboard
KPI cards, funnel, maps, bar/line charts
Insights & Recommendations
Provide actionable business insights

📘 Final Deliverables
✔ Cleaned Olist dataset
✔ GA query outputs
✔ Full Python pipeline
✔ Power BI dashboard (.pbix)
✔ SQL scripts
✔ Project report

👨‍💻 Author
Sridharan
Data Analyst | Business Intelligence | SQL | Python
📧 sridharanrajagopal@yahoo.com
🔗 Portfolio: https://portfolio-demo-sridharanrajagopal1s-projects.vercel.app/
🔗 LinkedIn: https://www.linkedin.com/in/sridharan-rajagopal/
