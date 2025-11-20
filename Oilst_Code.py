# olist_complete_pipeline_fixed.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Optional: Prophet for forecasting
try:
    from prophet import Prophet
    prophet_available = True
except Exception:
    prophet_available = False
    print("Prophet not available — forecast will use simple moving average fallback.")

plt.rcParams.update({'figure.figsize':(10,5), 'font.size':11})

# ---------------------------
# CONFIG: update this if needed
# ---------------------------
# You can set RAW_DIR to the folder containing the CSVs, or leave as-is if using absolute paths below.
RAW_DIR = "data/raw/olist"
CLEAN_DIR = "data/cleaned"
os.makedirs(CLEAN_DIR, exist_ok=True)

# ---------------------------
# 1) LOAD FILES (safe parsing)
# ---------------------------
def load_olist(raw_dir=RAW_DIR):
    print("Loading CSVs from", raw_dir)
    # If you have absolute paths, replace these with your absolute paths.
    # Using dayfirst=True because your timestamps are in DD-MM-YYYY HH:MM format.
    orders_path = os.path.join(raw_dir, "C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Olist Brazilian E-commerce Dataset (Highly Recommended)/olist_orders_dataset.csv")
    order_items_path = os.path.join(raw_dir, "C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Olist Brazilian E-commerce Dataset (Highly Recommended)/olist_order_items_dataset.csv")
    customers_path = os.path.join(raw_dir, "C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Olist Brazilian E-commerce Dataset (Highly Recommended)/olist_customers_dataset.csv")
    payments_path = os.path.join(raw_dir, "C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Olist Brazilian E-commerce Dataset (Highly Recommended)/olist_order_payments_dataset.csv")
    products_path = os.path.join(raw_dir, "C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Olist Brazilian E-commerce Dataset (Highly Recommended)/olist_products_dataset.csv")
    reviews_path = os.path.join(raw_dir, "C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Olist Brazilian E-commerce Dataset (Highly Recommended)/olist_order_reviews_dataset.csv")
    sellers_path = os.path.join(raw_dir, "C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Olist Brazilian E-commerce Dataset (Highly Recommended)/olist_sellers_dataset.csv")
    cat_translation_path = os.path.join(raw_dir, "C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Olist Brazilian E-commerce Dataset (Highly Recommended)/product_category_name_translation.csv")

    # If your CSVs are in a different folder (your earlier absolute paths), either update RAW_DIR or set these to the absolute paths.
    # Use parse_dates + dayfirst to avoid the warning; still we'll coerce again below to be 100% safe.
    parse_dates = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    # read (if file not found, raise a helpful error)
    def safe_read(path, **kwargs):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file: {path}\nPlease update RAW_DIR or paths.")
        return pd.read_csv(path, **kwargs)

    orders = safe_read(orders_path, parse_dates=parse_dates, dayfirst=True, low_memory=False)
    order_items = safe_read(order_items_path, low_memory=False)
    customers = safe_read(customers_path, low_memory=False)
    payments = safe_read(payments_path, low_memory=False)
    products = safe_read(products_path, low_memory=False)
    reviews = safe_read(reviews_path, parse_dates=['review_creation_date','review_answer_timestamp'], dayfirst=True, low_memory=False)
    sellers = safe_read(sellers_path, low_memory=False)
    cat_translation = safe_read(cat_translation_path, low_memory=False)

    return {
        "orders": orders,
        "order_items": order_items,
        "customers": customers,
        "payments": payments,
        "products": products,
        "reviews": reviews,
        "sellers": sellers,
        "cat_translation": cat_translation
    }

ds = load_olist()

# quick shapes
for k,v in ds.items():
    print(k, v.shape)

# ---------------------------
# 2) BASIC CLEANING
# ---------------------------
orders = ds['orders'].drop_duplicates().copy()
order_items = ds['order_items'].drop_duplicates().copy()
customers = ds['customers'].drop_duplicates().copy()
payments = ds['payments'].drop_duplicates().copy()
products = ds['products'].drop_duplicates().copy()
reviews = ds['reviews'].drop_duplicates().copy()
cat_translation = ds['cat_translation'].drop_duplicates().copy()

# Fix column names lowercase for consistency
orders.columns = orders.columns.str.lower()
order_items.columns = order_items.columns.str.lower()
customers.columns = customers.columns.str.lower()
payments.columns = payments.columns.str.lower()
products.columns = products.columns.str.lower()
reviews.columns = reviews.columns.str.lower()
cat_translation.columns = cat_translation.columns.str.lower()

# Ensure numeric
payments['payment_value'] = pd.to_numeric(payments['payment_value'], errors='coerce')

# Make sure date columns are datetimelike - coerce errors -> NaT (so .dt works)
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
for col in date_cols:
    if col in orders.columns:
        orders[col] = pd.to_datetime(orders[col], dayfirst=True, errors='coerce')

# Basic NA checks
print("orders missing by column:\n", orders.isna().sum().sort_values(ascending=False).head(20))

# ---------------------------
# 3) BUILD ORDERS_FULL (one-row per order)
# ---------------------------
# aggregate payments per order (some orders have multiple payment rows)
def mode_or_nan(series):
    modes = series.mode()
    return modes.iat[0] if not modes.empty else np.nan

payments_agg = payments.groupby('order_id', as_index=False).agg({
    'payment_value': 'sum',
    'payment_type': lambda x: mode_or_nan(x)
})

orders_full = orders.merge(payments_agg, on='order_id', how='left')
orders_full = orders_full.merge(customers, on='customer_id', how='left')

# create convenient columns
# safe: if order_purchase_timestamp is NaT, .dt.date will produce NaT -> we convert to pandas datetime date type where possible
orders_full['order_purchase_timestamp'] = pd.to_datetime(orders_full['order_purchase_timestamp'], dayfirst=True, errors='coerce')
orders_full['order_date'] = orders_full['order_purchase_timestamp'].dt.date

# filter out cancelled orders if desired:
if 'order_status' in orders_full.columns:
    orders_full = orders_full[orders_full['order_status'] != 'canceled'].copy()

# save cleaned orders_full
orders_full.to_parquet(os.path.join(CLEAN_DIR, "orders_full.parquet"), index=False)
print("orders_full rows:", orders_full.shape[0])

# ---------------------------
# 4) QUICK EDA METRICS
# ---------------------------
# Overall KPIs
total_revenue = orders_full['payment_value'].sum(min_count=1)  # min_count avoids sum -> 0 when all NaN
total_orders = orders_full['order_id'].nunique()
avg_order_value = (total_revenue / total_orders) if total_orders else np.nan
print(f"Total revenue: {total_revenue:.2f}, Total orders: {total_orders}, AOV: {avg_order_value:.2f}")

# Daily revenue series
daily = (orders_full.groupby('order_date', as_index=False)
         .agg(orders=('order_id','nunique'), revenue=('payment_value','sum')))
# convert order_date back to datetime (for plotting/time series)
daily['order_date'] = pd.to_datetime(daily['order_date'])
daily = daily.sort_values('order_date')
daily.to_csv(os.path.join(CLEAN_DIR, 'daily_sales.csv'), index=False)

# plot revenue trend (will skip if no display)
try:
    daily.set_index('order_date')['revenue'].plot(title='Daily Revenue (Olist)', ylabel='Revenue', xlabel='Date')
    plt.tight_layout()
    plt.show()
except Exception:
    pass

# ---------------------------
# 5) TOP PRODUCTS (join order_items & products)
# ---------------------------
items = order_items.copy()
items.columns = items.columns.str.lower()
products.columns = products.columns.str.lower()
if 'product_id' in items.columns and 'product_id' in products.columns:
    items = items.merge(products, on='product_id', how='left')

# revenue per line: use price; quantity not provided explicitly in olist_order_items_dataset - if you have quantity use it
if 'price' in items.columns:
    items['line_revenue'] = pd.to_numeric(items['price'], errors='coerce')
else:
    items['line_revenue'] = 0.0

prod_perf = items.groupby(['product_id','product_category_name'], dropna=False).agg(
    units_sold=('product_id','count'),
    revenue=('line_revenue','sum'),
    avg_price=('line_revenue','mean')
).reset_index().sort_values('revenue', ascending=False)

print("Top 10 product categories by revenue:")
print(prod_perf.groupby('product_category_name', dropna=False).agg({'revenue':'sum'}).sort_values('revenue', ascending=False).head(10))

# Top 20 products
print(prod_perf.sort_values('revenue', ascending=False).head(20))

# ---------------------------
# 6) DELIVERY PERFORMANCE (safe diffs)
# ---------------------------
# Ensure columns are datetimes (again, coerce to be safe)
for col in ['order_delivered_customer_date','order_estimated_delivery_date']:
    if col in orders_full.columns:
        orders_full[col] = pd.to_datetime(orders_full[col], dayfirst=True, errors='coerce')

# delivery_days = delivered_date - purchase_timestamp
orders_full['delivery_days'] = (orders_full['order_delivered_customer_date'] - orders_full['order_purchase_timestamp']).dt.days

# estimated_vs_actual_days = delivered_date - estimated_delivery_date
# Only meaningful where order_delivered_customer_date and order_estimated_delivery_date are present
orders_full['estimated_vs_actual_days'] = (orders_full['order_delivered_customer_date'] - orders_full['order_estimated_delivery_date']).dt.days

# summary
print("Delivery days summary:\n", orders_full['delivery_days'].describe())
late_pct = (orders_full['estimated_vs_actual_days'] > 0).mean()
print(f"Percent delivered after estimated date: {late_pct:.2%}")

# plot distribution (skip if no display)
try:
    orders_full['delivery_days'].dropna().hist(bins=30)
    plt.title("Distribution of delivery days")
    plt.xlabel("Days")
    plt.show()
except Exception:
    pass

# ---------------------------
# 7) REVIEWS: ratings and relation to delivery
# ---------------------------
# ensure date columns in reviews are datetime already (we parsed earlier)
reviews.columns = reviews.columns.str.lower()
if 'order_id' in reviews.columns:
    reviews_orders = reviews.merge(orders_full[['order_id','delivery_days','order_date','estimated_vs_actual_days']], on='order_id', how='left')
    print("Review score counts:")
    if 'review_score' in reviews_orders.columns:
        print(reviews_orders['review_score'].value_counts().sort_index())

    # average review by delivery lateness
    if 'estimated_vs_actual_days' in reviews_orders.columns and 'review_score' in reviews_orders.columns:
        reviews_orders['late'] = reviews_orders['estimated_vs_actual_days'] > 0
        print(reviews_orders.groupby('late')['review_score'].mean())
else:
    print("reviews dataframe does not contain 'order_id' - skipping review join")

# ---------------------------
# 8) RFM (customer segmentation)
# ---------------------------
# snapshot_date is one day after the last purchase timestamp we have
if 'order_purchase_timestamp' in orders_full.columns:
    snapshot_date = orders_full['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
else:
    snapshot_date = pd.Timestamp.now()

# group by customer_unique_id if available, else by customer_id
cust_col = 'customer_unique_id' if 'customer_unique_id' in orders_full.columns else 'customer_id'

rfm = orders_full.groupby(cust_col).agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'nunique',
    'payment_value': 'sum'
}).reset_index().rename(columns={
    'order_purchase_timestamp': 'recency_days',
    'order_id': 'frequency',
    'payment_value': 'monetary'
})

# handle missing / zero monetary
rfm = rfm[rfm['monetary'].notnull() & (rfm['monetary'] > 0)]

# RFM scoring (1-5). Use rank or qcut, but be robust to duplicates / small groups:
try:
    rfm['r_score'] = pd.qcut(rfm['recency_days'], 5, labels=[5,4,3,2,1]).astype(int)
except Exception:
    # fallback if qcut fails due to duplicate edges
    rfm['r_score'] = pd.cut(rfm['recency_days'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)

rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5]).astype(int)

rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
rfm['rfm_sum'] = rfm[['r_score','f_score','m_score']].sum(axis=1)

def seg_label(x):
    if x >= 13:
        return 'champion'
    if x >= 10:
        return 'loyal'
    if x >= 7:
        return 'needs_attention'
    return 'at_risk'

rfm['segment'] = rfm['rfm_sum'].apply(seg_label)

print("RFM segment sizes:")
print(rfm['segment'].value_counts())

rfm.to_csv(os.path.join(CLEAN_DIR,'rfm_segments.csv'), index=False)

# ---------------------------
# 9) JOIN DAILY OLIST (sales) with GA daily if you exported GA
# ---------------------------
ga_daily_path = "data/bq_exports/ga_daily.csv"
if os.path.exists(ga_daily_path):
    ga_daily = pd.read_csv(ga_daily_path, parse_dates=['date'])
    ga_daily.columns = [c.lower() for c in ga_daily.columns]
    daily_all = daily.merge(ga_daily, left_on='order_date', right_on='date', how='left')
    if 'sessions' in daily_all.columns:
        daily_all['conversion_rate'] = daily_all['orders'] / daily_all['sessions']
    daily_all.to_csv(os.path.join(CLEAN_DIR,'daily_with_ga.csv'), index=False)
    print("Merged daily with GA (if available).")
else:
    print("GA daily file not found in data/bq_exports/ — skip GA join.")

# ---------------------------
# 10) FORECAST: Prophet if available else moving average fallback
# ---------------------------
forecast_out_path = os.path.join(CLEAN_DIR,'sales_forecast.csv')
ts = daily[['order_date','revenue']].rename(columns={'order_date':'ds','revenue':'y'}).dropna()
if prophet_available and not ts.empty:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(ts)
    future = m.make_future_dataframe(periods=90)
    fc = m.predict(future)[['ds','yhat','yhat_lower','yhat_upper']]
    fc.to_csv(forecast_out_path, index=False)
    print("Prophet forecast saved:", forecast_out_path)
    try:
        m.plot(fc); plt.title("Prophet forecast of revenue"); plt.show()
    except Exception:
        pass
else:
    # simple 7-day moving avg forecast (naive)
    if ts.empty:
        print("No timeseries data available for forecasting.")
    else:
        ts = ts.set_index('ds').y
        rolling = ts.rolling(window=7, min_periods=1).mean()
        last_date = ts.index.max()
        future_dates = [last_date + timedelta(days=i) for i in range(1,91)]
        last_ma = rolling.iloc[-1]
        fc_df = pd.DataFrame({'ds':future_dates, 'yhat': last_ma})
        fc_df.to_csv(forecast_out_path, index=False)
        print("Fallback forecast saved:", forecast_out_path)

# ---------------------------
# 11) SAVE KEY OUTPUTS FOR DASHBOARD
# ---------------------------
daily.to_csv(os.path.join(CLEAN_DIR,'daily_sales.csv'), index=False)
prod_perf.to_csv(os.path.join(CLEAN_DIR,'product_performance.csv'), index=False)
orders_full.to_csv(os.path.join(CLEAN_DIR,'orders_full.csv'), index=False)
items.to_csv(os.path.join(CLEAN_DIR,'order_items_with_products.csv'), index=False)
print("Saved cleaned outputs to", CLEAN_DIR)

# ---------------------------
# 12) QUICK VISUALS (optional)
# ---------------------------
try:
    cat_rev = items.groupby('product_category_name').agg(revenue=('line_revenue','sum')).sort_values('revenue', ascending=False).head(10)
    cat_rev['revenue'].plot(kind='bar', title='Top 10 Categories by Revenue'); plt.ylabel('Revenue'); plt.show()
except Exception:
    pass

try:
    rfm['segment'].value_counts().plot(kind='bar', title='RFM segment counts'); plt.show()
except Exception:
    pass




