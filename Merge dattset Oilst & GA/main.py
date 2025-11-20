import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1️⃣ LOAD DATA
# -------------------------------------------------------------------

# Olist daily revenue/orders (you already created this earlier)
olist = pd.read_csv("C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Merge dattset Oilst & GA/daily_sales.csv", parse_dates=['order_date'])

# GA daily sessions (exported from BigQuery)
ga = pd.read_csv("C:/Users/karthikeyan/OneDrive/Desktop/Python/E-Commerce/Merge dattset Oilst & GA/Daily Session.csv", parse_dates=['date'])

print("Olist shape:", olist.shape)
print("GA shape:", ga.shape)

# -------------------------------------------------------------------
# 2️⃣ CLEAN + RENAME COLUMNS
# -------------------------------------------------------------------

# Convert column names to lowercase
olist.columns = olist.columns.str.lower()
ga.columns = ga.columns.str.lower()

# Example GA columns: date, sessions, pageviews, etc.
if "sessions" not in ga.columns:
    # If your GA export used another name, fix here
    session_cols = [c for c in ga.columns if "session" in c]
    if session_cols:
        ga.rename(columns={session_cols[0]: "sessions"}, inplace=True)

# -------------------------------------------------------------------
# 3️⃣ MERGE ON DATE
# -------------------------------------------------------------------

df = olist.merge(ga, left_on="order_date", right_on="date", how="left")

# -------------------------------------------------------------------
# 4️⃣ HANDLE MISSING VALUES
# -------------------------------------------------------------------

# If GA has no data for a date → set sessions = 0
df["sessions"] = df["sessions"].fillna(0)

# -------------------------------------------------------------------
# 5️⃣ CALCULATE CONVERSION RATE
# -------------------------------------------------------------------

# Avoid division by zero
df["conversion_rate"] = np.where(
    df["sessions"] > 0,
    df["orders"] / df["sessions"],
    0
)

# -------------------------------------------------------------------
# 6️⃣ SAVE MERGED DATA
# -------------------------------------------------------------------

df.to_csv("Olist Brazilian E-commerce Dataset (Highly Recommended).csv", index=False)

print("Merged file saved: data/cleaned/olist_ga_merged.csv")

# -------------------------------------------------------------------
# 7️⃣ QUICK INSIGHTS
# -------------------------------------------------------------------

print("\nBasic Stats:")
print(df.describe())

print("\nDate range:", df['order_date'].min(), "to", df['order_date'].max())

# -------------------------------------------------------------------
# 8️⃣ PLOT CONVERSION RATE
# -------------------------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(df['order_date'], df['conversion_rate'])
plt.title("Conversion Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Conversion Rate")
plt.grid(True)
plt.show()