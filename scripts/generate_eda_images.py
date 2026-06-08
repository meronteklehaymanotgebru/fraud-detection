#!/usr/bin/env python3
"""Generate key EDA figures for the Interim-1 report."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipaddress
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
OUT_DIR = "reports/images"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# Load e-commerce data
# -------------------------------
fraud = pd.read_csv("data/raw/Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])
ip_map = pd.read_csv("data/raw/IpAddress_to_Country.csv")

# Geolocation merge (same robust conversion as in notebook)
def ip_to_int(ip_val):
    if isinstance(ip_val, str):
        try:
            return int(ipaddress.ip_address(ip_val))
        except ValueError:
            pass
    try:
        return int(float(ip_val))
    except (ValueError, TypeError):
        return None

fraud["ip_int"] = fraud["ip_address"].apply(ip_to_int)
fraud.dropna(subset=["ip_int"], inplace=True)
fraud["ip_int"] = fraud["ip_int"].astype("int64")
ip_map["lower_bound_ip_address"] = ip_map["lower_bound_ip_address"].astype("int64")

fraud.sort_values("ip_int", inplace=True)
ip_map.sort_values("lower_bound_ip_address", inplace=True)
fraud = pd.merge_asof(
    fraud,
    ip_map[["lower_bound_ip_address", "country"]],
    left_on="ip_int",
    right_on="lower_bound_ip_address",
    direction="backward",
)
fraud.drop(columns=["ip_int", "lower_bound_ip_address"], inplace=True)

# Time features for analysis
fraud["hour"] = fraud["purchase_time"].dt.hour

# -------------------------------
# Figure 1: Class distribution
# -------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.countplot(x="class", data=fraud, palette="Set2", ax=ax1)
ax1.set_title("Class Distribution (0 = Legitimate, 1 = Fraud)", fontsize=14)
ax1.set_xlabel("Class")
ax1.set_ylabel("Count")
for p in ax1.patches:
    ax1.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha="center", va="bottom")
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig1)

# -------------------------------
# Figure 2: Fraud rate by hour of day
# -------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
hourly_fraud = fraud.groupby("hour")["class"].mean()
sns.barplot(x=hourly_fraud.index, y=hourly_fraud.values, palette="coolwarm", ax=ax2)
ax2.set_title("Fraud Rate by Hour of Day", fontsize=14)
ax2.set_xlabel("Hour")
ax2.set_ylabel("Fraud Rate")
ax2.set_xticks(range(0, 24))
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fraud_by_hour.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)

# -------------------------------
# Figure 3: Top 10 countries by fraud rate
# -------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))
country_fraud = fraud.groupby("country")["class"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=country_fraud.values, y=country_fraud.index, palette="rocket", ax=ax3)
ax3.set_title("Top 10 Countries by Fraud Rate", fontsize=14)
ax3.set_xlabel("Fraud Rate")
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "fraud_by_country.png"), dpi=150, bbox_inches="tight")
plt.close(fig3)

# -------------------------------
# Figure 4: Fraud rate by browser
# -------------------------------
fig4, ax4 = plt.subplots(figsize=(8, 5))
browser_fraud = fraud.groupby("browser")["class"].mean().sort_values(ascending=False)
sns.barplot(x=browser_fraud.values, y=browser_fraud.index, palette="viridis", ax=ax4)
ax4.set_title("Fraud Rate by Browser", fontsize=14)
ax4.set_xlabel("Fraud Rate")
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "fraud_by_browser.png"), dpi=150, bbox_inches="tight")
plt.close(fig4)

# -------------------------------
# Figure 5: Purchase value distribution
# -------------------------------
fig5, ax5 = plt.subplots(figsize=(8, 5))
fraud["purchase_value"].hist(bins=50, ax=ax5, color="steelblue", edgecolor="black")
ax5.set_title("Distribution of Purchase Value", fontsize=14)
ax5.set_xlabel("Purchase Value")
ax5.set_ylabel("Frequency")
fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, "purchase_value_dist.png"), dpi=150, bbox_inches="tight")
plt.close(fig5)

print(f"Images saved in {OUT_DIR}")