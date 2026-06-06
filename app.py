
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="Vendor Revenue Analysis", layout="wide")

# Global config
PALETTE = ["#1B4F72", "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
           "#3B1F2B", "#44BBA4", "#E94F37", "#393E41", "#F5A623"]
BG_COLOR = "#F7F9FC"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.grid": True,
    "grid.color": "#DDE3EC",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CITY_MARKETS = {
    "Buea": {"markets": ["Mile 17 Motor Park Market", "Molyko Market", "Sandpit Market"], "base_revenue": 18000, "n_vendors": 12},
    "Yaoundé": {"markets": ["Marché Mokolo", "Marché du Mfoundi", "Marché Essos", "Marché Mvog-Mbi"], "base_revenue": 32000, "n_vendors": 14},
    "Douala": {"markets": ["Marché Central (New Bell)", "Marché Sandaga", "Marché Nkoulouloun", "Marché Deido", "Marché Congo"], "base_revenue": 42000, "n_vendors": 14},
}

PRODUCT_CATEGORIES = [
    "Plantain & Cassava",
    "Fresh Vegetables",
    "Dried Fish & Crayfish",
    "Palm Oil & Condiments",
    "Fruits (Mango/Avocado/Pineapple)",
    "Second-Hand Clothing (Friperie)",
    "Mobile Phone Accessories",
    "Cosmetics & Hair Products",
    "Cooked Street Food (Soya/Beignets)",
    "Charcoal & Firewood",
]

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = ["September", "October", "November", "December", "January", "February", "March"]

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("cameroon_vendor_data.csv", parse_dates=["date"])
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Clustering", "Revenue Prediction", "Association Rules"])

# Home page
if page == "Home":
    st.title("🏪 Vendor Revenue Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes street vendor transaction patterns in Cameroon to predict daily revenue.
    
    **Data Overview:**
    - 8,480 daily transaction records
    - 40 vendors across 3 cities (Buea, Yaoundé, Douala)
    - Time period: September 2025 - March 2026
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Vendors", df["vendor_id"].nunique())
    with col3:
        st.metric("Avg Daily Revenue", f"{df['daily_revenue_xaf'].mean():,.0f} FCFA")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10), width='stretch')

# EDA page
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    # Plot 1: Revenue distribution by city
    st.subheader("Revenue Distribution by City")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, city, color in zip(axes, CITY_MARKETS.keys(), PALETTE):
        subset = df[df["city"] == city]["daily_revenue_xaf"] / 1000
        ax.hist(subset, bins=40, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(subset.mean(), color="red", linestyle="--", lw=1.8, label=f"Mean: {subset.mean():.1f}K")
        ax.axvline(subset.median(), color="orange", linestyle=":", lw=1.8, label=f"Median: {subset.median():.1f}K")
        ax.set_title(city, fontweight="bold", fontsize=12)
        ax.set_xlabel("Daily Revenue (000 XAF)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Plot 2: Day of week pattern
    st.subheader("Average Daily Revenue by Day of Week")
    dow_city = df.groupby(["city", "day_of_week"])["daily_revenue_xaf"].mean().reset_index()
    dow_city["day_of_week"] = pd.Categorical(dow_city["day_of_week"], categories=DOW_ORDER, ordered=True)
    dow_city = dow_city.sort_values("day_of_week")
    
    fig, ax = plt.subplots(figsize=(13, 5))
    for city, color in zip(CITY_MARKETS.keys(), PALETTE):
        sub = dow_city[dow_city["city"] == city]
        ax.plot(sub["day_of_week"], sub["daily_revenue_xaf"] / 1000, marker="o", label=city, color=color, linewidth=2.2, markersize=7)
    ax.set_title("Average Daily Revenue by Day of Week", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Avg Revenue (000 XAF)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Plot 3: External factors
    st.subheader("Revenue Impact of External Factors")
    factors = {
        "Rain Season": "rain_season",
        "Public Holiday": "is_holiday",
        "Big Market Day": "is_big_market_day",
        "Payday Period": "is_payday_period",
        "Weekend": "is_weekend",
    }
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for ax, (label, col), color in zip(axes, factors.items(), PALETTE):
        means = df.groupby(col)["daily_revenue_xaf"].mean() / 1000
        bars = ax.bar(["No", "Yes"], means.values, color=[color+"88", color], edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.1fK", padding=3, fontsize=9)
        pct = (means[1] - means[0]) / means[0] * 100
        ax.set_title(f"{label}\n({pct:+.1f}%)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Avg Revenue (000 XAF)" if ax is axes[0] else "")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Plot 4: Heatmap
    st.subheader("City × Product Category Revenue")
    pivot = df.pivot_table(values="daily_revenue_xaf", index="product_category", columns="city", aggfunc="mean") / 1000
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5, cbar_kws={"label": "Avg Daily Revenue (000 XAF)"})
    ax.set_title("Average Daily Revenue (000 XAF)\nCity × Product Category", fontweight="bold", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

# Clustering page
elif page == "Clustering":
    st.title("🔍 Vendor Clustering")
    
    # Prepare data for clustering
    vendor_features = df.groupby("vendor_id").agg({
        "daily_revenue_xaf": ["mean", "std"],
        "n_transactions": "mean",
        "avg_transaction_value_xaf": "mean",
        "city": "first",
        "product_category": "first"
    }).reset_index()
    vendor_features.columns = ["vendor_id", "rev_mean", "rev_std", "tx_mean", "avg_tx_mean", "city", "category"]
    
    # Encode categorical variables
    le_city = LabelEncoder()
    le_cat = LabelEncoder()
    vendor_features["city_enc"] = le_city.fit_transform(vendor_features["city"])
    vendor_features["category_enc"] = le_cat.fit_transform(vendor_features["category"])
    
    # Select features
    X_cluster = vendor_features[["rev_mean", "rev_std", "tx_mean", "avg_tx_mean", "city_enc", "category_enc"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # K-means clustering
    n_clusters = st.slider("Number of clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    vendor_features["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Display cluster info
    st.subheader("Cluster Profiles")
    cluster_summary = vendor_features.groupby("cluster").agg({
        "rev_mean": "mean",
        "tx_mean": "mean",
        "avg_tx_mean": "mean",
        "vendor_id": "count"
    }).round(0)
    cluster_summary.columns = ["Avg Revenue (XAF)", "Avg Transactions", "Avg Tx Value (XAF)", "Vendor Count"]
    st.dataframe(cluster_summary, width='stretch')
    
    # Plot clusters
    st.subheader("Clusters Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(vendor_features["rev_mean"], vendor_features["tx_mean"], c=vendor_features["cluster"], cmap="viridis", s=100, alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    ax.set_xlabel("Average Daily Revenue (XAF)")
    ax.set_ylabel("Average Number of Transactions")
    ax.set_title("Vendor Clusters")
    st.pyplot(fig)

# Prediction page
elif page == "Revenue Prediction":
    st.title("💰 Revenue Prediction")
    
    # Prepare data for prediction (cached)
    @st.cache_resource
    def get_model_and_encoders():
        df_pred = df.copy()
        le_city = LabelEncoder()
        le_market = LabelEncoder()
        le_category = LabelEncoder()
        le_dow = LabelEncoder()
        
        df_pred["city_enc"] = le_city.fit_transform(df_pred["city"])
        df_pred["market_enc"] = le_market.fit_transform(df_pred["market"])
        df_pred["category_enc"] = le_category.fit_transform(df_pred["product_category"])
        df_pred["dow_enc"] = le_dow.fit_transform(df_pred["day_of_week"])
        
        features = [
            "city_enc", 
            "market_enc", 
            "category_enc", 
            "dow_enc", 
            "month_num", 
            "is_weekend", 
            "is_holiday",
            "rain_season",
            "is_big_market_day",
            "is_payday_period"
        ]
        X = df_pred[features]
        y = df_pred["daily_revenue_xaf"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        performance = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        }
        
        return model, le_city, le_market, le_category, le_dow, features, performance
    
    model, le_city, le_market, le_category, le_dow, features, performance = get_model_and_encoders()
    
    st.subheader("Model Performance (Gradient Boosting)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{performance['MAE']:,.0f} FCFA")
    with col2:
        st.metric("RMSE", f"{performance['RMSE']:,.0f} FCFA")
    with col3:
        st.metric("R² Score", f"{performance['R2']:.3f}")
    
    # Prediction interface
    st.subheader("Predict Revenue")
    city = st.selectbox("City", list(CITY_MARKETS.keys()))
    market = st.selectbox("Market", CITY_MARKETS[city]["markets"])
    category = st.selectbox("Product Category", PRODUCT_CATEGORIES)
    dow = st.selectbox("Day of Week", DOW_ORDER)
    month = st.selectbox("Month", MONTH_ORDER)
    month_num = MONTH_ORDER.index(month) + 9
    is_weekend = 1 if dow in ["Saturday", "Sunday"] else 0
    is_holiday = st.checkbox("Is Public Holiday?", value=False)
    rain_season = st.checkbox("Is Rain Season?", value=False)
    is_big_market_day = st.checkbox("Is Big Market Day?", value=False)
    is_payday_period = st.checkbox("Is Payday Period?", value=False)
    
    # Encode inputs
    city_enc = le_city.transform([city])[0]
    market_enc = le_market.transform([market])[0]
    category_enc = le_category.transform([category])[0]
    dow_enc = le_dow.transform([dow])[0]
    
    input_data = pd.DataFrame([[
        city_enc, 
        market_enc, 
        category_enc, 
        dow_enc, 
        month_num, 
        is_weekend, 
        is_holiday,
        rain_season,
        is_big_market_day,
        is_payday_period
    ]], columns=features)
    prediction = model.predict(input_data)[0]
    
    st.metric("Predicted Daily Revenue", f"{prediction:,.0f} FCFA")

# Association Rules page
elif page == "Association Rules":
    st.title("🔗 Association Rules Mining")
    
    st.markdown("This section mines association rules between vendor attributes.")
    
    # Prepare transaction data
    vendor_attrs = df.groupby("vendor_id").agg({
        "city": "first",
        "product_category": "first",
        "market": "first"
    }).reset_index()
    
    transactions = []
    for _, row in vendor_attrs.iterrows():
        transactions.append([f"City:{row['city']}", f"Category:{row['product_category']}", f"Market:{row['market']}"])
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Parameters
    min_support = st.slider("Minimum Support", 0.05, 0.5, 0.1)
    min_threshold = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)
    
    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    
    if len(rules) > 0:
        st.subheader("Generated Rules")
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values("lift", ascending=False), width='stretch')
    else:
        st.info("No rules found with the current parameters. Try lowering the thresholds.")

