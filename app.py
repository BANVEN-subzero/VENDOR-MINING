
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Vendor Revenue Analysis",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4a90e2, #357abd) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
    }
    .stMetric {
        background: white !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15) !important;
        color: #111827 !important;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: #374151 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    h1 {
        color: #1e3a8a !important;
        font-weight: 800;
    }
    h2 {
        color: #3730a3 !important;
        font-weight: 700;
    }
    h3 {
        color: #4f46e5 !important;
        font-weight: 600;
    }
    .stMarkdown, .stText, .stMarkdown p {
        color: #1f2937 !important;
    }
    .stSelectbox label, .stSlider label {
        color: #374151 !important;
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
        color: #1f2937 !important;
    }
    .stSelectbox [data-baseweb="select"] div {
        color: #1f2937 !important;
    }
    .stCheckbox label {
        color: #1f2937 !important;
    }
    [data-testid="stWidgetLabel"] label {
        color: #374151 !important;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3a8a 0%, #3730a3 100%) !important;
    }
    .sidebar .sidebar-content .stRadio label {
        color: white !important;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Global config
PALETTE = ["#1e3a8a", "#3b82f6", "#60a5fa", "#93c5fd", "#f59e0b", "#ef4444", "#10b981"]
BG_COLOR = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.grid": True,
    "grid.color": "#e0e7ff",
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
st.sidebar.title("🏪 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Exploratory Data Analysis", "🔍 Vendor Clustering", "💰 Revenue Prediction", "🔗 Association Rules"])

# Home page
if page == "🏠 Home":
    st.title("🏪 Vendor Revenue Analysis Dashboard")
    st.markdown("---")
    
    st.markdown("""
    <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="margin-top: 0;">Welcome to the Vendor Revenue Analysis Dashboard!</h3>
        <p>This dashboard analyzes street vendor transaction patterns in Cameroon to predict daily revenue. Explore insights, clusters, and predictions with our interactive tools!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Total Records", f"{len(df):,}")
    with col2:
        st.metric("👥 Unique Vendors", df["vendor_id"].nunique())
    with col3:
        st.metric("💵 Avg Daily Revenue", f"{df['daily_revenue_xaf'].mean():,.0f} FCFA")
    
    st.write("")
    st.subheader("📋 Sample Data")
    st.dataframe(df.head(10), width='stretch')
    
    st.write("")
    st.markdown("---")
    st.subheader("🗺️ Key Features")
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    with feature_col1:
        st.markdown("""
        <div style="background: white; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4>📊 Exploratory Data Analysis</h4>
            <p>Visualize revenue distributions, trends, and factor impacts.</p>
        </div>
        """, unsafe_allow_html=True)
    with feature_col2:
        st.markdown("""
        <div style="background: white; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4>🔍 Vendor Clustering</h4>
            <p>Discover distinct vendor segments using K-means.</p>
        </div>
        """, unsafe_allow_html=True)
    with feature_col3:
        st.markdown("""
        <div style="background: white; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4>💰 Revenue Prediction</h4>
            <p>Predict daily revenue with Gradient Boosting.</p>
        </div>
        """, unsafe_allow_html=True)

# EDA page
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    # Plot 1: Revenue distribution by city
    st.subheader("📦 Revenue Distribution by City")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, city, color in zip(axes, CITY_MARKETS.keys(), PALETTE):
        subset = df[df["city"] == city]["daily_revenue_xaf"] / 1000
        ax.hist(subset, bins=40, color=color, alpha=0.85, edgecolor="white", linewidth=1)
        ax.axvline(subset.mean(), color="#dc2626", linestyle="--", lw=2, label=f"Mean: {subset.mean():.1f}K")
        ax.axvline(subset.median(), color="#f59e0b", linestyle=":", lw=2, label=f"Median: {subset.median():.1f}K")
        ax.set_title(city, fontweight="bold", fontsize=14, pad=12)
        ax.set_xlabel("Daily Revenue (000 FCFA)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("")
    
    # Plot 2: Day of week pattern
    st.subheader("📅 Average Daily Revenue by Day of Week")
    dow_city = df.groupby(["city", "day_of_week"])["daily_revenue_xaf"].mean().reset_index()
    dow_city["day_of_week"] = pd.Categorical(dow_city["day_of_week"], categories=DOW_ORDER, ordered=True)
    dow_city = dow_city.sort_values("day_of_week")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for city, color in zip(CITY_MARKETS.keys(), PALETTE):
        sub = dow_city[dow_city["city"] == city]
        ax.plot(sub["day_of_week"], sub["daily_revenue_xaf"] / 1000, marker="o", label=city, color=color, linewidth=3, markersize=9)
    ax.set_title("Average Daily Revenue by Day of Week", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("Avg Revenue (000 FCFA)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("")
    
    # Plot 3: External factors
    st.subheader("⚡ Revenue Impact of External Factors")
    factors = {
        "Rain Season": "rain_season",
        "Public Holiday": "is_holiday",
        "Big Market Day": "is_big_market_day",
        "Payday Period": "is_payday_period",
        "Weekend": "is_weekend",
    }
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, (label, col), color in zip(axes, factors.items(), PALETTE):
        means = df.groupby(col)["daily_revenue_xaf"].mean() / 1000
        bars = ax.bar(["No", "Yes"], means.values, color=[color + "66", color], edgecolor="white", width=0.6, linewidth=1.5)
        ax.bar_label(bars, fmt="%.1fK", padding=8, fontsize=10, fontweight="600")
        pct = (means[1] - means[0]) / means[0] * 100
        ax.set_title(f"{label}\n({pct:+.1f}%)", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Avg Revenue (000 FCFA)" if ax is axes[0] else "", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("")
    
    # Plot 4: Heatmap
    st.subheader("🔥 City × Product Category Revenue Heatmap")
    pivot = df.pivot_table(values="daily_revenue_xaf", index="product_category", columns="city", aggfunc="mean") / 1000
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", ax=ax, linewidths=0.8, cbar_kws={"label": "Avg Daily Revenue (000 FCFA)"}, annot_kws={"fontsize": 10})
    ax.set_title("Average Daily Revenue (000 FCFA)\nCity × Product Category", fontweight="bold", fontsize=14, pad=20)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

# Clustering page
elif page == "🔍 Vendor Clustering":
    st.title("🔍 Vendor Clustering")
    st.markdown("---")
    
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
    st.subheader("⚙️ Clustering Settings")
    n_clusters = st.slider("Number of clusters", 2, 6, 3, help="Select the number of vendor segments to identify")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    vendor_features["cluster"] = kmeans.fit_predict(X_scaled)
    
    st.write("")
    
    # Display cluster info
    st.subheader("📊 Cluster Profiles")
    cluster_summary = vendor_features.groupby("cluster").agg({
        "rev_mean": "mean",
        "tx_mean": "mean",
        "avg_tx_mean": "mean",
        "vendor_id": "count"
    }).round(0)
    cluster_summary.columns = ["Avg Revenue (FCFA)", "Avg Transactions", "Avg Tx Value (FCFA)", "Vendor Count"]
    st.dataframe(cluster_summary, width='stretch')
    
    st.write("")
    
    # Plot clusters
    st.subheader("🎯 Clusters Visualization")
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(vendor_features["rev_mean"], vendor_features["tx_mean"], c=vendor_features["cluster"], cmap="viridis", s=120, alpha=0.8, edgecolor='white', linewidth=1.5)
    plt.colorbar(scatter, label="Cluster", pad=0.02)
    ax.set_xlabel("Average Daily Revenue (FCFA)", fontsize=12, labelpad=12)
    ax.set_ylabel("Average Number of Transactions", fontsize=12, labelpad=12)
    ax.set_title("Vendor Clusters", fontsize=15, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Prediction page
elif page == "💰 Revenue Prediction":
    st.title("💰 Revenue Prediction")
    st.markdown("---")
    
    # Prepare data for prediction (cached)
    @st.cache_resource
    def get_model_and_encoders(data):
        df_pred = data.copy()
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
    
    model, le_city, le_market, le_category, le_dow, features, performance = get_model_and_encoders(df)
    
    st.subheader("📈 Model Performance (Gradient Boosting)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{performance['MAE']:,.0f} FCFA")
    with col2:
        st.metric("RMSE", f"{performance['RMSE']:,.0f} FCFA")
    with col3:
        st.metric("R² Score", f"{performance['R2']:.3f}")
    
    st.write("")
    st.markdown("---")
    
    # Prediction interface
    st.subheader("🔮 Predict Revenue")
    st.markdown("<div style='background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        city = st.selectbox("🏙️ City", list(CITY_MARKETS.keys()))
        market = st.selectbox("🏪 Market", CITY_MARKETS[city]["markets"])
        category = st.selectbox("📦 Product Category", PRODUCT_CATEGORIES)
        dow = st.selectbox("📅 Day of Week", DOW_ORDER)
    with input_col2:
        month = st.selectbox("📆 Month", MONTH_ORDER)
        month_num = MONTH_ORDER.index(month) + 9
        is_weekend = 1 if dow in ["Saturday", "Sunday"] else 0
        
        st.write("**External Factors:**")
        is_holiday = st.checkbox("🎉 Is Public Holiday?", value=False)
        rain_season = st.checkbox("🌧️ Is Rain Season?", value=False)
        is_big_market_day = st.checkbox("🛒 Is Big Market Day?", value=False)
        is_payday_period = st.checkbox("💸 Is Payday Period?", value=False)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
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
    
    st.write("")
    st.markdown("""
    <div style="text-align: center; padding: 32px; background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%); border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
        <h2 style="color: white; margin-bottom: 8px;">Predicted Daily Revenue</h2>
        <h1 style="color: white; font-size: 48px; margin: 0;">{:,.0f} FCFA</h1>
    </div>
    """.format(prediction), unsafe_allow_html=True)

# Association Rules page
elif page == "🔗 Association Rules":
    st.title("🔗 Association Rules Mining")
    st.markdown("---")
    
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="margin-top: 0;">📋 About Association Rules</h3>
        <p>This section mines interesting associations between vendor attributes like city, product category, and market.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
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
    st.subheader("⚙️ Parameters")
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        min_support = st.slider("Minimum Support", 0.05, 0.5, 0.1, 0.01, help="Minimum frequency of itemset")
    with param_col2:
        min_threshold = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05, help="Minimum confidence for rules")
    
    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    
    st.write("")
    
    if len(rules) > 0:
        st.subheader("✅ Generated Rules")
        st.markdown(f"**Found {len(rules)} rules**")
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values("lift", ascending=False), width='stretch')
    else:
        st.info("⚠️ No rules found with the current parameters. Try lowering the thresholds!")
