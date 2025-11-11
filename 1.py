import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ------------------------------
# Page Config and Styling
# ------------------------------
st.set_page_config(page_title="Domestic Flight Analysis", layout="wide")

st.markdown("""
    <style>
        .main {background-color:#0f172a; color:#e5e7eb;}
        h1, h2, h3 {color:#a78bfa; text-align:center;}
        .section {background:#111827; border-radius:10px; padding:25px; margin:15px;}
        .stTabs [role="tablist"] button {background:#1e293b; color:#a78bfa; border:none;}
        .stTabs [role="tablist"] button:hover {background:#312e81;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("domestic_city_processed.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/domestic/domestic_city_processed.csv")
        except FileNotFoundError:
            st.error("❌ 'domestic_city_processed.csv' not found. Place it in the same folder or under data/domestic/")
            return pd.DataFrame()
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_data()
if df.empty:
    st.stop()

st.markdown("<h1>🇮🇳 Domestic Flight Analysis Dashboard</h1>", unsafe_allow_html=True)

# ------------------------------
# Helper
# ------------------------------
def safe_group(df, group_col, value_col, aggfunc='sum'):
    try:
        return df.groupby(group_col)[value_col].agg(aggfunc).reset_index()
    except Exception:
        return pd.DataFrame()

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs([
    "Correlation", "Top Routes", "Top Cities", "Yearly Trend",
    "Monthly Seasonality", "Freight vs Passenger", "Mail vs Freight",
    "Growth", "PCA (2D)", "Clustering"
])

# ------------------------------
# 1️⃣ Correlation Heatmap
# ------------------------------
with tabs[0]:
    st.markdown("<div class='section'><h2>Correlation Heatmap</h2>", unsafe_allow_html=True)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        st.warning("No numeric columns found.")
    else:
        fig = px.imshow(num.corr(), text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 2️⃣ Top Routes
# ------------------------------
with tabs[1]:
    st.markdown("<div class='section'><h2>Top 10 Busiest City Pairs</h2>", unsafe_allow_html=True)
    if {"city1", "city2"}.issubset(df.columns):
        df["route"] = df["city1"].astype(str) + "–" + df["city2"].astype(str)
        if "total_passengers" in df.columns:
            top_routes = df.groupby("route")["total_passengers"].sum().nlargest(10).reset_index()
            fig = px.bar(top_routes, x="route", y="total_passengers", title="Top 10 Busiest City Pairs")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_routes)
        else:
            st.warning("Column 'total_passengers' not found.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 3️⃣ Top Cities (Origin/Destination)
# ------------------------------
with tabs[2]:
    st.markdown("<div class='section'><h2>Top Origin & Destination Cities</h2>", unsafe_allow_html=True)
    if "city1" in df.columns and "paxfromcity2" in df.columns:
        top_origin = df.groupby("city1")["paxfromcity2"].sum().nlargest(10).reset_index()
        top_dest = df.groupby("city2")["paxtocity2"].sum().nlargest(10).reset_index()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Origins")
            st.plotly_chart(px.bar(top_origin, x="city1", y="paxfromcity2"), use_container_width=True)
        with col2:
            st.subheader("Top Destinations")
            st.plotly_chart(px.bar(top_dest, x="city2", y="paxtocity2"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 4️⃣ Yearly Trend
# ------------------------------
with tabs[3]:
    st.markdown("<div class='section'><h2>Yearly Passenger Trend</h2>", unsafe_allow_html=True)
    if "year" in df.columns and "total_passengers" in df.columns:
        yearly = df.groupby("year")["total_passengers"].sum().reset_index()
        st.plotly_chart(px.line(yearly, x="year", y="total_passengers", markers=True), use_container_width=True)
        st.dataframe(yearly)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 5️⃣ Monthly Seasonality
# ------------------------------
with tabs[4]:
    st.markdown("<div class='section'><h2>Monthly Passenger Seasonality</h2>", unsafe_allow_html=True)
    if "month" in df.columns and "total_passengers" in df.columns:
        monthly = df.groupby("month")["total_passengers"].mean().reset_index()
        st.plotly_chart(px.line(monthly, x="month", y="total_passengers", markers=True), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 6️⃣ Freight vs Passenger
# ------------------------------
with tabs[5]:
    st.markdown("<div class='section'><h2>Freight vs Passenger Volume</h2>", unsafe_allow_html=True)
    if {"total_freight", "total_passengers"}.issubset(df.columns):
        fig = px.scatter(df, x="total_passengers", y="total_freight", trendline="ols",
                         title="Freight vs Passenger Volume")
        st.plotly_chart(fig, use_container_width=True)
        corr = df["total_passengers"].corr(df["total_freight"])
        st.info(f"Correlation: {corr:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 7️⃣ Mail vs Freight
# ------------------------------
with tabs[6]:
    st.markdown("<div class='section'><h2>Mail vs Freight Growth (Yearly)</h2>", unsafe_allow_html=True)
    if {"year", "total_mail", "total_freight"}.issubset(df.columns):
        yearly_mail = df.groupby("year")["total_mail"].sum()
        yearly_freight = df.groupby("year")["total_freight"].sum()
        combo = pd.DataFrame({"year": yearly_mail.index, "mail": yearly_mail.values, "freight": yearly_freight.values})
        fig = px.line(combo, x="year", y=["mail", "freight"], markers=True)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 8️⃣ Growth %
# ------------------------------
with tabs[7]:
    st.markdown("<div class='section'><h2>Top 10 Fastest Growing City Pairs</h2>", unsafe_allow_html=True)
    if "pax_growth_pct" in df.columns:
        if {"city1", "city2"}.issubset(df.columns):
            df["route"] = df["city1"] + "–" + df["city2"]
        top_growth = df.groupby("route")["pax_growth_pct"].mean().nlargest(10).reset_index()
        fig = px.bar(top_growth, x="route", y="pax_growth_pct")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_growth)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 9️⃣ PCA
# ------------------------------
with tabs[8]:
    st.markdown("<div class='section'><h2>PCA (2D Projection)</h2>", unsafe_allow_html=True)
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[1] >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric)
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(comps, columns=["PC1", "PC2"])
        fig = px.scatter(df_pca, x="PC1", y="PC2", title="2D PCA Projection")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Explained Variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# 🔟 Clustering
# ------------------------------
with tabs[9]:
    st.markdown("<div class='section'><h2>K-Means Clustering</h2>", unsafe_allow_html=True)
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[1] >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric)
        k = st.slider("Number of Clusters (K)", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_scaled)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])
        df_plot["Cluster"] = df["Cluster"].astype(str)
        fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster", title="K-Means Clustering")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.groupby("Cluster").mean().round(2))
    st.markdown("</div>", unsafe_allow_html=True)
