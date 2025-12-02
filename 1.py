
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
import networkx as nx
import plotly.graph_objects as go


def plot_network_graph(G, theme_color):

    pos = nx.spring_layout(G, k=0.3, iterations=50)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color="#888"),
        hoverinfo='none'
    )
    node_x, node_y = zip(*pos.values())
    node_text = list(pos.keys())
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=12, color=theme_color),
        text=node_text,
        textposition="top center",
        hoverinfo='text'
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="Route Network",
        showlegend=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


@st.cache_resource
def build_network_graph(df):

    G = nx.from_pandas_edgelist(
        df,
        source="city1",
        target="city2",
        edge_attr="total_passengers",
        create_using=nx.Graph()
    )
    return G

@st.cache_data
def get_route_summary(df):
    df["route"] = df["city1"] + " → " + df["city2"]
    table = df.groupby("route")["total_passengers"].sum().reset_index()
    return table.sort_values("total_passengers", ascending=False)

def compute_route_stats(df, route):
    city1, city2 = route.split(" → ")
    route_data = df[(df["city1"] == city1) & (df["city2"] == city2)]

    avg_pax = route_data["total_passengers"].mean()
    mode_month_num = route_data["month"].mode()[0]

    month_map = {
        1:"January",2:"February",3:"March",4:"April",
        5:"May",6:"June",7:"July",8:"August",
        9:"September",10:"October",11:"November",12:"December"
    }

    monthly = (
        route_data.groupby("month")["total_passengers"]
        .mean()
        .reset_index()
    )
    monthly["month_name"] = monthly["month"].map(month_map)

    season_map = {
        12: "Winter", 1: "Winter",
        2: "Spring", 3: "Spring",
        4: "Summer", 5: "Summer", 6: "Summer",
        7: "Monsoon", 8: "Monsoon",
        9: "Festive", 10: "Festive", 11: "Festive"
    }

    route_data["season"] = route_data["month"].map(season_map)

    seasonal_avg = (
        route_data.groupby("season")["total_passengers"]
        .mean()
        .reset_index()
        .sort_values("total_passengers", ascending=False)
    )

    top_season = seasonal_avg.iloc[0]["season"]

    explanation = {
        "Winter": "High winter travel demand due to holidays and tourism.",
        "Spring": "Moderate festive/holiday travel.",
        "Summer": "Vacation-heavy route with high tourism.",
        "Monsoon": "Generally low travel due to weather.",
        "Festive": "High travel during festival + business period."
    }

    season_explanation = explanation.get(top_season, "Seasonal pattern detected.")

    return avg_pax,month_map[mode_month_num],monthly,seasonal_avg,top_season,season_explanation



st.set_page_config(page_title="Flight Analysis Dashboard", layout="wide")

st.markdown("""
<style>
/* ====== Global Layout ====== */
.main {
    background-color: #0b1220;
    color: #e0eaf1;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* ====== Headings ====== */
h1, h2, h3 {
    text-align: center;
    color: #4fd1c5;  /* teal accent */
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ====== Section Cards ====== */
.section {
    background: linear-gradient(145deg, #0f172a, #111a2e);
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.3s ease-in-out;
}
.section:hover {
    transform: scale(1.01);
    box-shadow: 0 0 15px rgba(79, 209, 197, 0.25);
}

/* ====== Tabs ====== */
.stTabs [role="tablist"] {
    justify-content: center;
    margin-bottom: 1rem;
}
.stTabs [role="tablist"] button {
    background-color: #1a2337;
    color: #a0aec0;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    margin: 0 4px;
    transition: all 0.25s ease;
    font-weight: 500;
}
.stTabs [role="tablist"] button:hover {
    background-color: #2d3748;
    color: #63e6be;
    transform: translateY(-1px);
}
.stTabs [role="tablist"] button[aria-selected="true"] {
    background-color: #2c5282;
    color: #e6f9ff;
    box-shadow: 0 0 10px rgba(99, 230, 190, 0.4);
}

/* ====== DataFrame Tables ====== */
.dataframe {
    background-color: #0f172a !important;
    color: #e0eaf1 !important;
    border-radius: 8px;
    border: 1px solid #2c5282;
}

/* ====== Buttons ====== */
.stButton>button {
    background: linear-gradient(90deg, #3182ce, #2b6cb0);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-weight: 500;
    box-shadow: 0 3px 8px rgba(50, 150, 255, 0.25);
    transition: all 0.25s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #2b6cb0, #2c5282);
    box-shadow: 0 0 10px rgba(79, 209, 197, 0.3);
    transform: translateY(-2px);
}

/* ====== Plot Titles ====== */
.plotly-graph-div text {
    fill: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1>✈️ Flight Analysis Dashboard</h1>", unsafe_allow_html=True)

dataset_choice = st.selectbox(
    "Select Dataset:",
    ["Domestic"],
    help="Choose dataset to analyze."
)

@st.cache_data
def load_data(choice):
    if choice == "Domestic":
        paths = ["domestic_city_processed.csv"]
    else:
        paths = ["city_internatinal.csv"]

    for path in paths:
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower().str.strip()
            return df
        except FileNotFoundError:
            continue

    st.error(f"Data file for {choice} not found. Please ensure it's in the correct folder.")
    return pd.DataFrame()

df = load_data(dataset_choice)
if df.empty:
    st.stop()

theme_color = "#a78bfa" 

month_names = {
    1: "January", 2: "February", 3: "March",
    4: "April",   5: "May",      6: "June",
    7: "July",    8: "August",   9: "September",
    10: "October",11: "November",12: "December"
}

df["month_name"] = df["month"].map(month_names)

if dataset_choice == "Domestic":
    main_tab1, main_tab2, main_tab3 = st.tabs(["Statistics", "Correlation", "Modeling"])

    with main_tab1:
        st.subheader("Statistics")
        stats_tabs = st.tabs([
            "Overview", "Top Routes", "Top Cities",
            "Yearly Trend", "Monthly trend",
            "Traffic Composition", "Route Composition","Route analysis"
        ])
        
        with stats_tabs[0]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Overview Summary</h2>", unsafe_allow_html=True)

            latest_year = df['year'].max()
            df_latest = df[df['year'] == latest_year]
            total_passengers = df_latest['total_passengers'].sum()
            total_freight = df_latest['total_freight'].sum()
            total_mail = df_latest['total_mail'].sum()
            month_order = [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ]
            month_avg = df.groupby('month_name')['total_passengers'].mean().reindex(month_order) 
            busiest_month = month_avg.idxmax()
            df['route'] = df['city1'] + " → " + df['city2']
            busiest_route = (
                df.groupby("route")["total_passengers"]
                .sum()
                .idxmax()
            )
            top_cities = (
                df.groupby("city1")["paxfromcity2"].sum() +
                df.groupby("city2")["paxtocity2"].sum()
            ).nlargest(3)

            total_cities = len(set(df['city1']).union(set(df['city2'])))
            total_routes = df['route'].nunique()
            avg_passengers_per_route = (
                df.groupby("route")["total_passengers"].sum().mean()
            )
            route_year = (
                df.groupby(["route", "year"])["total_passengers"]
                .sum()
                .reset_index()
            )
            route_year["growth"] = route_year.groupby("route")["total_passengers"].pct_change()
            if route_year["growth"].notna().any():
                fastest_growth_row = route_year.loc[route_year["growth"].idxmax()]
                fastest_growing_route = fastest_growth_row["route"]
                fastest_growing_growth = fastest_growth_row["growth"]*100
            else:
                fastest_growing_route = "N/A"
                fastest_growing_growth = 0


            st.markdown(f"<h3 style='color:{theme_color}; text-align:center;'>✨ Key Performance Indicators (for {latest_year})</h3>", 
                        unsafe_allow_html=True)

            kpi_style = """
            background: linear-gradient(135deg, #1e293b, #0f172a);
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            border: 1px solid #1e293b;
            transition: all 0.3s ease;
            """

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Total Passengers</h4>
                        <p style="font-size:22px; font-weight:600; color:#e2e8f0;">
                            {total_passengers:,.0f}
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Year {latest_year}</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Total Freight</h4>
                        <p style="font-size:22px; font-weight:600; color:#e2e8f0;">
                            {total_freight:,.0f} kg
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Year {latest_year}</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Total Mail</h4>
                        <p style="font-size:22px; font-weight:600; color:#e2e8f0;">
                            {total_mail:,.0f} kg
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Year {latest_year}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

            col4, col5, col6 = st.columns(3)

            with col4:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Busiest Month</h4>
                        <p style="font-size:22px; font-weight:600; color:#e2e8f0;">
                            {busiest_month}
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Based on Avg. Monthly Traffic</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col5:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Top Route</h4>
                        <p style="font-size:20px; font-weight:600; color:#e2e8f0;">
                            {busiest_route}
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Highest Passenger Volume</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col6:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Total Cities / Airports</h4>
                        <p style="font-size:22px; font-weight:600; color:#e2e8f0;">
                            {total_cities}
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Across the Dataset</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
            col7, col8, col9 = st.columns(3)
            with col7:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Total Routes</h4>
                        <p style="font-size:22px; font-weight:600; color:#e2e8f0;">
                            {total_routes}
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Unique City Pairs</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col8:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Avg Passengers per Route</h4>
                        <p style="font-size:22px; font-weight:600; color:#e2e8f0;">
                            {avg_passengers_per_route:,.0f}
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Across All Routes</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col9:
                st.markdown(
                    f"""
                    <div style="{kpi_style}">
                        <h4 style="color:#63e6be;">Fastest Growing Route</h4>
                        <p style="font-size:20px; font-weight:600; color:#e2e8f0;">
                            {fastest_growing_route}
                        </p>
                        <p style="color:#94a3b8; font-size:12px;">Highest Route Growth</p>
                    </div>
                    """, unsafe_allow_html=True
                )


                
            st.markdown("### Top 3 Cities by Total Passenger Traffic")

            for city, traffic in top_cities.items():
                st.markdown(
                    f"""
                    <div style="
                        background-color:#111a2e;
                        padding:12px 18px;
                        margin:8px 0;
                        border-radius:10px;
                        border-left:4px solid #4fd1c5;
                        font-size:16px;
                        color:#e2e8f0;">
                        <strong style="color:#63e6be;">{city}</strong>  
                        <span style="float:right; color:#94a3b8;">{traffic:,.0f} passengers</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown("### Passenger Share by Month")
            fig = px.line(month_avg.reset_index(), x="month_name", y="total_passengers", markers=True, color_discrete_sequence=[theme_color])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        
        with stats_tabs[1]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Top 10 Busiest City Pairs (by route) </h2>", unsafe_allow_html=True)
            if {"city1", "city2"}.issubset(df.columns):
                df["route"] = df["city1"].astype(str) + "->" + df["city2"].astype(str)
                if "total_passengers" in df.columns:
                    top_routes = df.groupby("route")["total_passengers"].sum().nlargest(10).reset_index()
                    fig = px.bar(top_routes, x="route", y="total_passengers", color_discrete_sequence=[theme_color])
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(top_routes)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with stats_tabs[2]:
            st.markdown(
                f"<div class='section'><h2 style='color:{theme_color};'>Top Origin & Destination Cities</h2>",
                unsafe_allow_html=True
            )

            required_cols = {"city1", "city2", "paxfromcity2", "paxtocity2"}
            if required_cols.issubset(df.columns):

                top_origin = (
                    df.groupby("city1")["paxfromcity2"]
                    .sum()
                    .nlargest(10)
                    .reset_index()
                )

                top_dest = (
                    df.groupby("city2")["paxtocity2"]
                    .sum()
                    .nlargest(10)
                    .reset_index()
                )
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top Origins")
                    fig_origin = px.bar(
                        top_origin,
                        x="city1",
                        y="paxfromcity2",
                        color_discrete_sequence=[theme_color]
                    )
                    st.plotly_chart(fig_origin, use_container_width=True)
                    st.dataframe(top_origin)

                with col2:
                    st.subheader("Top Destinations")
                    fig_dest = px.bar(
                        top_dest,
                        x="city2",
                        y="paxtocity2",
                        color_discrete_sequence=[theme_color]
                    )
                    st.plotly_chart(fig_dest, use_container_width=True)
                    st.dataframe(top_dest)

            else:
                st.warning("Required columns for Tab 2 are missing in the dataset.")

            st.markdown("</div>", unsafe_allow_html=True)
            
        with stats_tabs[3]:
            st.markdown(
                f"<div class='section'><h2 style='color:{theme_color};'>Yearly Passenger Traffic Trend</h2>",
                unsafe_allow_html=True
            )
            if {"year", "total_passengers"}.issubset(df.columns):
                yearly_trend = (
                    df.groupby("year")["total_passengers"]
                    .sum()
                    .sort_index()
                    .reset_index()
                )
                fig = px.line(
                    yearly_trend,
                    x="year",
                    y="total_passengers",
                    markers=True,
                    color_discrete_sequence=[theme_color]
                )
                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Total Passengers",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(yearly_trend)
            else:
                st.warning("Required columns 'year' or 'total_passengers' are missing.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with stats_tabs[4]:
            st.markdown(
                f"<div class='section'><h2 style='color:{theme_color};'>Monthly Passenger Seasonality</h2>",
                unsafe_allow_html=True
            )
            if {"month", "total_passengers"}.issubset(df.columns):
                month_map = {
                    1: "Jan",  2: "Feb",  3: "Mar",  4: "Apr",
                    5: "May",  6: "Jun",  7: "Jul",  8: "Aug",
                    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                }
                monthly_seasonality = (
                    df.groupby("month")["total_passengers"]
                    .mean()
                    .reindex(range(1, 12 + 1))
                    .reset_index()
                )
                monthly_seasonality["month_name"] = monthly_seasonality["month"].map(month_map)
                fig = px.line(
                    monthly_seasonality,
                    x="month_name",
                    y="total_passengers",
                    markers=True,
                    color_discrete_sequence=[theme_color]
                )
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Average Passengers",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(monthly_seasonality)
            else:
                st.warning("Required columns 'month' or 'total_passengers' are missing.")

            st.markdown("</div>", unsafe_allow_html=True)

        with stats_tabs[5]:
            
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Traffic Volume Distribution</h2>", unsafe_allow_html=True)
            st.markdown("### Histogram Distribution (Passengers, Freight, Mail)")
            cols = st.columns(3)
            with cols[0]:
                fig = px.histogram(
                    df,
                    x="total_passengers",
                    nbins=40,
                    title="Total Passengers",
                    color_discrete_sequence=[theme_color]
                )
                st.plotly_chart(fig, use_container_width=True)
            with cols[1]:
                fig = px.histogram(
                    df,
                    x="total_freight",
                    nbins=40,
                    title="Total Freight",
                    color_discrete_sequence=[theme_color]
                )
                st.plotly_chart(fig, use_container_width=True)
            with cols[2]:
                fig = px.histogram(
                    df,
                    x="total_mail",
                    nbins=40,
                    title="Total Mail",
                    color_discrete_sequence=[theme_color]
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Log-Scale Distribution (Handles Skewness)")

            cols2 = st.columns(3)
            with cols2[0]:
                fig = px.histogram(
                    df,
                    x=np.log1p(df["total_passengers"]),
                    nbins=40,
                    title="Passengers (Log Scale)",
                    color_discrete_sequence=[theme_color]
                )
                st.plotly_chart(fig, use_container_width=True)

            with cols2[1]:
                fig = px.histogram(
                    df,
                    x=np.log1p(df["total_freight"]),
                    nbins=40,
                    title="Freight (Log Scale)",
                    color_discrete_sequence=[theme_color]
                )
                st.plotly_chart(fig, use_container_width=True)

            with cols2[2]:
                fig = px.histogram(
                    df,
                    x=np.log1p(df["total_mail"]),
                    nbins=40,
                    title="Mail (Log Scale)",
                    color_discrete_sequence=[theme_color]
                )
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown("### Violin Plots (Spread & Density)")
            cols3 = st.columns(3)

            with cols3[0]:
                fig = px.violin(df, y="total_passengers", box=True, points="all",
                                title="Passengers Violin Plot",
                                color_discrete_sequence=[theme_color])
                st.plotly_chart(fig, use_container_width=True)

            with cols3[1]:
                fig = px.violin(df, y="total_freight", box=True, points="all",
                                title="Freight Violin Plot",
                                color_discrete_sequence=[theme_color])
                st.plotly_chart(fig, use_container_width=True)

            with cols3[2]:
                fig = px.violin(df, y="total_mail", box=True, points="all",
                                title="Mail Violin Plot",
                                color_discrete_sequence=[theme_color])
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)
            
        with stats_tabs[6]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>🛫 Route Traffic Composition</h2>", unsafe_allow_html=True)

            df["route"] = df["city1"] + " → " + df["city2"]

            comp = (
                df.groupby("route")[["total_passengers", "total_freight", "total_mail"]]
                .sum()
                .reset_index()
            )
            comp["total_traffic"] = (
                comp["total_passengers"] +
                comp["total_freight"] +
                comp["total_mail"]
            )

            top_n = st.slider("Select number of top routes", 5, 50, 10)
            comp_top = comp.nlargest(top_n, "total_traffic")
            comp_melted = comp_top.melt(
                id_vars="route",
                value_vars=["total_passengers", "total_freight", "total_mail"],
                var_name="Traffic Type",
                value_name="Volume"
            )

            comp_melted["Traffic Type"] = comp_melted["Traffic Type"].replace({
                "total_passengers": "Passengers",
                "total_freight": "Freight",
                "total_mail": "Mail"
            })
            fig = px.bar(
                comp_melted,
                x="route",
                y="Volume",
                color="Traffic Type",
                title="Passenger vs Freight vs Mail (Route-wise)",
                color_discrete_map={
                    "Passengers": "#4fd1c5",
                    "Freight": "#60a5fa",
                    "Mail": "#a78bfa"
                }
            )

            fig.update_layout(xaxis=dict(title="Route", tickangle=45))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Data Table")
            st.dataframe(comp_top[["route", "total_passengers", "total_freight", "total_mail"]])
            st.markdown("</div>", unsafe_allow_html=True)
            
        with stats_tabs[7]:

            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>🕸 Route Network</h2>", unsafe_allow_html=True)

            G = build_network_graph(df)

            if st.checkbox("Show Route Network Graph"):
                fig = plot_network_graph(G, theme_color) 
                st.plotly_chart(fig, use_container_width=True)

            routes_df = get_route_summary(df)
            st.subheader("All Routes")
            st.write(f"Total Routes: **{len(routes_df)}**")
            st.dataframe(routes_df)

            selected_route = st.selectbox("Select a Route:", routes_df["route"])
            (
                avg_pax,
                mode_month,
                monthly,
                seasonal_avg,
                top_season,
                season_explanation
            ) = compute_route_stats(df, selected_route)

            st.metric("Average Monthly Passengers", f"{avg_pax:,.0f}")
            st.metric("Most Frequent Month", mode_month)
            st.subheader("Seasonal Trend Classification")
            st.metric("Top Season", top_season)
            st.info(season_explanation)
            st.write("### Seasonal Passenger Averages")
            st.dataframe(seasonal_avg)

            fig_season = px.bar(
                seasonal_avg,
                x="season",
                y="total_passengers",
                color="season",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title=f"Seasonal Trend for {selected_route}"
            )
            st.plotly_chart(fig_season, use_container_width=True)
            st.subheader("Monthly Passenger Trend")

            st.plotly_chart(
                px.line(
                    monthly,
                    x="month_name",
                    y="total_passengers",
                    markers=True,
                    title=f"Monthly Trend for {selected_route}",
                    color_discrete_sequence=[theme_color]
                ),
                use_container_width=True
    
            )


    with main_tab2:
        corr_tabs = st.tabs([
            "Correlation Heatmap", "EDA",
            "Mail vs Freight","Association rules"
        ])

        with corr_tabs[0]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Correlation Heatmap</h2>", unsafe_allow_html=True)

            num = df.select_dtypes(include=[np.number])

            if num.shape[1] < 2:
                st.warning("Not enough numeric columns to compute correlation.")
            else:
                corr = num.corr().round(2)

                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale=[
                        "#991b1b",  
                        "#f87171",   
                        "#1e293b",   
                        "#4fd1c5",  
                        "#0d9488"    
                    ],
                    aspect="auto",
                    title="Correlation Matrix (Numeric Variables)"
                )
                fig.update_layout(
                    width=900,
                    height=600,
                    margin=dict(l=50, r=50, t=50, b=50),
                    coloraxis_colorbar=dict(
                        title="Correlation",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticks="outside"
                    )
                )
                fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with corr_tabs[1]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Bivariate Analysis</h2>", unsafe_allow_html=True)

            cols = df.columns.tolist()
            feature1 = st.selectbox("Select Feature 1", cols)
            feature2 = st.selectbox("Select Feature 2", cols, index=1)
            f1_type = "numeric" if pd.api.types.is_numeric_dtype(df[feature1]) else "categorical"
            f2_type = "numeric" if pd.api.types.is_numeric_dtype(df[feature2]) else "categorical"

            st.write(f"### Visualization for **{feature1}** vs **{feature2}**")

            if f1_type == "numeric" and f2_type == "numeric":
                fig = px.scatter(
                    df, x=feature1, y=feature2, trendline="ols",
                    color_discrete_sequence=[theme_color],
                    title=f"{feature1} vs {feature2}"
                )
                st.plotly_chart(fig, use_container_width=True)


            elif f1_type == "numeric" and f2_type == "categorical":
                fig = px.box(
                    df, x=feature2, y=feature1,
                    color_discrete_sequence=[theme_color],
                    title=f"{feature1} distribution across {feature2}"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif f1_type == "categorical" and f2_type == "numeric":
                fig = px.box(
                    df, x=feature1, y=feature2,
                    color_discrete_sequence=[theme_color],
                    title=f"{feature2} distribution across {feature1}"
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                crosstab = pd.crosstab(df[feature1], df[feature2])
                fig = px.imshow(
                    crosstab,
                    text_auto=True,
                    title=f"Relationship between {feature1} and {feature2}",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Univariate Analysis</h2>", unsafe_allow_html=True)
            cols = df.columns.tolist()
            selected_col = st.selectbox("Select a Column to Analyze", cols)
            st.write(f"## Univariate Analysis of **{selected_col}**")
            col_type = "numeric" if pd.api.types.is_numeric_dtype(df[selected_col]) else "categorical"


            if col_type == "numeric":
                fig = px.histogram(
                    df, x=selected_col, nbins=30, 
                    marginal="box",
                    color_discrete_sequence=[theme_color],
                    title=f"Distribution of {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.write("###  Summary Statistics")
                st.write(df[selected_col].describe())

            else:
                counts = df[selected_col].value_counts()
                fig = px.bar(
                    counts,
                    x=counts.index, 
                    y=counts.values,
                    color_discrete_sequence=[theme_color],
                    title=f"Value Counts for {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.write("### Percentage Distribution")
                st.write(round((counts / counts.sum()) * 100, 2))

            st.markdown(""" 
                        Numeric vs Numeric → Scatter Plot + Trendline 
                        Numeric vs Categorical → Box Plot
                        Categorical vs Categorical → Heatmap
                        
                        Numeric Column → Histogram + Box Plot
                        Categorical Column → Bar Chart
                        """)

            st.markdown("</div>", unsafe_allow_html=True)


        with corr_tabs[2]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Mail vs Freight</h2>", unsafe_allow_html=True)
            if {"year", "total_mail", "total_freight"}.issubset(df.columns):
                yearly_mail = df.groupby("year")["total_mail"].sum()
                yearly_freight = df.groupby("year")["total_freight"].sum()
                combo = pd.DataFrame({"year": yearly_mail.index, "mail": yearly_mail.values, "freight": yearly_freight.values})
                st.plotly_chart(px.line(combo, x="year", y=["mail", "freight"], markers=True))
            st.markdown("</div>", unsafe_allow_html=True)

        with corr_tabs[3]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Accociation rules and dbscan</h2>", unsafe_allow_html=True)
            if st.button("Next ➜"):
                st.switch_page("pages/asso.py")

            st.markdown("</div>", unsafe_allow_html=True)

    with main_tab3:
        model_tabs = st.tabs([
            "Standardization", "PCA (2D)", "PCA Loadings",
            "Clustering", "Cluster Summary"
        ])
        with model_tabs[0]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>Data Standardization Preview</h2>", unsafe_allow_html=True)
            num = df.select_dtypes(include=[np.number]).dropna()
            if num.empty:
                st.warning("No numeric columns available for standardization.")
            else:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(num)
                df_scaled = pd.DataFrame(scaled_data, columns=num.columns)
                st.dataframe(df_scaled.head())
            st.markdown("</div>", unsafe_allow_html=True)
        
        with model_tabs[1]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>PCA (2D Projection)</h2>", unsafe_allow_html=True)
            num = df.select_dtypes(include=[np.number]).dropna()
            if num.shape[1] >= 2:
                X = StandardScaler().fit_transform(num)
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X)
                fig = px.scatter(x=coords[:,0], y=coords[:,1], color_discrete_sequence=[theme_color])
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with model_tabs[2]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>PCA Loadings</h2>", unsafe_allow_html=True)
            num = df.select_dtypes(include=[np.number]).dropna()
            if num.shape[1] >= 2:
                X = StandardScaler().fit_transform(num)
                pca = PCA(n_components=2)
                pca.fit(X)
                loadings = pd.DataFrame(pca.components_.T, index=num.columns, columns=["PC1", "PC2"])
                st.dataframe(loadings)
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        with model_tabs[3]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>K-Means Clustering</h2>", unsafe_allow_html=True)

            features_to_use = ["total_passengers", "total_freight", "total_mail"]
            num = df[features_to_use]

            if num.shape[1] >= 2:
                X = StandardScaler().fit_transform(num)

                method = st.selectbox("Choose Clustering Method", ["K-Means"])

                @st.cache_resource
                def compute_clusters(method, X, k=None, eps=None, min_samples=None):
                    if method == "K-Means":
                        model = KMeans(n_clusters=k, random_state=42)
                        labels = model.fit_predict(X)
                    return labels

                if method == "K-Means":
                    k = st.slider("Number of Clusters (K)", 2, 10, 4)
                    labels = compute_clusters("K-Means", X, k=k)

                pca = PCA(n_components=2)
                coords = pca.fit_transform(X)
                dfp = pd.DataFrame(coords, columns=["PC1", "PC2"])
                dfp["Cluster"] = labels.astype(str)

                fig = px.scatter(
                    dfp,
                    x="PC1", y="PC2",
                    color="Cluster",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    title=f"{method} Clustering Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            

            st.markdown("</div>", unsafe_allow_html=True)

        with model_tabs[4]:
            st.markdown(f"<div class='section'><h2 style='color:{theme_color};'>K-Means Cluster Summary (K-Means)</h2>", unsafe_allow_html=True)

            features_to_use = ["total_passengers", "total_freight", "total_mail"]
            num = df[features_to_use].dropna()

            if num.shape[1] < 2:
                st.warning("Not enough numeric columns for clustering.")
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(num)

                st.subheader("Elbow Method (Inertia)")
                inertias = []
                K_range = range(1, 11)

                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42)
                    km.fit(X)
                    inertias.append(km.inertia_)

                fig_elbow = px.line(
                    x=list(K_range),
                    y=inertias,
                    title="Elbow Curve: Inertia vs K",
                    markers=True,
                    labels={"x": "Number of Clusters (k)", "y": "Inertia"}
                )
                st.plotly_chart(fig_elbow, use_container_width=True)


                # st.subheader("Silhouette Score")
                # from sklearn.metrics import silhouette_score

                # sil_scores = []
                # K_sil = range(2, 6)

                # for k in K_sil:
                #     km = KMeans(n_clusters=k, random_state=42)
                #     labels = km.fit_predict(X)
                #     sil = silhouette_score(X, labels)
                #     sil_scores.append(sil)

                # fig_sil = px.line(
                #     x=list(K_sil),
                #     y=sil_scores,
                #     title="Silhouette Score vs K",
                #     markers=True,
                #     labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"}
                # )
                # st.plotly_chart(fig_sil, use_container_width=True)

                # Optimal k = highest silhouette

                # if st.button("DBSCAN ➜"):
                #     st.switch_page("pages/dbscan.py")

                optimal_k = 3
                st.success(f"Optimal K found by elbow method = **{optimal_k}**")
                km_final = KMeans(n_clusters=optimal_k, random_state=42)
                labels_final = km_final.fit_predict(X)

                df_cluster = pd.DataFrame(X, columns=num.columns)
                df_cluster["Cluster"] = labels_final

                st.subheader("Cluster Summary (Scaled Feature Means)")
                summary = df_cluster.groupby("Cluster").mean().round(2)
                st.dataframe(summary)

                st.subheader("PCA Visualization of Clusters")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                df_plot = pd.DataFrame({
                    "PCA1": X_pca[:, 0],
                    "PCA2": X_pca[:, 1],
                    "Cluster": labels_final.astype(str)
                })

                centers_pca = pca.transform(km_final.cluster_centers_)

                fig_pca = px.scatter(
                    df_plot,
                    x="PCA1",
                    y="PCA2",
                    color="Cluster",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title=f"K-Means Clusters (k = {optimal_k})"
                )

                fig_pca.add_scatter(
                    x=centers_pca[:, 0],
                    y=centers_pca[:, 1],
                    mode="markers",
                    marker=dict(size=15, color="black", symbol="x"),
                    name="Centers"
                )
                st.plotly_chart(fig_pca, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)


   


    