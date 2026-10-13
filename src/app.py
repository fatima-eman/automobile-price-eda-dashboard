import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration & Theme
st.set_page_config(
    page_title="Automobile Price Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# 2. Data Loading Function with Caching
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/cleaned_vehicles.csv')

df = load_data()

# 3. Sidebar Filters
st.sidebar.title("🔍 Data Filters")

# Manufacturer Filter
makes = ['All'] + sorted(df['make'].dropna().unique().tolist())
selected_make = st.sidebar.selectbox("Select Manufacturer", makes)

# Fuel Type Filter
fuel_types = ['All'] + sorted(df['fuel-type'].dropna().unique().tolist())
selected_fuel = st.sidebar.selectbox("Select Fuel Type", fuel_types)

# Price Range Slider
min_price, max_price = int(df['price'].min()), int(df['price'].max())
selected_price = st.sidebar.slider(
    "Select Price Range ($)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# Apply Filtering Logic
filtered_df = df.copy()

if selected_make != 'All':
    filtered_df = filtered_df[filtered_df['make'] == selected_make]

if selected_fuel != 'All':
    filtered_df = filtered_df[filtered_df['fuel-type'] == selected_fuel]

filtered_df = filtered_df[
    (filtered_df['price'] >= selected_price[0]) & 
    (filtered_df['price'] <= selected_price[1])
]

# 4. Header & Top-Level KPI Summary
st.title("🚗 Automobile Price Exploratory Analytics Dashboard")
st.markdown("Interactively analyze price drivers, engine specifications, and correlation patterns across the UCI Automobile Dataset.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Vehicles", len(filtered_df))
col2.metric("Average Price", f"${filtered_df['price'].mean():,.0f}" if len(filtered_df) > 0 else "$0")
col3.metric("Avg Horsepower", f"{filtered_df['horsepower'].mean():.0f} HP" if len(filtered_df) > 0 else "0 HP")
col4.metric("Avg Highway MPG", f"{filtered_df['highway-mpg'].mean():.1f}" if len(filtered_df) > 0 else "0")

st.divider()

# 5. Multi-Tab Navigation
tab1, tab2, tab3 = st.tabs(["📊 Distributions & Raw Data", "📈 Bivariate Analysis", "🔥 Correlation Matrix"])

# --- TAB 1: Distributions & Raw Data ---
with tab1:
    st.subheader("Filtered Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.subheader("Univariate Distribution Explorer")
    dist_col = st.selectbox(
        "Select Feature for Distribution Plot", 
        ['price', 'horsepower', 'highway-mpg', 'engine-size', 'curb-weight']
    )
    
    if len(filtered_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(filtered_df[dist_col], kde=True, bins='auto', color='steelblue', ax=ax)
        ax.set_title(f"Distribution of {dist_col.capitalize()}")
        st.pyplot(fig)
    else:
        st.warning("No vehicles match the selected filter criteria.")

# --- TAB 2: Bivariate Relationships ---
with tab2:
    st.subheader("Driver Relationships vs. Price")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**Continuous Drivers (Scatter Plot)**")
        num_feature = st.selectbox(
            "Select Continuous X-Axis Variable", 
            ['horsepower', 'engine-size', 'curb-weight', 'highway-mpg', 'city-mpg']
        )
        if len(filtered_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.regplot(
                data=filtered_df, x=num_feature, y='price',
                scatter_kws={'alpha': 0.6, 'color': 'steelblue'},
                line_kws={'color': 'darkred'}, ax=ax
            )
            ax.set_title(f"{num_feature.capitalize()} vs. Price")
            st.pyplot(fig)
            
    with col_right:
        st.markdown("**Categorical Drivers (Box Plot)**")
        cat_feature = st.selectbox(
            "Select Categorical Category", 
            ['body-style', 'drive-wheels', 'num-of-cylinders', 'fuel-system', 'engine-location']
        )
        if len(filtered_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            order = filtered_df.groupby(cat_feature)['price'].median().sort_values(ascending=False).index
            sns.boxplot(data=filtered_df, x=cat_feature, y='price', order=order, palette='viridis', ax=ax)
            ax.set_title(f"Price Distribution by {cat_feature.capitalize()}")
            st.pyplot(fig)

# --- TAB 3: Feature Correlation Heatmap ---
with tab3:
    st.subheader("Multivariate Feature Correlation Heatmap")
    
    all_num = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    discrete_cols = ['num-of-doors', 'symboling', 'num-of-cylinders']
    continuous_cols = [c for c in all_num if c not in discrete_cols]
    
    if len(filtered_df) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = filtered_df[continuous_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title("Pearson Correlation Coefficients")
        st.pyplot(fig)
    else:
        st.warning("Insufficient data selected to calculate correlation matrix.")