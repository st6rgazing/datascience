## Step 00 - Import of the packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# =============================================================================
# APPLE-INSPIRED DARK STYLING
# =============================================================================
APPLE_CSS = """
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        background: linear-gradient(180deg, #0b0c10 0%, #111318 100%);
        color: #f5f5f7;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Sidebar - clean Apple style */
    [data-testid="stSidebar"] {
        background: rgba(17,19,24,0.92);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #1b1f29 !important;
        color: #f5f5f7 !important;
        border: 1px solid #343A46 !important;
        border-radius: 10px !important;
    }
    [data-baseweb="select"] svg {
        color: #c3c8d2 !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: #007AFF !important;
    }
    
    /* Card containers */
    .apple-card {
        background: #161a22;
        border-radius: 16px;
        padding: 28px;
        margin: 20px 0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.28);
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    /* Headers - Apple typography */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #f5f5f7 !important;
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.5rem !important; margin-top: 2rem !important; }
    h3 { font-size: 1.15rem !important; }
    p, li, label, span {
        color: #c3c8d2;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #007AFF !important;
    }
    
    /* DataFrames - clean table */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.22);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.5rem !important;
        border: none !important;
        background: #007AFF !important;
        color: white !important;
    }
    .stButton > button:hover {
        background: #0051D5 !important;
        box-shadow: 0 4px 12px rgba(0,122,255,0.3) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
        background: #1b1f29;
        color: #c3c8d2;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #242a36 !important;
        color: #f5f5f7 !important;
        border: 1px solid #3a4150 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        border-radius: 10px;
        background: #1b1f29;
    }
    
    /* Success/Warning messages */
    .stSuccess, .stWarning {
        border-radius: 12px;
        padding: 16px 20px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

# Chart styling - Apple-inspired dark look
CHART_STYLE = {
    'figure.facecolor': '#111318',
    'axes.facecolor': '#161a22',
    'axes.edgecolor': '#3a3f4b',
    'axes.labelcolor': '#e5e7eb',
    'axes.titlecolor': '#f5f5f7',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': '#c3c8d2',
    'ytick.color': '#c3c8d2',
    'text.color': '#f5f5f7',
    'grid.color': '#2a2f3a',
    'font.family': 'sans-serif',
    'font.size': 11,
}


def style_axes(ax):
    """Apply consistent dark styling to matplotlib axes."""
    ax.tick_params(colors='#c3c8d2', labelsize=10)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#3a3f4b')
    ax.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.7)

# Apple color palette
APPLE_COLORS = ['#007AFF', '#34C759', '#FF9500', '#FF3B30', '#AF52DE', '#5AC8FA']

# Page configuration
st.set_page_config(
    page_title="Mobile Phone Price Prediction",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(APPLE_CSS, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

df = load_data()

# Sidebar - Apple-style
st.sidebar.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='display: flex; align-items: center; gap: 10px;'>
    <span style='font-size: 28px;'>📱</span>
    <div>
        <div style='font-weight: 600; font-size: 1.1rem; color: #f5f5f7;'>Mobile Price Predictor</div>
        <div style='font-size: 0.75rem; color: #9DA3AE;'>Linear Regression</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<div style='height: 1px; background: #343A46; margin: 16px 0;'></div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigate to",
    ["📘 Business Case & Data", "📊 Data Visualization", "🔮 Model Prediction"],
    label_visibility="collapsed"
)
st.sidebar.markdown("<div style='margin-top: 24px; font-size: 0.85rem; color: #9DA3AE;'>Data Science • Linear Regression</div>", unsafe_allow_html=True)

# =============================================================================
# PAGE 1: Business Case Presentation & Data Presentation
# =============================================================================
if page == "📘 Business Case & Data":
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    # Hero stats banner
    st.markdown(f"""
    <div style='display: flex; gap: 24px; margin-bottom: 24px; padding: 20px 24px; background: linear-gradient(135deg, #161a22 0%, #111318 100%); border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);'>
        <div><span style='font-size: 2rem;'>📊</span><br><span style='font-weight: 600; color: #007AFF;'>{len(df):,}</span><br><span style='font-size: 0.8rem; color: #9DA3AE;'>Phones</span></div>
        <div><span style='font-size: 2rem;'>📋</span><br><span style='font-weight: 600; color: #34C759;'>{len(df.columns)}</span><br><span style='font-size: 0.8rem; color: #9DA3AE;'>Features</span></div>
        <div><span style='font-size: 2rem;'>🎯</span><br><span style='font-weight: 600; color: #FF9500;'>4</span><br><span style='font-size: 0.8rem; color: #9DA3AE;'>Price Ranges</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.title("Mobile Phone Price Range Prediction")
    st.markdown("*Predict price segments from technical specifications*")
    st.markdown("<div style='height: 2px; background: linear-gradient(90deg, #007AFF, transparent); margin: 24px 0; border-radius: 2px;'></div>", unsafe_allow_html=True)
    
    # Business Case Section - Card layout
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="apple-card">
            <h3 style="color: #007AFF; margin-top: 0;">🎯 Problem Statement</h3>
            <p style="color: #c3c8d2; line-height: 1.6; margin-bottom: 0;">
            In the competitive mobile phone market, <strong>retailers and manufacturers</strong> need to understand 
            which specifications drive phone pricing. <strong>Consumers</strong> want to know what 
            features they get at different price points.
            </p>
            <p style="color: #f5f5f7; margin-top: 12px; margin-bottom: 0;">
            <strong>Our Solution:</strong> Build a predictive model using Linear Regression to estimate 
            a phone's price range (0–3) based on its technical specifications.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="apple-card">
            <h3 style="color: #34C759; margin-top: 0;">💡 Business Value</h3>
            <ul style="color: #c3c8d2; line-height: 2; margin-bottom: 0;">
                <li><strong>Retailers:</strong> Optimize inventory and pricing</li>
                <li><strong>Manufacturers:</strong> Align specs with price segments</li>
                <li><strong>Consumers:</strong> Make informed purchase decisions</li>
                <li><strong>Analysts:</strong> Understand feature–price relationships</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Presentation Section
    st.header("2. Data Presentation")
    
    st.markdown(f"""
    <div class="apple-card">
        <h3 style="margin-top: 0;">Dataset Overview</h3>
        <p style="color: #c3c8d2; line-height: 1.6;">
        The dataset contains <strong style="color: #007AFF;">{len(df):,} mobile phones</strong> with 
        <strong>{len(df.columns)} features</strong> describing technical specifications. 
        Target: <strong>price_range</strong> (0 = low, 1 = medium, 2 = high, 3 = very high cost).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 Feature Descriptions", expanded=False):
        feature_desc = pd.DataFrame({
            "Feature": ["battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", 
                       "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height", 
                       "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", 
                       "touch_screen", "wifi", "price_range"],
            "Description": [
                "Battery capacity (mAh)", "Bluetooth (0/1)", "Clock speed (GHz)",
                "Dual SIM (0/1)", "Front camera (MP)", "4G support (0/1)",
                "Internal memory (GB)", "Mobile depth (cm)", "Mobile weight (g)",
                "Number of cores", "Primary camera (MP)", "Pixel resolution height",
                "Pixel resolution width", "RAM (MB)", "Screen height (cm)",
                "Screen width (cm)", "Talk time (hours)", "3G support (0/1)",
                "Touch screen (0/1)", "WiFi (0/1)", "Price range (0-3)"
            ]
        })
        st.dataframe(feature_desc, use_container_width=True, hide_index=True)
    
    st.subheader("Data Preview")
    rows = st.slider("Number of rows to display", 5, 50, 10, key="rows_slider")
    st.dataframe(df.head(rows), use_container_width=True, hide_index=True)
    
    st.subheader("Data Quality")
    missing = df.isnull().sum()
    col1, col2 = st.columns(2)
    with col1:
        if missing.sum() == 0:
            st.success("✅ No missing values — dataset is complete")
        else:
            st.dataframe(missing[missing > 0])
    
    with col2:
        if st.button("Show Summary Statistics"):
            summary_stats = df.describe().reset_index().rename(columns={"index": "Statistic"})
            st.dataframe(summary_stats, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 2: Data Visualization
# =============================================================================
elif page == "📊 Data Visualization":
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display: flex; gap: 16px; margin-bottom: 20px; padding: 16px 20px; background: #161a22; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);'>
        <span style='font-size: 1.5rem;'>📈</span>
        <span style='font-size: 0.9rem; color: #c3c8d2;'>Key Insights • Correlation Analysis • Feature Explorer</span>
    </div>
    """, unsafe_allow_html=True)
    st.title("Data Visualization & Insights")
    st.markdown("*Explore patterns and relationships in the dataset*")
    st.markdown("<div style='height: 2px; background: linear-gradient(90deg, #34C759, transparent); margin: 24px 0; border-radius: 2px;'></div>", unsafe_allow_html=True)
    
    df_numeric = df.select_dtypes(include=[np.number])
    
    tab1, tab2, tab3 = st.tabs(["📈 Key Insights", "🔥 Correlation Analysis", "📊 Feature Explorer"])
    
    with tab1:
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        
        # Price distribution - clean chart
        st.subheader("Price Range Distribution")
        fig1, ax1 = plt.subplots(figsize=(9, 5))
        plt.rcParams.update(CHART_STYLE)
        price_counts = df["price_range"].value_counts().sort_index()
        colors = ['#007AFF', '#34C759', '#FF9500', '#FF3B30']
        bars = ax1.bar(price_counts.index.astype(str), price_counts.values, color=colors, 
                       edgecolor='none', width=0.6)
        ax1.set_xlabel("Price Range (0=Low, 1=Medium, 2=High, 3=Very High)", fontsize=11)
        ax1.set_ylabel("Number of Phones", fontsize=11)
        ax1.set_title("Distribution of Mobile Phones by Price Range", fontsize=13, fontweight='600', pad=16)
        ax1.set_ylim(0, max(price_counts.values) * 1.15)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 12, 
                     str(int(bar.get_height())), ha='center', fontsize=11, fontweight='500', color='#f5f5f7')
        style_axes(ax1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig1)
        plt.close()
        
        st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)
        
        # RAM vs Price
        st.subheader("RAM vs Price Range")
        st.caption("Higher RAM strongly correlates with higher price — key predictor")
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        plt.rcParams.update(CHART_STYLE)
        sns.boxplot(data=df, x="price_range", y="ram", palette=['#007AFF', '#34C759', '#FF9500', '#FF3B30'], ax=ax2)
        ax2.set_xlabel("Price Range", fontsize=11)
        ax2.set_ylabel("RAM (MB)", fontsize=11)
        ax2.set_title("RAM Distribution by Price Range", fontsize=13, fontweight='600', pad=16)
        style_axes(ax2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig2)
        plt.close()
        
        st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)
        
        # Battery vs Price
        st.subheader("Battery Power vs Price Range")
        fig3, ax3 = plt.subplots(figsize=(9, 5))
        plt.rcParams.update(CHART_STYLE)
        sns.violinplot(data=df, x="price_range", y="battery_power", 
                       palette=['#007AFF', '#34C759', '#FF9500', '#FF3B30'], ax=ax3)
        ax3.set_xlabel("Price Range", fontsize=11)
        ax3.set_ylabel("Battery Power (mAh)", fontsize=11)
        ax3.set_title("Battery Capacity by Price Segment", fontsize=13, fontweight='600', pad=16)
        style_axes(ax3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig3)
        plt.close()
    
    with tab2:
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        
        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(12, 9))
        plt.rcParams.update(CHART_STYLE)
        corr_matrix = df_numeric.corr()
        sns.heatmap(corr_matrix, annot=False, cmap="RdBu_r", center=0, 
                    square=True, linewidths=0.5, ax=ax_corr, vmin=-0.5, vmax=0.5)
        ax_corr.set_title("Feature Correlation Matrix", fontsize=13, fontweight='600', pad=16)
        ax_corr.tick_params(colors='#c3c8d2', labelsize=9)
        plt.tight_layout(pad=2)
        st.pyplot(fig_corr)
        plt.close()
        
        st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)
        
        st.subheader("Correlation with Price Range")
        price_corr = df_numeric.corr()["price_range"].drop("price_range").sort_values(ascending=True)
        fig_bar, ax_bar = plt.subplots(figsize=(9, 6))
        plt.rcParams.update(CHART_STYLE)
        price_corr.plot(kind="barh", ax=ax_bar, color='#007AFF', edgecolor='none')
        ax_bar.axvline(x=0, color='#c3c8d2', linewidth=0.5)
        ax_bar.set_xlabel("Correlation with Price Range", fontsize=11)
        ax_bar.set_title("Features Most Correlated with Price Range", fontsize=13, fontweight='600', pad=16)
        style_axes(ax_bar)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig_bar)
        plt.close()
    
    with tab3:
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        col_x = st.selectbox("X-axis variable", df_numeric.columns.tolist(), index=0, key="col_x")
        col_y = st.selectbox("Y-axis variable", df_numeric.columns.tolist(), index=13, key="col_y")
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(9, 6))
        plt.rcParams.update(CHART_STYLE)
        scatter = ax_scatter.scatter(df[col_x], df[col_y], c=df["price_range"], 
                                    cmap="viridis", alpha=0.7, s=50, edgecolors='#1b1f29', linewidth=0.4)
        ax_scatter.set_xlabel(col_x, fontsize=11)
        ax_scatter.set_ylabel(col_y, fontsize=11)
        ax_scatter.set_title(f"{col_x} vs {col_y} (colored by price range)", fontsize=13, fontweight='600', pad=16)
        cbar = plt.colorbar(scatter, ax=ax_scatter, label="Price Range")
        cbar.ax.yaxis.label.set_color('#e5e7eb')
        cbar.ax.tick_params(colors='#c3c8d2')
        cbar.outline.set_edgecolor('#3a3f4b')
        style_axes(ax_scatter)
        ax_scatter.spines['top'].set_visible(False)
        ax_scatter.spines['right'].set_visible(False)
        plt.tight_layout(pad=2)
        st.pyplot(fig_scatter)
        plt.close()

# =============================================================================
# PAGE 3: Model Prediction
# =============================================================================
elif page == "🔮 Model Prediction":
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display: flex; gap: 16px; margin-bottom: 20px; padding: 16px 20px; background: linear-gradient(135deg, #21192e 0%, #161a22 100%); border-radius: 12px; border-left: 4px solid #AF52DE; border: 1px solid rgba(255,255,255,0.08);'>
        <span style='font-size: 1.5rem;'>🔮</span>
        <span style='font-size: 0.9rem; color: #c3c8d2;'>Scikit-Learn Linear Regression • Configurable features • Live metrics</span>
    </div>
    """, unsafe_allow_html=True)
    st.title("Linear Regression Model")
    st.markdown("*Predict price range from phone specifications*")
    st.markdown("<div style='height: 2px; background: linear-gradient(90deg, #AF52DE, transparent); margin: 24px 0; border-radius: 2px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="apple-card">
        <p style="color: #c3c8d2; line-height: 1.6; margin: 0;">
        Uses <strong>Scikit-Learn's Linear Regression</strong> to predict price range. 
        Select features in the sidebar and evaluate model performance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df_model = df.dropna()
    feature_cols = [c for c in df_model.columns if c != "price_range"]
    
    st.sidebar.markdown("<div style='height: 1px; background: #343A46; margin: 16px 0;'></div>", unsafe_allow_html=True)
    st.sidebar.subheader("Model Configuration")
    features_selection = st.sidebar.multiselect(
        "Select Features (X)", 
        feature_cols, 
        default=["battery_power", "ram", "px_height", "px_width", "int_memory"]
    )
    
    if len(features_selection) < 1:
        st.warning("⚠️ Please select at least one feature from the sidebar.")
    else:
        X = df_model[features_selection]
        y = df_model["price_range"]
        X = X.select_dtypes(include=[np.number])
        
        if X.empty:
            st.error("Selected features contain non-numeric data.")
        else:
            test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, key="test_size")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            predictions_clipped = np.clip(predictions, 0, 3)
            
            # Metrics in cards
            st.subheader("Model Performance")
            mse = metrics.mean_squared_error(y_test, predictions_clipped)
            mae = metrics.mean_absolute_error(y_test, predictions_clipped)
            r2 = metrics.r2_score(y_test, predictions_clipped)
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            with m2:
                st.metric("Mean Absolute Error", f"{mae:.4f}")
            with m3:
                st.metric("R² Score", f"{r2:.4f}")
            
            st.success(f"Model trained successfully — average error: ±{mae:.2f} price range units")
            
            st.subheader("Feature Coefficients")
            coef_df = pd.DataFrame({
                "Feature": features_selection,
                "Coefficient": model.coef_
            })
            coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            
            st.subheader("Actual vs Predicted")
            fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
            plt.rcParams.update(CHART_STYLE)
            ax_pred.scatter(y_test, predictions_clipped, alpha=0.6, s=60, c='#007AFF', 
                           edgecolors='#1b1f29', linewidth=0.5)
            min_val = min(y_test.min(), predictions_clipped.min())
            max_val = max(y_test.max(), predictions_clipped.max())
            ax_pred.plot([min_val, max_val], [min_val, max_val], "--", color='#FF3B30', 
                        linewidth=2, label="Perfect prediction")
            ax_pred.set_xlabel("Actual Price Range", fontsize=11)
            ax_pred.set_ylabel("Predicted Price Range", fontsize=11)
            ax_pred.set_title("Actual vs Predicted Price Range", fontsize=13, fontweight='600', pad=16)
            style_axes(ax_pred)
            legend = ax_pred.legend(loc='lower right', frameon=True, facecolor='#161a22', edgecolor='#3a3f4b')
            for text in legend.get_texts():
                text.set_color('#e5e7eb')
            ax_pred.spines['top'].set_visible(False)
            ax_pred.spines['right'].set_visible(False)
            plt.tight_layout(pad=2)
            st.pyplot(fig_pred)
            plt.close()
            
            st.subheader("Residuals Distribution")
            residuals = y_test.values - predictions_clipped
            fig_res, ax_res = plt.subplots(figsize=(8, 5))
            plt.rcParams.update(CHART_STYLE)
            ax_res.hist(residuals, bins=28, color='#007AFF', edgecolor='#1b1f29', alpha=0.8)
            ax_res.axvline(x=0, color='#FF3B30', linestyle='--', linewidth=2)
            ax_res.set_xlabel("Residual (Actual − Predicted)", fontsize=11)
            ax_res.set_ylabel("Frequency", fontsize=11)
            ax_res.set_title("Distribution of Prediction Errors", fontsize=13, fontweight='600', pad=16)
            style_axes(ax_res)
            ax_res.spines['top'].set_visible(False)
            ax_res.spines['right'].set_visible(False)
            plt.tight_layout(pad=2)
            st.pyplot(fig_res)
            plt.close()
