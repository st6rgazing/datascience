## Step 00 - Import of the packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Page configuration
st.set_page_config(
    page_title="Mobile Phone Price Prediction 📱",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="expanded",
)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("📱 Mobile Price Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to",
    ["📘 Business Case & Data", "📊 Data Visualization", "🔮 Model Prediction"],
    label_visibility="collapsed"
)

# =============================================================================
# PAGE 1: Business Case Presentation & Data Presentation
# =============================================================================
if page == "📘 Business Case & Data":
    st.title("Mobile Phone Price Range Prediction")
    st.markdown("---")
    
    # Business Case Section
    st.header("1. Business Case Presentation")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🎯 Problem Statement")
        st.markdown("""
        In the competitive mobile phone market, **retailers and manufacturers** need to understand 
        which specifications drive phone pricing. Similarly, **consumers** want to know what 
        features they get at different price points.
        
        **Our Solution:** Build a predictive model using Linear Regression to estimate 
        a phone's price range (0-3) based on its technical specifications.
        """)
    
    with col2:
        st.subheader("💡 Business Value")
        st.markdown("""
        - **Retailers:** Optimize inventory and pricing strategies
        - **Manufacturers:** Align product specs with target price segments  
        - **Consumers:** Make informed purchase decisions
        - **Market analysts:** Understand feature-price relationships
        """)
    
    st.markdown("---")
    
    # Data Presentation Section
    st.header("2. Data Presentation")
    
    st.subheader("Dataset Overview")
    st.markdown(f"""
    The dataset contains **{len(df):,} mobile phones** with **{len(df.columns)} features** 
    describing technical specifications. The target variable is **price_range** (0 = low cost, 
    1 = medium cost, 2 = high cost, 3 = very high cost).
    """)
    
    # Feature descriptions
    with st.expander("📋 Feature Descriptions"):
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
    
    # Data preview
    st.subheader("Data Preview")
    rows = st.slider("Number of rows to display", 5, 50, 10)
    st.dataframe(df.head(rows), use_container_width=True, hide_index=True)
    
    # Missing values
    st.subheader("Data Quality")
    missing = df.isnull().sum()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Missing values per column:**")
        if missing.sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.dataframe(missing[missing > 0])
    
    with col2:
        st.markdown("**Summary Statistics:**")
        if st.button("Show Describe Table"):
            st.dataframe(df.describe(), use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 2: Data Visualization
# =============================================================================
elif page == "📊 Data Visualization":
    st.title("Data Visualization & Insights")
    st.markdown("---")
    
    # Select numeric columns (exclude id if present)
    df_numeric = df.select_dtypes(include=[np.number])
    
    tab1, tab2, tab3 = st.tabs(["📈 Key Insights", "🔥 Correlation Analysis", "📊 Feature Distributions"])
    
    with tab1:
        st.subheader("Price Range Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        price_counts = df["price_range"].value_counts().sort_index()
        colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
        bars = ax1.bar(price_counts.index.astype(str), price_counts.values, color=colors)
        ax1.set_xlabel("Price Range (0=Low, 1=Medium, 2=High, 3=Very High)")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of Mobile Phones by Price Range")
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                     str(int(bar.get_height())), ha='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
        st.subheader("RAM vs Price Range (Key Driver)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="price_range", y="ram", palette="viridis", ax=ax2)
        ax2.set_xlabel("Price Range")
        ax2.set_ylabel("RAM (MB)")
        ax2.set_title("RAM Distribution by Price Range - Higher RAM correlates with higher price")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
        st.subheader("Battery Power vs Price Range")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x="price_range", y="battery_power", palette="muted", ax=ax3)
        ax3.set_xlabel("Price Range")
        ax3.set_ylabel("Battery Power (mAh)")
        ax3.set_title("Battery Capacity by Price Segment")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    
    with tab2:
        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
        corr_matrix = df_numeric.corr()
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, 
                    square=True, linewidths=0.5, ax=ax_corr)
        ax_corr.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close()
        
        st.subheader("Correlation with Price Range")
        price_corr = df_numeric.corr()["price_range"].drop("price_range").sort_values(ascending=False)
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        price_corr.plot(kind="barh", ax=ax_bar, color="steelblue")
        ax_bar.set_xlabel("Correlation with Price Range")
        ax_bar.set_title("Features Most Correlated with Price Range")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close()
    
    with tab3:
        col_x = st.selectbox("Select X-axis variable", df_numeric.columns.tolist(), index=0)
        col_y = st.selectbox("Select Y-axis variable", df_numeric.columns.tolist(), index=13)
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        scatter = ax_scatter.scatter(df[col_x], df[col_y], c=df["price_range"], 
                                    cmap="viridis", alpha=0.6)
        ax_scatter.set_xlabel(col_x)
        ax_scatter.set_ylabel(col_y)
        ax_scatter.set_title(f"{col_x} vs {col_y} (colored by price range)")
        plt.colorbar(scatter, ax=ax_scatter, label="Price Range")
        plt.tight_layout()
        st.pyplot(fig_scatter)
        plt.close()

# =============================================================================
# PAGE 3: Model Prediction
# =============================================================================
elif page == "🔮 Model Prediction":
    st.title("Linear Regression Model - Price Range Prediction")
    st.markdown("---")
    
    st.markdown("""
    This page uses **Scikit-Learn's Linear Regression** to predict the price range of mobile phones 
    based on their specifications. Select features and evaluate model performance.
    """)
    
    # Data preprocessing
    df_model = df.dropna()
    
    # Feature selection
    feature_cols = [c for c in df_model.columns if c != "price_range"]
    
    st.sidebar.markdown("---")
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
        
        # Handle any non-numeric (shouldn't exist in this dataset)
        X = X.select_dtypes(include=[np.number])
        
        if X.empty:
            st.error("Selected features contain non-numeric data. Please select numeric columns.")
        else:
            # Train-test split
            test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Model training
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Clip predictions to valid range [0, 3]
            predictions_clipped = np.clip(predictions, 0, 3)
            
            # Metrics
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            
            mse = metrics.mean_squared_error(y_test, predictions_clipped)
            mae = metrics.mean_absolute_error(y_test, predictions_clipped)
            r2 = metrics.r2_score(y_test, predictions_clipped)
            
            with col1:
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            with col2:
                st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            st.success(f"✅ Model trained successfully. Average prediction error: ±{mae:.2f} price range units")
            
            # Feature coefficients
            st.subheader("Feature Coefficients (Model Interpretability)")
            coef_df = pd.DataFrame({
                "Feature": features_selection,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", key=abs, ascending=False)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            
            # Actual vs Predicted plot
            st.subheader("Actual vs Predicted Values")
            fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
            ax_pred.scatter(y_test, predictions_clipped, alpha=0.5, edgecolors="k", linewidth=0.5)
            min_val = min(y_test.min(), predictions_clipped.min())
            max_val = max(y_test.max(), predictions_clipped.max())
            ax_pred.plot([min_val, max_val], [min_val, max_val], "--r", linewidth=2, label="Perfect prediction")
            ax_pred.set_xlabel("Actual Price Range")
            ax_pred.set_ylabel("Predicted Price Range")
            ax_pred.set_title("Actual vs Predicted Price Range")
            ax_pred.legend()
            ax_pred.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_pred)
            plt.close()
            
            # Residuals distribution
            st.subheader("Residuals Distribution")
            residuals = y_test.values - predictions_clipped
            fig_res, ax_res = plt.subplots(figsize=(8, 5))
            ax_res.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
            ax_res.axvline(x=0, color="r", linestyle="--")
            ax_res.set_xlabel("Residual (Actual - Predicted)")
            ax_res.set_ylabel("Frequency")
            ax_res.set_title("Distribution of Prediction Errors")
            plt.tight_layout()
            st.pyplot(fig_res)
            plt.close()
