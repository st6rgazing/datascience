"""
Generate a PDF presentation for the Mobile Phone Price Prediction project.
Apple-inspired design with clean typography, graphics, and no overlap.
Run: python generate_presentation.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib.backends.backend_pdf import PdfPages

# Apple-inspired color palette
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_ORANGE = '#FF9500'
APPLE_RED = '#FF3B30'
APPLE_PURPLE = '#AF52DE'
APPLE_GRAY = '#8E8E93'
APPLE_LIGHT_GRAY = '#F5F5F7'
APPLE_DARK = '#1d1d1f'

# Set style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#d2d2d7',
    'axes.labelcolor': APPLE_DARK,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, "train.csv"))

def draw_phone_icon(ax, x, y, size=0.6, color=APPLE_BLUE):
    """Draw a simple phone icon."""
    # Phone body (rounded rectangle)
    body = FancyBboxPatch((x-size*0.4, y-size*0.8), size*0.8, size*1.6, 
                          boxstyle="round,pad=0.02", facecolor=color, 
                          edgecolor='none', alpha=0.9)
    ax.add_patch(body)
    # Screen
    screen = Rectangle((x-size*0.3, y+size*0.2), size*0.6, size*0.9, 
                       facecolor='white', edgecolor='none', alpha=0.3)
    ax.add_patch(screen)

def create_title_page():
    """Apple-style title page with graphics."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')
    
    # Decorative background elements
    ax.add_patch(Circle((1, 11), 0.8, facecolor=APPLE_BLUE, alpha=0.08))
    ax.add_patch(Circle((9, 1), 1.2, facecolor=APPLE_PURPLE, alpha=0.06))
    ax.add_patch(Circle((9, 11), 0.5, facecolor=APPLE_GREEN, alpha=0.08))
    
    # Phone icons
    draw_phone_icon(ax, 3, 8, 0.8, APPLE_BLUE)
    draw_phone_icon(ax, 5, 8.2, 0.9, APPLE_PURPLE)
    draw_phone_icon(ax, 7, 8, 0.8, APPLE_GREEN)
    
    # Title
    ax.text(5, 6.2, 'Mobile Phone Price', fontsize=32, fontweight='bold', ha='center', color=APPLE_DARK)
    ax.text(5, 5.6, 'Range Prediction', fontsize=32, fontweight='bold', ha='center', color=APPLE_DARK)
    ax.text(5, 4.5, 'Streamlit App with Linear Regression', fontsize=14, ha='center', color=APPLE_GRAY)
    ax.text(5, 3.8, 'Data Science Group Project', fontsize=12, ha='center', color=APPLE_GRAY)
    
    # Tech stack pills
    tech = 'Streamlit  •  Pandas  •  NumPy  •  Seaborn  •  Matplotlib  •  Scikit-Learn'
    ax.text(5, 2.5, tech, fontsize=10, ha='center', color=APPLE_GRAY, style='italic')
    
    # Bottom accent line
    ax.axhline(y=1.5, xmin=0.1, xmax=0.9, color=APPLE_BLUE, linewidth=3, alpha=0.5)
    
    return fig

def create_section_page(title, content_lines, accent_color=APPLE_BLUE):
    """Create a text section page with clean layout."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')
    
    # Title with accent
    ax.text(5, 11, title, fontsize=22, fontweight='bold', ha='center', color=APPLE_DARK)
    ax.axhline(y=10.5, xmin=0.15, xmax=0.85, color=accent_color, linewidth=2, alpha=0.6)
    
    # Content with generous spacing
    y = 9.5
    for line in content_lines:
        if line.startswith('**'):
            ax.text(0.8, y, line.replace('**', ''), fontsize=12, fontweight='bold', color=APPLE_DARK)
            y -= 0.6
        elif line.startswith('•'):
            ax.text(1, y, line, fontsize=11, color=APPLE_GRAY)
            y -= 0.5
        elif line.startswith('○'):
            ax.text(1, y, line[1:], fontsize=11, color=APPLE_GRAY)
            y -= 0.5
        elif line.startswith('---'):
            ax.axhline(y=y-0.2, xmin=0.1, xmax=0.9, color='#e5e5ea', linewidth=1)
            y -= 0.6
        else:
            ax.text(0.8, y, line, fontsize=11, color=APPLE_GRAY, wrap=True)
            y -= 0.55
    
    return fig

def create_app_architecture_diagram():
    """Clean architecture diagram with Apple colors."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')
    
    ax.text(5, 7.5, 'App Architecture', fontsize=18, fontweight='bold', ha='center', color=APPLE_DARK)
    ax.axhline(y=7.1, xmin=0.2, xmax=0.8, color=APPLE_BLUE, linewidth=1.5, alpha=0.5)
    
    # Spacious box layout - generous spacing to prevent overlap
    boxes = [
        (5, 5.8, 3.5, 0.9, 'Data Loading\nPandas + NumPy', APPLE_BLUE),
        (1.8, 3.6, 2.2, 1.1, 'Page 1\nBusiness Case\n& Data', APPLE_GREEN),
        (5, 3.6, 2.2, 1.1, 'Page 2\nVisualization\n(Seaborn/Matplotlib)', APPLE_PURPLE),
        (8.2, 3.6, 2.2, 1.1, 'Page 3\nPrediction\n(Scikit-Learn)', APPLE_ORANGE),
        (5, 1.5, 4, 0.9, 'Linear Regression → Price Range (0-3)', APPLE_RED),
    ]
    
    for x, y, w, h, text, color in boxes:
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08", 
                             facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=9, ha='center', va='center', fontweight='600', color='white')
    
    # Arrows - adjusted for new box positions
    ax.annotate('', xy=(5, 5.35), xytext=(5, 5.8), 
                arrowprops=dict(arrowstyle='->', lw=2, color=APPLE_GRAY))
    ax.annotate('', xy=(2.9, 4.15), xytext=(4.2, 5.25), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color=APPLE_GRAY))
    ax.annotate('', xy=(5, 4.15), xytext=(5, 5.25), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color=APPLE_GRAY))
    ax.annotate('', xy=(7.1, 4.15), xytext=(5.8, 5.25), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color=APPLE_GRAY))
    ax.annotate('', xy=(5, 1.95), xytext=(5, 3.05), 
                arrowprops=dict(arrowstyle='->', lw=2, color=APPLE_GRAY))
    
    return fig

def create_data_pipeline_diagram():
    """Clean horizontal pipeline."""
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')
    
    ax.text(5, 4.2, 'Data Pipeline', fontsize=16, fontweight='bold', ha='center', color=APPLE_DARK)
    ax.axhline(y=3.7, xmin=0.2, xmax=0.8, color=APPLE_GREEN, linewidth=1.5, alpha=0.5)
    
    stages = [
        (1.5, 2.2, 'train.csv\n2,000 phones', APPLE_BLUE),
        (3.5, 2.2, 'Load & Clean\nPandas', APPLE_GREEN),
        (5.5, 2.2, 'Feature\nSelection', APPLE_PURPLE),
        (7.5, 2.2, 'Train/Test\n80/20', APPLE_ORANGE),
        (9, 2.2, 'Linear\nRegression', APPLE_RED),
    ]
    
    for i, (x, y, text, color) in enumerate(stages):
        box = FancyBboxPatch((x-0.65, y-0.5), 1.3, 1, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=9, ha='center', va='center', fontweight='600', color='white')
        if i < len(stages) - 1:
            ax.annotate('', xy=(x+0.85, y), xytext=(x+0.7, y), 
                       arrowprops=dict(arrowstyle='->', lw=2, color=APPLE_GRAY))
    
    return fig

def create_chart(figsize=(8, 5), title=None):
    """Create chart with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    if title:
        ax.set_title(title, fontsize=14, fontweight='600', pad=20, color=APPLE_DARK)
    ax.tick_params(colors=APPLE_GRAY)
    return fig, ax

def create_price_distribution_chart():
    """Price distribution with Apple colors."""
    fig, ax = create_chart((8, 5), title='Distribution of Mobile Phones by Price Range')
    price_counts = df['price_range'].value_counts().sort_index()
    colors = [APPLE_BLUE, APPLE_GREEN, APPLE_ORANGE, APPLE_RED]
    bars = ax.bar(price_counts.index.astype(str), price_counts.values, color=colors, 
                  edgecolor='none', width=0.65)
    ax.set_xlabel('Price Range (0=Low, 1=Medium, 2=High, 3=Very High)', fontsize=11, color=APPLE_GRAY)
    ax.set_ylabel('Number of Phones', fontsize=11, color=APPLE_GRAY)
    ax.set_ylim(0, max(price_counts.values) * 1.2)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 18, 
                str(int(bar.get_height())), ha='center', fontsize=11, fontweight='600')
    plt.tight_layout(pad=2)
    return fig

def create_ram_vs_price_chart():
    """RAM vs price box plot."""
    fig, ax = create_chart((8, 5.5), title='RAM by Price Range — Key Predictor')
    sns.boxplot(data=df, x='price_range', y='ram', palette=[APPLE_BLUE, APPLE_GREEN, APPLE_ORANGE, APPLE_RED], ax=ax)
    ax.set_xlabel('Price Range', fontsize=11, color=APPLE_GRAY)
    ax.set_ylabel('RAM (MB)', fontsize=11, color=APPLE_GRAY)
    plt.tight_layout(pad=2)
    return fig

def create_battery_vs_price_chart():
    """Battery vs price violin plot."""
    fig, ax = create_chart((8, 5.5), title='Battery Capacity by Price Segment')
    sns.violinplot(data=df, x='price_range', y='battery_power', 
                   palette=[APPLE_BLUE, APPLE_GREEN, APPLE_ORANGE, APPLE_RED], ax=ax)
    ax.set_xlabel('Price Range', fontsize=11, color=APPLE_GRAY)
    ax.set_ylabel('Battery Power (mAh)', fontsize=11, color=APPLE_GRAY)
    plt.tight_layout(pad=2)
    return fig

def create_correlation_heatmap():
    """Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    df_numeric = df.select_dtypes(include=[np.number])
    sns.heatmap(df_numeric.corr(), annot=False, cmap='RdBu_r', center=0, 
                square=True, linewidths=0.5, ax=ax, vmin=-0.5, vmax=0.5)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='600', pad=20, color=APPLE_DARK)
    plt.tight_layout(pad=2)
    return fig

def create_model_performance_chart():
    """Actual vs predicted."""
    X = df[['battery_power', 'ram', 'px_height', 'px_width', 'int_memory']]
    y = df['price_range']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0, 3)
    
    fig, ax = create_chart((8, 6), title='Model Performance: Actual vs Predicted')
    ax.scatter(y_test, predictions, alpha=0.6, s=60, c=APPLE_BLUE, edgecolors='white', linewidth=0.5)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=APPLE_RED, linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Price Range', fontsize=11, color=APPLE_GRAY)
    ax.set_ylabel('Predicted Price Range', fontsize=11, color=APPLE_GRAY)
    ax.legend(loc='lower right', frameon=False)
    
    mae = metrics.mean_absolute_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)
    ax.text(0.05, 0.95, f'MAE: {mae:.3f}\nR²: {r2:.3f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor=APPLE_LIGHT_GRAY, edgecolor='none', alpha=0.9))
    plt.tight_layout(pad=2)
    return fig

def create_feature_importance_chart():
    """Feature coefficients."""
    X = df[['battery_power', 'ram', 'px_height', 'px_width', 'int_memory']]
    y = df['price_range']
    model = LinearRegression()
    model.fit(X, y)
    
    fig, ax = create_chart((8, 5), title='Feature Importance — Coefficient Impact')
    coefs = pd.Series(model.coef_, index=X.columns)
    coefs = coefs.reindex(coefs.abs().sort_values().index)
    colors = [APPLE_RED if c < 0 else APPLE_GREEN for c in coefs.values]
    coefs.plot(kind='barh', ax=ax, color=colors, edgecolor='none')
    ax.axvline(x=0, color=APPLE_DARK, linestyle='-', linewidth=0.5)
    ax.set_xlabel('Coefficient (Impact on Price Range)', fontsize=11, color=APPLE_GRAY)
    plt.tight_layout(pad=2)
    return fig

def create_pdf():
    """Generate the full PDF."""
    output_path = os.path.join(SCRIPT_DIR, "Mobile_Phone_Price_Prediction_Presentation.pdf")
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title
        fig = create_title_page()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 2: Business Case
        business_text = [
            "**Problem Statement**",
            "In the competitive mobile phone market, retailers and manufacturers need to",
            "understand which specifications drive phone pricing. Consumers want to know",
            "what features they get at different price points.",
            "",
            "**Our Solution:** Build a predictive model using Linear Regression to estimate",
            "a phone's price range (0-3) based on its technical specifications.",
            "---",
            "**Business Value**",
            "• Retailers: Optimize inventory and pricing strategies",
            "• Manufacturers: Align product specs with target price segments",
            "• Consumers: Make informed purchase decisions",
            "• Market analysts: Understand feature-price relationships",
        ]
        fig = create_section_page('1. Business Case Presentation', business_text, APPLE_BLUE)
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 3: Architecture
        fig = create_app_architecture_diagram()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 4: Data Presentation
        data_text = [
            f"The dataset contains {len(df):,} mobile phones with {len(df.columns)} features.",
            "Target variable: price_range (0=low, 1=medium, 2=high, 3=very high cost)",
            "Features: battery_power, RAM, camera specs, screen size, connectivity, etc.",
            "Data quality: No missing values — complete dataset",
        ]
        fig = create_section_page('2. Data Presentation', data_text, APPLE_GREEN)
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 5: Data Pipeline
        fig = create_data_pipeline_diagram()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 6: Price distribution
        fig = create_price_distribution_chart()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 7: RAM vs Price
        fig = create_ram_vs_price_chart()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 7b: Battery vs Price
        fig = create_battery_vs_price_chart()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 9: Correlation
        fig = create_correlation_heatmap()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 10: Feature importance
        fig = create_feature_importance_chart()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 11: Model performance
        fig = create_model_performance_chart()
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 12: Conclusion
        conclusion_text = [
            "**Summary**",
            "This project demonstrates a complete data science workflow: from business",
            "problem definition, through data exploration and visualization, to predictive",
            "modeling. The Streamlit app provides an interactive interface for stakeholders.",
            "---",
            "**Key Takeaways**",
            "• RAM is the strongest predictor of price range",
            "• Linear Regression provides interpretable coefficients",
            "• Model helps retailers and consumers make data-driven decisions",
            "---",
            "**Tech Stack**",
            "Streamlit • Pandas • NumPy • Seaborn • Matplotlib • Scikit-Learn",
        ]
        fig = create_section_page('5. Conclusion', conclusion_text, APPLE_PURPLE)
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"✅ PDF created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_pdf()
