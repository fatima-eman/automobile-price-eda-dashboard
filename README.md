# 🚗 Automobile Price Analytics & Exploratory Data Analysis

An end-to-end data science project performing statistical Exploratory Data Analysis (EDA) on the classic **UCI Automobile Dataset** (ID 10) to uncover key price drivers, followed by an interactive **Streamlit Analytics Dashboard** deployed for interactive exploration.

---

## 📌 Project Overview

Understanding the primary determinants of vehicle pricing is crucial for automotive pricing strategy, inventory valuation, and consumer intelligence. This project builds a production-grade EDA pipeline and web application using Python to analyze mechanical specifications, fuel economy, and categorical features against vehicle price.

### Key Analytical Findings

* **Engine Size & Horsepower:** Highly positive linear correlation with price. Larger engines and higher horsepower serve as the single strongest numerical drivers of vehicle cost.


* **Fuel Economy Trade-Off:** Highway and city MPG exhibit a strong **negative correlation** with vehicle price. High-priced luxury/performance vehicles sacrifice fuel efficiency, whereas budget cars prioritize high MPG.


* **Drive Wheel Influence:** Rear-Wheel Drive (`rwd`) vehicles command a substantially higher median price compared to Front-Wheel Drive (`fwd`) and All-Wheel Drive (`4wd`) alternatives.
* **Multicollinearity:** Features such as `city-mpg` and `highway-mpg`, as well as `curb-weight` and `engine-size`, demonstrate strong inter-feature correlation.



---

## 📁 Repository Architecture

```text
automobile-price-eda-dashboard/
├── data/                      # Excluded from version control (Git-ignored)
│   ├── raw/                   # Immutable raw data (raw_vehicles.csv)
│   └── processed/             # Cleaned analytical dataset (cleaned_vehicles.csv)
├── notebooks/                 # Sequential exploratory notebooks
│   ├── 01_data_ingestion_and_cleaning.ipynb
│   └── 02_exploratory_data_analysis.ipynb
│   └── 03_feature_engineering.ipynb
├── reports/                   # Visual output artifacts
│   └── figures/               # High-resolution exported EDA charts
│       ├── continuous_distributions.png
│       ├── categorical_distributions.png
│       ├── bivariate_continuous_vs_price.png
│       ├── bivariate_categorical_vs_price.png
│       └── correlation_heatmap.png
├── src/                       # Production application source code
│   └── app.py                 # Streamlit multi-tab dashboard script
├── .gitignore                 # Enforces data privacy and environment exclusion
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project documentation

```

---

## 🧹 Data Engineering & Preprocessing

1. **Ingestion:** Fetched programmatically using `ucimlrepo` directly from the UCI repository.


2. **Missing Value Imputation:**
* Numerical columns imputed using the **median** (resilient to extreme skewness).


* Categorical features imputed using the **mode** (most frequent category).




3. **Data Type Casting:** Text placeholders and legacy objects coerced into explicit `float64` / `int64` standard types.


4. **Outlier Mitigation (Winsorization):** Applied the Interquartile Range (IQR) method ($1.5 \times \text{IQR}$) with `.clip()` boundary capping to preserve sample size while neutralizing severe mathematical distortion.

---

## 📊 Exploratory Visual Analysis 

1. **Univariate Analysis:** Generated histograms with `KDE` curves to evaluate continuous variables, identifying rightward skews in `price` and `horsepower`. Used dynamic count plots to visualize the distributions of separated discrete and categorical columns.

2. **Bivariate Analysis**: Built scatter plots featuring linear regression trendlines to track how continuous features impact vehicle `price`. Applied box plots to categorical features to map their direct financial influence.

---
## 🔗 Multivariate Correlation 

1. **Correlation Matrix:** Rendered a Pearson correlation heatmap to mathematically quantify relationships across features.
2. **Price Drivers & Multicollinearity:** The heatmap identified strong positive `price` drivers, such as `engine size`, and negative `price` drivers, like `highway MPG`. It also exposed redundant features that suffer from multicollinearity.

---
## ⚙️ Feature Engineering 

1. **Feature Creation:** Calculated a `power-to-weight ratio` and averaged the `city` and `highway MPG` to resolve multicollinearity and strengthen predictive signals.
2. **Categorical Encoding:** Applied targeted strategies to convert text into machine-readable numbers without triggering the curse of dimensionality.

* 🔀 **Binary Encoding:** Converted two-option categories into simple `1` and `0` flags.
* 🎯 **Target Encoding:** Grouped high-cardinality features into numerical tiers based on median `prices`.
* ⚡ **One-Hot Encoding:** Split unranked nominal categories into distinct binary columns.

---

## 🖥️ Streamlit Interactive Dashboard

The interactive dashboard (`src/app.py`) provides real-time filtering and visualization across three distinct analysis tabs:

1. **📊 Distributions & Raw Data:** Searchable preview of the filtered subset alongside dynamic univariate distribution histograms.


2. **📈 Bivariate Analysis:** Interactive scatter plots with fitted regression trendlines and category box plots sorted by median vehicle price.


3. **🔥 Correlation Matrix:** Dynamic Pearson correlation heatmaps updating based on active filter criteria.



---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/fatima-eman/automobile-price-eda-dashboard.git
cd automobile-price-eda-dashboard

```

### 2. Set Up Virtual Environment

**On Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate

```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Streamlit Dashboard

```bash
streamlit run src/app.py

```

The application will launch locally at `http://localhost:8501`.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.