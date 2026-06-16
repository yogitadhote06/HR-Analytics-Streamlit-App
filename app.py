import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HR Analytics Dashboard")
st.markdown("Analyze Employee Attrition, Salary Trends, Segmentation and Predictions")

# ---------------------------
# LOAD DATA
# ---------------------------

df = pd.read_csv("HR_Analytics.csv")

# ---------------------------
# TRAIN MODEL
# ---------------------------

data = df.copy()

data['Attrition'] = data['Attrition'].map({
    'Yes': 1,
    'No': 0
})

for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop('Attrition', axis=1)
y = data['Attrition']

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X, y)

# ---------------------------
# SIDEBAR
# ---------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Attrition Analysis",
        "Salary Analysis",
        "Heatmap",
        "Employee Segmentation",
        "Prediction"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("Dashboard Overview")

    total_emp = len(df)

    attrition_rate = (
        df['Attrition']
        .value_counts(normalize=True)['Yes']
        * 100
    )

    avg_salary = df['MonthlyIncome'].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Employees", total_emp)
    c2.metric("Attrition Rate", f"{attrition_rate:.2f}%")
    c3.metric("Average Salary", f"${avg_salary:,.0f}")

    fig = px.pie(
        df,
        names='Attrition',
        title='Attrition Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ATTRITION ANALYSIS
# =====================================================

elif page == "Attrition Analysis":

    st.header("Employee Attrition Analysis")

    fig = px.histogram(
        df,
        x='Department',
        color='Attrition',
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(
        df,
        x='Gender',
        color='Attrition'
    )

    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# SALARY ANALYSIS
# =====================================================

elif page == "Salary Analysis":

    st.header("Salary Distribution")

    fig = px.histogram(
        df,
        x='MonthlyIncome',
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(
        df,
        x='Department',
        y='MonthlyIncome'
    )

    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# HEATMAP
# =====================================================

elif page == "Heatmap":

    st.header("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=['int64', 'float64']
    )

    corr = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    sns.heatmap(
        corr,
        cmap='coolwarm',
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# EMPLOYEE SEGMENTATION
# =====================================================

elif page == "Employee Segmentation":

    st.header("Employee Segmentation")

    features = df[
        [
            'Age',
            'MonthlyIncome',
            'JobSatisfaction',
            'YearsAtCompany'
        ]
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42
    )

    df['Cluster'] = kmeans.fit_predict(scaled)

    fig = px.scatter(
        df,
        x='Age',
        y='MonthlyIncome',
        color=df['Cluster'].astype(str),
        title="Employee Clusters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.header("🤖 Employee Attrition Prediction")

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=10000
    )

    satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        2
    )

    years = st.slider(
        "Years At Company",
        0,
        40,
        5
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

    if st.button("Predict"):

        risk_score = 0

        if satisfaction <= 2:
            risk_score += 1

        if years < 3:
            risk_score += 1

        if overtime == "Yes":
            risk_score += 1

        if income < 5000:
            risk_score += 1

        if risk_score >= 3:
            st.error("⚠️ High Attrition Risk")

        elif risk_score == 2:
            st.warning("⚠️ Moderate Attrition Risk")

        else:
            st.success("✅ Low Attrition Risk")