import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Titanic Data Analysis",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Dataset Analysis Dashboard")

DATA_PATH = Path("data/raw/titanic_train.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()

# Sidebar
st.sidebar.header("Dataset Information")

st.sidebar.write("Rows:", df.shape[0])
st.sidebar.write("Columns:", df.shape[1])

# Dataset preview
st.header("Dataset Preview")

st.dataframe(df.head(20))

# Basic statistics
st.header("Dataset Statistics")

st.dataframe(df.describe())

# Missing values
st.header("Missing Values")

missing = df.isnull().sum()
st.bar_chart(missing)

# Survival distribution
st.header("Survival Distribution")

fig, ax = plt.subplots()
sns.countplot(data=df, x="Survived", ax=ax)
ax.set_title("Survival Count")
st.pyplot(fig)

# Survival by gender
st.header("Survival by Gender")

fig, ax = plt.subplots()
sns.countplot(data=df, x="Sex", hue="Survived", ax=ax)
ax.set_title("Survival by Gender")
st.pyplot(fig)

# Survival by passenger class
st.header("Survival by Passenger Class")

fig, ax = plt.subplots()
sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax)
ax.set_title("Survival by Class")
st.pyplot(fig)

# Age distribution
st.header("Age Distribution")

fig, ax = plt.subplots()
sns.histplot(df["Age"].dropna(), bins=30, kde=True, ax=ax)
ax.set_title("Age Distribution")
st.pyplot(fig)

# Fare distribution
st.header("Fare Distribution")

fig, ax = plt.subplots()
sns.histplot(df["Fare"], bins=30, kde=True, ax=ax)
ax.set_title("Fare Distribution")
st.pyplot(fig)

# Correlation heatmap
st.header("Feature Correlation")

numeric_df = df.select_dtypes(include=["float64", "int64"])

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Interactive filtering
st.header("Interactive Data Exploration")

selected_class = st.selectbox(
    "Select Passenger Class",
    df["Pclass"].unique()
)

filtered_df = df[df["Pclass"] == selected_class]

st.write("Filtered Data", filtered_df)