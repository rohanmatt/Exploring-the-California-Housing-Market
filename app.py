import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the dataset
california_housing = fetch_california_housing()
housing_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
housing_data['target'] = california_housing.target  # Add the target variable (median house value)

# Set the page configuration
st.set_page_config(page_title="California Housing Market Analysis", page_icon=":house:", layout="wide")

# Load CSS styles
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.title("Exploring the California Housing Market")

# Introduction
st.markdown("<h1>The Story Begins</h1>", unsafe_allow_html=True)
st.write("Welcome to our interactive exploration of the California housing market. Join us on this journey as we uncover the secrets behind median house values and the factors that shape this dynamic landscape.")

# EDA and Visualizations
st.markdown("<h1>Unveiling the Insights</h1>", unsafe_allow_html=True)
st.write("Through Exploratory Data Analysis (EDA) and insightful visualizations, we've uncovered compelling patterns and relationships that shed light on the housing market's complexities.")

# Scatter Plot
st.markdown("<h2>Median Income: The Economic Driving Force</h2>", unsafe_allow_html=True)
st.write("Our analysis reveals a strong positive correlation between median income and median house values. As median incomes rise, so do housing prices, reflecting the significant influence of economic factors on the housing market.")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='MedInc', y='target', data=housing_data, ax=ax)
ax.set_title('Median House Value vs. Median Income')
ax.set_xlabel('Median Income')
ax.set_ylabel('Median House Value')
st.markdown('<div class="plot-container"></div>', unsafe_allow_html=True)
st.pyplot(fig)

# Feature Relationships
st.markdown("<h2>Understanding the Driving Factors</h2>", unsafe_allow_html=True)
st.write("To gain insights into the relationships between various features and the median house value, we've created a heatmap showcasing the correlations. Explore the strengths and directions of these connections, guiding your understanding of the housing market dynamics.")
numeric_features = housing_data.select_dtypes(include=[float, int]).columns
corr_matrix = housing_data[numeric_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
st.markdown('<div class="plot-container"></div>', unsafe_allow_html=True)
st.pyplot(fig)

# Box Plot
st.markdown("<h2>House Age: A Tale of Time and Value</h2>", unsafe_allow_html=True)
st.write("Interestingly, the box plot reveals that older houses tend to have lower median values, potentially due to factors like deterioration or outdated features. However, there are also outliers with high median values for older houses, suggesting that other factors, such as location or renovations, may play a role.")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='HouseAge', y='target', data=housing_data, ax=ax)
ax.set_title('Median House Value by House Age', fontsize=16)
ax.set_xlabel('House Age', fontsize=14)
ax.set_ylabel('Median House Value', fontsize=14)
plt.xticks(rotation=45)
st.markdown('<div class="plot-container"></div>', unsafe_allow_html=True)
st.pyplot(fig)

# Call to Action
st.markdown("<h1>Join the Housing Market Adventure</h1>", unsafe_allow_html=True)
st.write("Explore our interactive visualizations, uncover hidden patterns, and gain a deeper understanding of the California housing market. Empower your decision-making with data-driven insights.")

# User Interaction
if st.button("View Summary Statistics"):
    st.write("Summary Statistics:")
    st.write(housing_data.describe())


