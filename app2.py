import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Fetching COVID-19 data
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])

# Display COVID-19 data
st.title("COVID-19 Data for USA")
st.write("Current COVID-19 data for the USA:")
st.write(df)

# Visualize COVID-19 data using Matplotlib
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

fig, ax = plt.subplots()
ax.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
ax.set_xlabel("Category")
ax.set_ylabel("Count")
ax.set_title("COVID-19 Data for USA")
st.pyplot(fig)

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Display historical data
st.write("Historical COVID-19 data (last 30 days):")
st.write(df_historical)

# Train a linear regression model
X = df_historical[["day"]]
y = df_historical["cases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict next day's cases
next_day = np.array([[31]])
predicted_cases = model.predict(next_day)

# User input for day number prediction
st.write("Predict COVID-19 cases for a specific day:")
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
