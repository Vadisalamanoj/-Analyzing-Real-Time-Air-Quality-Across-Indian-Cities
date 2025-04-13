#  AQI Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Set dark background theme
plt.style.use('dark_background')
sns.set(style="darkgrid")

# --------------------------------------------
#  Objective 1: Load and Clean the Dataset
# --------------------------------------------
data = pd.read_csv(r"C:\Users\manoj\Desktop\INT375 project\Real time Air Quality Index from various locations.csv")

# Clean column names and handle datetime
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
data['last_update'] = pd.to_datetime(data['last_update'], errors='coerce')
data.dropna(subset=['city', 'pollutant_avg'], inplace=True)

# --------------------------------------------
#  Objective 2: Analyze AQI Levels Across Cities
# --------------------------------------------
city_avg = data.groupby('city')['pollutant_avg'].mean()
top10 = city_avg.sort_values(ascending=False).head(10)

# Display top cities
print("️ Top 10 Most Polluted Cities by Avg AQI:\n", top10)

# Bar Chart for Top 10 Cities
plt.figure(figsize=(10, 6))
sns.barplot(x=top10.values, y=top10.index, hue=top10.index, palette="Reds", dodge=False, legend=False)
plt.title("Top 10 Cities with Highest Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("City")
plt.tight_layout()
plt.show()

# --------------------------------------------
#  Objective 3: Most Frequent and Severe Pollutants
# --------------------------------------------
pollutants = data['pollutant_id'].values
unique, counts = np.unique(pollutants, return_counts=True)
freq_dict = dict(zip(unique, counts))
freq_series = pd.Series(freq_dict).sort_values(ascending=False).head(5)

# Display most frequent pollutants
print("\n Most Frequent Pollutants:\n", freq_series)

# Pie Chart – Top 5 Most Frequent Pollutants
plt.figure(figsize=(6, 6))
freq_series.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
plt.title("Top 5 Most Frequent Pollutants")
plt.ylabel('')
plt.tight_layout()
plt.show()

# Most Severe Pollutants by Average AQI
severe_pollutants = data.groupby('pollutant_id')['pollutant_avg'].mean().sort_values(ascending=False).head(5)
print("\n🔥 Most Severe Pollutants:\n", severe_pollutants)

# Bar Chart – Most Severe Pollutants
plt.figure(figsize=(8, 5))
sns.barplot(x=severe_pollutants.values, y=severe_pollutants.index, hue=severe_pollutants.index, palette="Blues", dodge=False, legend=False)
plt.title("Top 5 Most Severe Pollutants by Avg AQI")
plt.xlabel("Average AQI")
plt.ylabel("Pollutant")
plt.tight_layout()
plt.show()
# --------------------------------------------
#  Objective 4: Pie Chart of AQI Trends
# --------------------------------------------
# Choose sample city (most polluted)
sample_city = data[data['city'] == top10.index[0]].copy()

# AQI classification function
def classify_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

sample_city['aqi_category'] = sample_city['pollutant_avg'].apply(classify_aqi)
category_pie = sample_city['aqi_category'].value_counts()

# Pie chart – AQI category share for sample city
plt.figure(figsize=(6, 6))
category_pie.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette("cool"))
plt.title(f"AQI Category Share in {top10.index[0]}")
plt.ylabel('')
plt.tight_layout()
plt.show()

# --------------------------------------------
#  Objective 5: Classify Cities by AQI Categories
# --------------------------------------------
city_cat = data.groupby('city')['pollutant_avg'].mean().reset_index()
city_cat['category'] = city_cat['pollutant_avg'].apply(classify_aqi)

# Count of cities in each AQI category
category_counts = city_cat['category'].value_counts()

# Bar Chart – AQI Category Distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=category_counts.index, y=category_counts.values, hue=category_counts.index, palette="pastel", dodge=False, legend=False)
plt.title("Number of Cities in Each AQI Category")
plt.xlabel("AQI Category")
plt.ylabel("Number of Cities")
plt.tight_layout()
plt.show()

# --------------------------------------------
#  Objective 6: Smaller Scatter Plot for Air Quality by State
# --------------------------------------------
# Group by state and calculate average AQI
state_avg = data.groupby('state')['pollutant_avg'].mean().reset_index()

# Create a scatter plot with a smaller size
plt.figure(figsize=(6, 4))  # Smaller figure size

# Scatter plot: state average AQI vs. state
sns.scatterplot(data=state_avg, x='state', y='pollutant_avg', color='green', s=100)

# Title and labels
plt.title("Air Quality by State (Average AQI)", fontsize=10)
plt.xlabel("State", fontsize=8)
plt.ylabel("Average AQI", fontsize=8)
plt.xticks(rotation=90)  # Rotate state names for better visibility
plt.tight_layout()  # Ensure the plot fits well within the figure size
plt.show()
