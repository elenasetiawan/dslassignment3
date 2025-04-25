# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 19:32:19 2025

@author: Asus
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import r2_score
from errors import *

def read_csv (filename, year = None):
    """
    Reads a csv file, removes rows with NaN values, and returns it as a 
    dataframe with the specified year
    Parameters
    filename : str, name of the csv file
    year : str or None, the year to filter the data

    Returns
    df_complete : dataframe
    """
    df = pd.read_csv(filename, skiprows=4)
    df_clean = df.drop(columns=["Country Code", "Indicator Name", 
                                "Indicator Code"], errors="ignore")
    df_clean = df_clean[df_clean['Country Name'].notna()]

    if year is not None:
       if year not in df_clean.columns:
           raise ValueError(f"Year {year} not found in {filename}")
       
       df_year = df_clean[['Country Name', year]].copy()
       df_year = df_year.dropna()
       return df_year
   
    year_cols = [col for col in df_clean.columns if col.isnumeric()]
    df_long = df_clean[['Country Name'] + year_cols].copy()
    df_long = df_long.melt(id_vars='Country Name', 
                           var_name='Year', 
                           value_name='Value')
    df_long.dropna(inplace=True)
    df_long['Year'] = df_long['Year'].astype(int)
    return df_long


def log_model(x, a, b):
    """function for a logarithmic model"""
    return a * np.log(x) + b


#%% datas
# emissions excluding LULUCF per capita (tCO2e/capita)
emissions20 = read_csv ("API_EN.GHG.CO2.PC.CE.AR5_DS2_en_csv_v2_21047.csv", 
                        "2020")
emissions20.columns = ['Country Name', 'CO2 per Capita']
# GDP per capita (current US$)
gdp20 = read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_19346.csv", "2020")
gdp20.columns = ['Country Name', 'GDP per Capita']

# data for all years
co2_all_years = read_csv("API_EN.GHG.CO2.PC.CE.AR5_DS2_en_csv_v2_21047.csv")
co2_all_years.rename(columns={"Value": "CO2 per Capita"}, inplace=True)
gdp_all_years = read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_19346.csv")
gdp_all_years.rename(columns={"Value": "GDP per Capita"}, inplace=True)

merged20 = emissions20.merge(gdp20, on='Country Name')
merged20 = merged20.dropna()

# normalize
features = merged20[['CO2 per Capita', 'GDP per Capita']].copy()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

      
#%% checks silhouette scores to see number of clusters

for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f"Silhouette score for k={k}: {score:.3f}")

# statistically, k=2 is the best choice

# clustering with k=2
kmeans2 = KMeans(n_clusters=2, random_state=0, n_init=10)
merged20['Cluster_k2'] = kmeans2.fit_predict(scaled_features)

# clustering with k=3
kmeans3 = KMeans(n_clusters=3, random_state=0)
merged20['Cluster'] = kmeans3.fit_predict(scaled_features)

# plotting comparison of graph with k=2 and k=3
fig, axes = plt.subplots(1, 2, figsize=(12,6))

# k=2
scatter1 = axes[0].scatter(
    merged20['GDP per Capita'], 
    merged20['CO2 per Capita'], 
    c=merged20['Cluster_k2'], 
    cmap='viridis', s=60)
axes[0].set_title('Clustering with k=2')
axes[0].set_xlabel('GDP per Capita')
axes[0].set_ylabel('CO2 per Capita')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].grid()

# k=3
scatter2 = axes[1].scatter(
    merged20['GDP per Capita'], 
    merged20['CO2 per Capita'], 
    c=merged20['Cluster'], 
    cmap='viridis', s=60)
axes[1].set_title('Clustering with k=3')
axes[1].set_xlabel('GDP per Capita')
axes[1].set_ylabel('CO2 per Capita')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].grid()

plt.suptitle('Comparison of KMeans Clustering: k=2 vs k=3', fontsize=16)
plt.show()
    
#%% plots the clusters using k=3

# back-transforms the cluster centers to the original scale
cluster_centers = scaler.inverse_transform(kmeans3.cluster_centers_)

plt.figure()
scatter = plt.scatter(merged20['GDP per Capita'], merged20['CO2 per Capita'],
                      c=merged20['Cluster'], cmap='viridis', s=60)
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], c='red', marker='x', 
            s=100, label='Cluster Centers')

plt.xlabel('GDP per Capita')
plt.ylabel('CO2 Emissions per Capita')
plt.title('Clusters of Countries (2020)')
plt.xscale('log')
plt.yscale('log')
plt.colorbar(scatter)
plt.legend()
plt.grid(True)
plt.show()      

# shows which cluster each country belongs to
for cluster_id in sorted(merged20['Cluster'].unique()):
    countries = merged20[merged20['Cluster'] == 
                         cluster_id]['Country Name'].tolist()
    print(f"\nCluster {cluster_id}:")
    for country in countries:
        print(f" - {country}")

#%% data for each clusters

cluster0_countries = merged20[merged20["Cluster"] == 0]["Country Name"].values
cluster_0 = co2_all_years[co2_all_years["Country Name"].isin
                          (cluster0_countries)].copy()
cluster0_df = cluster_0[cluster_0["CO2 per Capita"] > 0]
cluster0_df = cluster0_df.merge(gdp_all_years[['Country Name', 'Year', 
                                               'GDP per Capita']],
                                on=['Country Name', 'Year'], how='left')
cluster0_df = cluster0_df[cluster0_df['GDP per Capita'] > 0]

cluster1_countries = merged20[merged20["Cluster"] == 1]["Country Name"].values
cluster_1 = co2_all_years[co2_all_years["Country Name"].isin
                          (cluster1_countries)].copy()
cluster1_df = cluster_1[cluster_1["CO2 per Capita"] > 0]
cluster1_df = cluster1_df.merge(gdp_all_years[['Country Name', 'Year', 
                                               'GDP per Capita']],
                                on=['Country Name', 'Year'], how='left')
cluster1_df = cluster1_df[cluster1_df['GDP per Capita'] > 0]


cluster2_countries = merged20[merged20["Cluster"] == 2]["Country Name"].values
cluster_2 = co2_all_years[co2_all_years["Country Name"].isin
                          (cluster2_countries)].copy()
cluster2_df = cluster_2[cluster_2["CO2 per Capita"] > 0]
cluster2_df = cluster2_df.merge(gdp_all_years[['Country Name', 'Year', 
                                               'GDP per Capita']],
                                on=['Country Name', 'Year'], how='left')
cluster2_df = cluster2_df[cluster2_df['GDP per Capita'] > 0]


print (cluster0_df, cluster1_df, cluster2_df)

       
#%% fitting CO2 per capita as a function of GDP per capita

x_data = merged20['GDP per Capita'].values
y_data = merged20['CO2 per Capita'].values

# filters dataa to only positive values
mask = (x_data > 0) & (y_data > 0)
x_data = x_data[mask]
y_data = y_data[mask]

merged = merged20.loc[mask].copy()

# fit model
params, cov = optimize.curve_fit(log_model, x_data, y_data)

x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = log_model(x_fit, *params)

# confidence interval
sigma = error_prop(x_fit, log_model, params, cov)
lower = y_fit - sigma
upper = y_fit + sigma


# compute predicted values for CO2 per capita
predicted_y = log_model(x_data, *params)
residuals = y_data - predicted_y

# Add residuals back into the dataframe
merged['Residual'] = residuals
merged['Abs Residual'] = np.abs(residuals)

# Sort by absolute residual to find biggest outliers
outliers = merged.sort_values(by='Abs Residual', ascending=False).head(10)

print("Top 10 outliers:")
print(outliers[['Country Name', 'GDP per Capita', 'CO2 per Capita', 
                'Residual']])


# plot (with confidence)
plt.figure(figsize = (12,6))
scatter = plt.scatter(merged['GDP per Capita'], merged['CO2 per Capita'],
                      c=merged['Cluster'], cmap='viridis', s=60, label='Data')
plt.plot(x_fit, y_fit, color='red', label='Log Fit')
plt.fill_between(x_fit, lower, upper, color='grey', alpha=0.3, 
                 label='Confidence Range')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('GDP per Capita')
plt.ylabel('CO2 per Capita')
plt.title('Log Fit: CO2 vs GDP (2020)')
plt.grid()
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show ()


# Predicted CO2 values using the fitted model
predicted_y = log_model(x_data, *params)

# Calculate R² score
r2 = r2_score(y_data, predicted_y)

print(f"R² Score: {r2}")


#%% predictions

# specific countries (2 from each cluster)
countries_of_interest = ['China', 'United Kingdom', 'Germany', 'Japan', 
                         'Brazil', 'Indonesia']
predictions = []

for country in countries_of_interest:
    # 2020 GDP per capita
    gdp_current = merged20.loc[merged20['Country Name'] == country, 
                               'GDP per Capita'].values[0]    
    # assume 30% growth rate in GDP
    gdp_future = gdp_current * 1.3
    co2_pred = log_model(gdp_future, *params)

    # confidence interval
    sigma = error_prop(np.array([gdp_future]), log_model, params, cov)[0]
    lower = co2_pred - sigma
    upper = co2_pred + sigma

    predictions.append((country, gdp_future, co2_pred, lower, upper))

# Print a clean table
print(f"{'Country':<15} {'Future GDP':>12} {'Pred CO2':>12}" 
      f"{'Lower bound':>12}{'Upper bound':>12}")
for country, gdp, co2, lo, hi in predictions:
    print(f"{country:<15} {gdp:12.2f} {co2:12.2f} {lo:12.2f} {hi:12.2f}")


# plot
colors = plt.cm.tab10.colors

plt.figure(figsize=(10,6))

# plotting historical CO2 per capita
for i, country in enumerate(countries_of_interest):
    country_data = co2_all_years[(co2_all_years['Country Name'] == country) & 
                                 (co2_all_years['CO2 per Capita'] > 0)]
    
    plt.plot(country_data['Year'], country_data['CO2 per Capita'],
             label=f"{country} (historical)", color=colors[i % len(colors)])
    
    _, gdp_future, co2_pred, lower, upper = [pred for pred in predictions if pred[0] == country][0]
    
    plt.errorbar(2030, co2_pred, yerr=[[co2_pred - lower], [upper - co2_pred]], 
                 fmt='o', color=colors[i % len(colors)], capsize=5, label=f"{country} (predicted)")

plt.title("CO₂ per Capita Over Time + 2025 Prediction by Country")
plt.xlabel("Year")
plt.ylabel("CO₂ per Capita")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.yscale("log")
plt.show()













     
       
       
       
       
       
       
       
       
       
       