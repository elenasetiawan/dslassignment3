# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 19:32:19 2025

@author: Asus
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def read_csv (filename, year):
    """
    Reads a csv file, removes rows with NaN values, and returns it as a dataframe with the specified year
    Parameters
    filename : str, name of the csv file
    year : str, the year to filter the data

    Returns
    df_complete : dataframe
    """
    
    df = pd.read_csv(filename, skiprows=4)  
    df_clean = df.drop(columns=["Country Code", "Indicator Code", "Indicator Name"], errors="ignore")
    if year not in df_clean.columns:
        raise ValueError(f"Year {year} not found in {filename}")
    
    df_year = df_clean[['Country Name', year]].copy()
    df_year = df_year.dropna()
    
    return df_year

# emissions excluding LULUCF per capita (tCO2e/capita)
emissions20 = read_csv ("API_EN.GHG.CO2.PC.CE.AR5_DS2_en_csv_v2_21047.csv", "2020")
emissions20.columns = ['Country Name', 'CO2 per Capita']
# GDP per capita (current US$)
gdp20 = read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_19346.csv", "2020")
gdp20.columns = ['Country Name', 'GDP per Capita']


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
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], c='red', marker='x', s=100, label='Cluster Centers')

plt.xlabel('GDP per Capita')
plt.ylabel('CO2 Emissions per Capita')
plt.title('Clusters of Countries (2020)')
plt.xscale('log')
plt.yscale('log')
plt.colorbar(scatter)
plt.legend()
plt.grid(True)
plt.show()      
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       