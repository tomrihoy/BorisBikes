#%%
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
from matplotlib.dates import date2num
import numpy as np
from fastdtw import fastdtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import folium
import seaborn as sns
from datetime import datetime


#%%
emptyslots_df=pd.read_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/emptyslotsAll_data.csv', index_col=None)
freebikes_df=pd.read_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/freebikesAll_data.csv', index_col=None)
emptyslots_df.reset_index()
emptyslots_df=emptyslots_df.set_index('Unnamed: 0')
freebikes_df.reset_index()
freebikes_df=freebikes_df.set_index('Unnamed: 0')
latitudes=pd.read_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/latitudes.csv').iloc[:,-1].to_list()
longitudes=pd.read_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/longitudes.csv').iloc[:,-1].to_list()


#%%
occupancy=freebikes_df.to_numpy()/(emptyslots_df.to_numpy()+freebikes_df.to_numpy())*100
occupancy_df=pd.DataFrame(data=occupancy, index=emptyslots_df.index, columns=emptyslots_df.columns)
occupancy_df.index = pd.to_datetime(occupancy_df.index)
occupancy_df=occupancy_df.rename_axis('timestamp')
freebikes_df.index = pd.to_datetime(freebikes_df.index)
freebikes_df=freebikes_df.rename_axis('timestamp')


date_02_26 = '2024-02-26'
date_02_27 = '2024-02-27'
date_02_28 = '2024-02-28'
date_02_29 = '2024-02-29'
date_03_01 = '2024-03-01'
date_03_02 = '2024-03-02'
date_03_03 = '2024-03-03'
date_03_04 = '2024-03-04'
date_03_05 = '2024-03-05'

# Subset the DataFrame based on the date range
occupancy_df_26 = occupancy_df[(occupancy_df.index >= date_02_26) & (occupancy_df.index <= date_02_27)]
occupancy_df_27 = occupancy_df[(occupancy_df.index >= date_02_27) & (occupancy_df.index <= date_02_28)]
occupancy_df_28 = occupancy_df[(occupancy_df.index >= date_02_28) & (occupancy_df.index <= date_02_29)]
occupancy_df_29 = occupancy_df[(occupancy_df.index >= date_02_29) & (occupancy_df.index <= date_03_01)]
occupancy_df_01 = occupancy_df[(occupancy_df.index >= date_03_01) & (occupancy_df.index <= date_03_02)]
occupancy_df_02 = occupancy_df[(occupancy_df.index >= date_03_02) & (occupancy_df.index <= date_03_03)]
occupancy_df_03 = occupancy_df[(occupancy_df.index >= date_03_03) & (occupancy_df.index <= date_03_04)]
occupancy_df_04 = occupancy_df[(occupancy_df.index >= date_03_04) & (occupancy_df.index <= date_03_05)]

freebikes_df_26 = freebikes_df[(freebikes_df.index >= date_02_26) & (freebikes_df.index <= date_02_27)]
freebikes_df_27 = freebikes_df[(freebikes_df.index >= date_02_27) & (freebikes_df.index <= date_02_28)]
freebikes_df_28 = freebikes_df[(freebikes_df.index >= date_02_28) & (freebikes_df.index <= date_02_29)]
freebikes_df_29 = freebikes_df[(freebikes_df.index >= date_02_29) & (freebikes_df.index <= date_03_01)]
freebikes_df_01 = freebikes_df[(freebikes_df.index >= date_03_01) & (freebikes_df.index <= date_03_02)]
freebikes_df_02 = freebikes_df[(freebikes_df.index >= date_03_02) & (freebikes_df.index <= date_03_03)]
freebikes_df_03 = freebikes_df[(freebikes_df.index >= date_03_03) & (freebikes_df.index <= date_03_04)]
freebikes_df_04 = freebikes_df[(freebikes_df.index >= date_03_04) & (freebikes_df.index <= date_03_05)]



#%%
#PERFORMS TIME SERIES CLUSTERING FOR N CLUSTERS
#>>>>>>>This line specifies the dataset to use for the clustering and plotting<<<<<<<<<
data=occupancy_df_01

#data=occupancy_df_29
# Extract time series values
time_series_values = data.values.transpose()
# Scale the time series
scaled_values = TimeSeriesScalerMeanVariance().fit_transform(time_series_values)
# Apply time series clustering with KMeans using DTW metric
n_clusters = 3  # Adjust the number of clusters as needed
# Apply time series clustering with KMeans using the custom DTW metric
kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', verbose=False, random_state=42)
y_pred = kmeans.fit_predict(scaled_values)


color_mapping = {0:'green', 1:'blue', 2:'yellow', 3: 'red'}
color_list=[color_mapping[val] for val in y_pred]

# %%
# PLOTS STATION LOCATIONS ON MAP AND COLOUR CODED FOR CLUSTER TYPE
london_map = folium.Map(location=[51.5098, -0.118092], zoom_start=11)
# Add nodes to the map with color coding
for lat, lon, color in zip(latitudes, longitudes, color_list):
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup='Node Information',  # Add your node-specific information here
    ).add_to(london_map)

# Save the map as an HTML file or display it in a Jupyter Notebook
london_map.save('london_map.html')
london_map

# %%
#GENERATES ELBOW PLOT FOR DIFFERENT TIME SERIES CLUSTERS
# Extract time series values

time_series_values = data.values.transpose()
# Scale the time series
scaled_values = TimeSeriesScalerMeanVariance().fit_transform(time_series_values)
# Initialize a range of cluster numbers
cluster_range = range(1, 10)  # Adjust the range as needed
# Initialize an empty list to store inertia values
inertia_values = []
# Calculate inertia for each number of clusters
for n_clusters in cluster_range:
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=42)
    kmeans.fit(scaled_values)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow plot
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Plot for Time Series Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Error)')
plt.show()

# %%
#EXTRACT THE INDEX VALUES OF THE CLUSTER ELEMNTS AND FIND AN 
# AVERAGE PROFILE OF THE CLUSTER BASED OFF THESE

def extract_all_indexes(cluster_labels):
    result = {}
    for label in set(cluster_labels):
        indices = [i for i, x in enumerate(cluster_labels) if x == label]
        result[label] = indices
    return result

# Example usage
cluster_indexes02 = extract_all_indexes(y_pred02)
#%%
cluster1avg=np.mean(data.iloc[:,cluster_indexes[0]].to_numpy(), axis=1)
cluster2avg=np.mean(data.iloc[:,cluster_indexes[1]].to_numpy(), axis=1)
cluster3avg=np.mean(data.iloc[:,cluster_indexes[2]].to_numpy(), axis=1)
#cluster4avg=np.mean(occupancy_df_27.iloc[:,cluster_indexes[3]].to_numpy(), axis=1)

plt.plot(cluster1avg, label='cluster 1', color='green')
plt.plot(cluster2avg, label='cluster 2', color='blue')
plt.plot(cluster3avg, label='cluster 3', color='yellow')
#plt.plot(cluster4avg, label='cluster 4', color='red')
plt.ylabel('Average station occupancy %')
#plt.ylabel('Average Free bikes available')
plt.xlabel('Data points')
plt.title(f'Clusters of the data based off occupancy')
plt.legend()
# %%

# Convert timestamp to numeric format using date2num
data['timestamp_num'] = date2num(data.index)

# Plotting using scatter
plt.scatter(data['timestamp_num'], cluster1avg)
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Irregular Time Series Plot')
plt.xticks(rotation=45)

# Convert the numeric timestamp back to date for better x-axis labeling
plt.gca().xaxis_date()

plt.show()

# %%
# %%
#EXTRACT THE INDEX VALUES OF THE CLUSTER ELEMNTS AND FIND AN 
# AVERAGE PROFILE OF THE CLUSTER BASED OFF THESE

def extract_all_indexes(cluster_labels):
    result = {}
    for label in set(cluster_labels):
        indices = [i for i, x in enumerate(cluster_labels) if x == label]
        result[label] = indices
    return result

# Example usage
cluster_indexes = extract_all_indexes(y_pred01)
cluster1avg=np.mean(data.iloc[:,cluster_indexes[0]].to_numpy(), axis=1)
cluster2avg=np.mean(data.iloc[:,cluster_indexes[1]].to_numpy(), axis=1)
cluster3avg=np.mean(data.iloc[:,cluster_indexes[2]].to_numpy(), axis=1)
#cluster4avg=np.mean(occupancy_df_27.iloc[:,cluster_indexes[3]].to_numpy(), axis=1)

plt.plot(cluster1avg, label='cluster 1', color='green')
plt.plot(cluster2avg, label='cluster 2', color='blue')
plt.plot(cluster3avg, label='cluster 3', color='yellow')
#plt.plot(cluster4avg, label='cluster 4', color='red')
plt.ylabel('Average station occupancy %')
#plt.ylabel('Average Free bikes available')
plt.xlabel('Data points')
plt.title(f'Clusters of the data based off occupancy')
plt.legend()
# %%
