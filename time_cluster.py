#%%
import os
import json
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
from matplotlib.dates import date2num

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
occupancy=emptyslots_df.to_numpy()/(emptyslots_df.to_numpy()+freebikes_df.to_numpy())*100
occupancy_df=pd.DataFrame(data=occupancy, index=emptyslots_df.index, columns=emptyslots_df.columns)
occupancy_df.index = pd.to_datetime(occupancy_df.index)
occupancy_df=occupancy_df.rename_axis('timestamp')
occupancy_df.head()
date_02_26 = '2024-02-26'
date_02_27 = '2024-02-27'
date_02_28 = '2024-02-28'
date_02_29 = '2024-02-29'

# Subset the DataFrame based on the date range
occupancy_df_26 = occupancy_df[(occupancy_df.index >= date_02_26) & (occupancy_df.index <= date_02_27)]
occupancy_df_27 = occupancy_df[(occupancy_df.index >= date_02_27) & (occupancy_df.index <= date_02_28)]
occupancy_df_28 = occupancy_df[(occupancy_df.index >= date_02_28) & (occupancy_df.index <= date_02_29)]


occupancy_df_26interp = occupancy_df_26.resample('15T').interpolate(method='linear')
occupancy_df_27interp = occupancy_df_27.resample('15T').interpolate(method='linear')
occupancy_df_28interp = occupancy_df_28.resample('15T').interpolate(method='linear')
rows_with_nan26 = occupancy_df_26interp[occupancy_df_26interp.isna().any(axis=1)]
rows_with_nan27 = occupancy_df_27interp[occupancy_df_27interp.isna().any(axis=1)]
rows_with_nan28 = occupancy_df_28interp[occupancy_df_28interp.isna().any(axis=1)]
occupancy_df_26interp = occupancy_df_26interp.dropna()
occupancy_df_27interp = occupancy_df_27interp.dropna()
occupancy_df_28interp = occupancy_df_28interp.dropna()

#%%
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Assuming df is your DataFrame with stations along columns and timestamps along rows
# df should have timestamps as index and different columns for each station

# Extract time series values
time_series_values = occupancy_df_26.values.transpose()

# Scale the time series
scaled_values = TimeSeriesScalerMeanVariance().fit_transform(time_series_values)

# Apply time series clustering with KMeans using DTW metric
n_clusters = 3  # Adjust the number of clusters as needed

# Define a custom distance function using fastdtw
def dtw_distance(series1, series2):
    _, distance = fastdtw(series1, series2)
    return distance

# Apply time series clustering with KMeans using the custom DTW metric
kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', verbose=True, random_state=42)
y_pred = kmeans.fit_predict(scaled_values)


# %%
indicies=list(range(10))
color_mapping = {0: 'green', 1: 'white', 2: 'white'}
# Use list comprehension to create a list of colors based on the numeric values
color_list = [color_mapping[val] for val in y_pred]
for index in indicies:
    plt.plot(occupancy_df_26.iloc[:,index], color=color_list[index])
# %%
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with stations along columns and timestamps along rows
# df should have timestamps as index and different columns for each station

# Extract time series values
time_series_values = occupancy_df_26.values.transpose()

# Scale the time series
scaled_values = TimeSeriesScalerMeanVariance().fit_transform(time_series_values)

# Initialize a range of cluster numbers
cluster_range = range(1, 6)  # Adjust the range as needed

# Initialize an empty list to store inertia values
inertia_values = []

# Calculate inertia for each number of clusters
for n_clusters in cluster_range:
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=42)
    kmeans.fit(scaled_values)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow plot
#%%
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Plot for Time Series Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Error)')
plt.show()
# %%
import folium

# Example data (replace with your actual data)

# Create a folium map centered around London
london_map = folium.Map(location=[51.509865, -0.118092], zoom_start=13)

# Add nodes to the map with color coding
for lat, lon, color in zip(latitudes, longitudes, color_list[0:10]):
    folium.CircleMarker(
        location=[lon, lat],
        radius=10,
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
