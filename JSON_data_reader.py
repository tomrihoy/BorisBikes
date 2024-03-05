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
def extract_station_info(json_data):
    # Extract station information from the JSON data
    stations = json_data.get("network", {}).get("stations", [])
    return stations

def process_json_file(file_path):
    # Read JSON file
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # Extract station information
    stations_info = extract_station_info(json_data)

    # Create a DataFrame from the extracted station information
    df = pd.DataFrame(stations_info)

    return df

def process_folder(folder_path, category, stations_to_exclude):
    # List all files in the folder
    files = os.listdir(folder_path)
    file_number=count_json_files(folder_path)
    #791 is the maximum number of stations
    # holder=np.full((792-len(stations_to_exclude),file_number),np.nan)
    # lat_holder=np.full((792-len(stations_to_exclude),file_number),np.nan)
    # long_holder=np.full((792-len(stations_to_exclude),file_number),np.nan)
    # time_holder=np.zeros(shape=(792-len(stations_to_exclude)*2, file_number), dtype=object)
    # Extract datetime information from file names and create a dictionary
    file_dict = {}
    for file_name in files:
        if file_name.endswith(".json"):
            # Extract datetime information from the file name
            match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', file_name)
            if match:
                datetime_str = match.group(1)
                file_dict[file_name] = pd.to_datetime(datetime_str, format='%Y-%m-%d_%H-%M-%S')

    # Sort the file names based on datetime values
    sorted_files = sorted(file_dict, key=file_dict.get)
    #for file_name in sorted_files:
    for file_name in sorted_files:
        file_path = os.path.join(folder_path, file_name)
        # Process the JSON file and save as a DataFrame
        df = process_json_file(file_path)
        condition = ~df['name'].isin(stations_to_exclude)
        reduction=condition.sum()
    holder=np.full((reduction,file_number),np.nan)
    lat_holder=np.full((reduction,file_number),np.nan)
    long_holder=np.full((reduction,file_number),np.nan)
    time_holder=np.zeros(shape=(reduction, file_number), dtype=object)    
    
    # Process each JSON file in the sorted order
    for file_name in sorted_files:
        file_path = os.path.join(folder_path, file_name)
        # Process the JSON file and save as a DataFrame
        df = process_json_file(file_path)
        condition = ~df['name'].isin(stations_to_exclude)
        #condition = df['name'] != "002665 - Lord's, St. John's Wood"
        df = df.loc[condition]
        long_holder[:, sorted_files.index(file_name)] = df['longitude']
        lat_holder[:, sorted_files.index(file_name)] = df['latitude']
        holder[:, sorted_files.index(file_name)] = df[category]
        time_holder[:, sorted_files.index(file_name)] = df['timestamp']
        del df
    rounded_timestamps=pd.to_datetime(time_holder.flatten()).round('T')[0:file_number]
    return rounded_timestamps, holder, lat_holder, long_holder
           

def count_json_files(folder_path):
    # Use glob to find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    # Return the count of JSON files
    return len(json_files)

def check_station_count(folder_path):
    files = os.listdir(folder_path)
    file_dict = {}
    file_number=count_json_files(folder_path)
    for file_name in files:
        if file_name.endswith(".json"):
            match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', file_name)
            if match:
                datetime_str = match.group(1)
                file_dict[file_name] = pd.to_datetime(datetime_str, format='%Y-%m-%d_%H-%M-%S')
    sorted_files = sorted(file_dict, key=file_dict.get)
    station_present = []
    for file_name in sorted_files:
        file_path = os.path.join(folder_path, file_name)
        # Process the JSON file and save as a DataFrame
        df = process_json_file(file_path)
        station_list = df['name'].tolist()
        station_present.extend(station_list)
        del df
    element_counts = Counter(station_present)
    
    missing_stations=[]
    for element, count in element_counts.items():
        if count != file_number:
            print(f"Element {element} occurs {count} times")
            missing_stations.append(element)
    print(f"rest of stations appear {file_number} times ")
    remaining_station_list = [item for item in station_present if item not in missing_stations]
    station_list = []
    seen_elements = set()

    for element in remaining_station_list:
        if element not in seen_elements:
            seen_elements.add(element)
            station_list.append(element)
    return missing_stations, station_list


      
            
#%%
#================================================================================

#Choose day, current days available are 26_Feb-28_Feb
Day="All_data"
#Day="28_Feb"
folder_path = "/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/JSON/"+Day  

#for reference    
column_names=['empty_slots','free_bikes','id','latitude','longitude','name','timestamp','extra.installDate','extra.installed','extra.locked',
                'extra.name','extra.removalDate','extra.temporary','extra.terminalName','extra.uid']

missing_stations, remaining_station=check_station_count(folder_path)

#process_folder(folder_path, 'empty_slots',missing_stations)


times,emptyslots, lat, long=process_folder(folder_path, 'empty_slots',missing_stations)
times,freebikes, _, _=process_folder(folder_path, 'free_bikes',missing_stations)
emptyslots_df=pd.DataFrame(emptyslots.T, index=times, columns=remaining_station)
freebikes_df=pd.DataFrame(freebikes.T, index=times, columns=remaining_station)
lat_df=pd.DataFrame(lat.T, index=times, columns=remaining_station)
long_df=pd.DataFrame(long.T, index=times, columns=remaining_station)
duplicate_indices = emptyslots_df.index.duplicated()
emptyslots_df = emptyslots_df[~duplicate_indices]
freebikes_df = freebikes_df[~duplicate_indices]
lat_df = lat_df[~duplicate_indices].iloc[0,:]
long_df=long_df[~duplicate_indices].iloc[0,:]

emptyslots_df.to_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/emptyslots'+Day+'.csv')
freebikes_df.to_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/freebikes'+Day+'.csv')
lat_df.to_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/latitudes.csv')
long_df.to_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/longitudes.csv')
# %%
new_timestamps = pd.date_range(emptyslots_df.index.min(), emptyslots_df.index.max(), freq='15min')
df_interpolated = pd.DataFrame(index=new_timestamps)
for column in emptyslots_df.columns:
    df_interpolated[column] = emptyslots_df[column].interpolate(method='linear').round().astype(int)
# %%
