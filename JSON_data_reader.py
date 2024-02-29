#%%
import os
import json
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re


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

# def process_folder(folder_path, category, holder):
#     file_number=count_json_files(folder_path)
#     size_list=[]
#     station_list=[]
#     # List all files in the folder
#     files = os.listdir(folder_path)
#     file_number=count_json_files(folder_path)
    
#     # Extract datetime information from file names and create a dictionary
    
    
    
#     # Process each JSON file in the folder
#     for index in range(file_number):
#         file_name=files[index]
#         # Check if the file is a JSON file
#         if file_name.endswith(".json"):
#             file_path = os.path.join(folder_path, file_name)
#             print(file_name)
#             # Process the JSON file and save as a DataFrame
#             df = process_json_file(file_path)
#             #lords is sometimes not present in the data
#             condition = df['name']!= "002665 - Lord's, St. John's Wood"
#             #df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%SZ')

#             df = df.loc[condition]
            
#             stations_present=df['name'].tolist()
#             station_list.extend(stations_present)
            
#             holder[:,index]=df[category]
#             del df, stations_present
#     del file_number
#     return holder, station_list


def process_folder(folder_path, category, holder):
    # List all files in the folder
    files = os.listdir(folder_path)
    
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

    size_list = []
    station_list = []

    # Process each JSON file in the sorted order
    for file_name in sorted_files:
        file_path = os.path.join(folder_path, file_name)
        # Process the JSON file and save as a DataFrame
        df = process_json_file(file_path)
        condition = df['name'] != "002665 - Lord's, St. John's Wood"
        df = df.loc[condition]

        stations_present = df['name'].tolist()
        number_stations=len(stations_present)
        station_list.extend(stations_present)

        holder[:, sorted_files.index(file_name)] = df[category]
        del df, stations_present

    return holder, station_list[0:number_stations]
           

def count_json_files(folder_path):
    # Use glob to find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    # Return the count of JSON files
    return len(json_files)


def check_station_count(folder_path):
    files = os.listdir(folder_path)
    file_number=count_json_files(folder_path)
    
    # Process each JSON file in the folder
    for index in range(file_number):
        file_name=files[index]
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            df = process_json_file(file_path)
            stations_present=df['name'].tolist()
            station_list.extend(stations_present)
            
            del df, stations_present
    element_counts = Counter(station_list)
    for element, count in element_counts.items():
        if count != file_number:
            print(f"Element {element} occurs {count} times.")
            
        



#%%
#================================================================================
#paste path to JSON folder here
folder_path = "/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/JSON"  

#for reference    
column_names=['empty_slots','free_bikes','id','latitude','longitude','name','timestamp','extra.installDate','extra.installed','extra.locked',
                'extra.name','extra.removalDate','extra.temporary','extra.terminalName','extra.uid']
row_number=791
file_number=count_json_files(folder_path)
#create holding matrix to hold data, better for memory
holding_matrix=np.zeros(shape=(row_number, file_number))
empty_slots, station_list=process_folder(folder_path, 'empty_slots', holding_matrix)
holding_matrix1=np.zeros(shape=(row_number, file_number))
free_bikes, station_list=process_folder(folder_path, 'free_bikes', holding_matrix1)
holding_matrix2=np.zeros(shape=(row_number, file_number), dtype=object)
timestamp, station_list=process_folder(folder_path, 'timestamp', holding_matrix2)
rounded_timestamps = pd.to_datetime(timestamp.flatten()).round('T')[0:file_number]

freebikes_df = pd.DataFrame(free_bikes.T, index=rounded_timestamps, columns=station_list)
emptyslots_df = pd.DataFrame(empty_slots.T, index=rounded_timestamps, columns=station_list)
#duplicated values taken at very simialr time points present in dataframe for some reason
#this line deletes them
freebikes_df=freebikes_df[~freebikes_df.index.duplicated(keep='first')]
emptyslots_df=emptyslots_df[~emptyslots_df.index.duplicated(keep='first')]

freebikes_df.to_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/freebike.csv')
emptyslots_df.to_csv('/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/Processed_CSV/emptyslots.csv')







# station_number=100
# plt.plot(free_bikes[station_number,0:24]/(empty_slots[station_number,0:24]+free_bikes[station_number,0:24])*100)
# plt.plot(free_bikes[station_number,25:48]/(empty_slots[station_number,25:48]+free_bikes[station_number,25:48])*100)
# plt.title(f'% occupancy for station {station_number}')
# plt.ylabel('station occupancy %')
# plt.ylim(bottom=0)
# plt.show()


# avg_bikes=np.mean(free_bikes[:,25:48]/(empty_slots[:,25:48]+free_bikes[:,25:48])*100, axis=1)
# plt.plot(avg_bikes)
# plt.title(f'% occupancy for station {station_number}')
# plt.ylabel('station occupancy %')
# plt.ylim(bottom=0)
# plt.show()

# check_station_count(folder_path)
# print(count_json_files(folder_path))


# plt.plot(np.mean(empty_slots[:,0:24]/(empty_slots[:,0:24]+free_bikes[:,0:24])*100,axis=1))
# plt.ylabel('global station occupancy %')
# plt.ylim(bottom=0)
# plt.show()


# # Define start and end timestamps
# start_timestamp = '2024-02-27 00:00:00'
# end_timestamp = '2024-02-28 00:00:00'

# # Generate timestamps at 15-minute intervals
# time_stamps = pd.date_range(start=start_timestamp, end=end_timestamp, freq='15T')

# # Convert to a NumPy array if needed
# time_matrix = time_stamps.to_numpy()



#This isn't working right now will fix later
#time_recorded, station_list=process_folder(folder_path, 'timestamp', holding_matrix)

#EXAMPLE USAGE:
#empty_slots[station_number, time]
#empty_slots[x, :] =gives the empty slots for the station number x for all times










#plt.plot(empty_slots[0:20,0:10])

#%%
element_counts = Counter(station_list)
for element, count in element_counts.items():
    if count != file_number:
        print(f"Element {element} occurs {count} times.")
    else:    
        print(element)
    


# %%
