
import os
import json
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter



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

def process_folder(folder_path, category, holder):
    size_list=[]
    station_list=[]
    # List all files in the folder
    files = os.listdir(folder_path)
    file_number=count_json_files(folder_path)
    # Process each JSON file in the folder
    for index in range(file_number):
        file_name=files[index]
        # Check if the file is a JSON file
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            
            # Process the JSON file and save as a DataFrame
            df = process_json_file(file_path)
            #lords is sometimes not present in the data
            condition = df['name']!= "002665 - Lord's, St. John's Wood"
            #df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%SZ')

            df = df.loc[condition]
            
            stations_present=df['name'].tolist()
            station_list.extend(stations_present)
            holder[:,index]=df[category]
            del df, stations_present
    return holder, station_list
           

def count_json_files(folder_path):
    # Use glob to find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    # Return the count of JSON files
    return len(json_files)





folder_path = "/Users/tomrihoy/Desktop/ICL/Academic_work/Data_science/Santander_bikes_project/JSON"  
    
column_names=['empty_slots','free_bikes','id','latitude','longitude','name','timestamp','extra.installDate','extra.installed','extra.locked',
                'extra.name','extra.removalDate','extra.temporary','extra.terminalName','extra.uid']
row_number=791
file_number=count_json_files(folder_path)
#create holding matrix to hold data, better for memory
holding_matrix=np.zeros(shape=(row_number, file_number))


empty_slots, station_list=process_folder(folder_path, 'empty_slots', holding_matrix)
free_bikes, station_list=process_folder(folder_path, 'free_bikes', holding_matrix)
#This isn't working right now will fix later
#time_recorded, station_list=process_folder(folder_path, 'timestamp', holding_matrix)


#EXAMPLE USAGE:
#empty_slots[station_number, time]
#empty_slots[x, :] =gives the empty slots for the station number x for all times










#plt.plot(empty_slots[0:20,0:10])


# element_counts = Counter(station_list)
# for element, count in element_counts.items():
#     if count != file_number:
#         print(f"Element {element} occurs {count} times.")
#         print(element)
    

