from flask import Flask, render_template
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os

app = Flask(__name__)

def process_transit_data():
    # Load datasets (adjusting path to look inside the data folder)
    base_path = os.path.dirname(os.path.abspath(__file__))
    wards_path = os.path.join(base_path, 'data', 'ward_level_collated.csv')
    transit_path = os.path.join(base_path, 'data', 'unified_mumbai_transport.csv')
    
    df_wards = pd.read_csv(wards_path)
    df_transit = pd.read_csv(transit_path)

    df_transit = df_transit.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    df_transit = df_transit.dropna(subset=['latitude', 'longitude'])
    df_wards = df_wards.dropna(subset=['Ward_Alphabet'])

    mumbai_ward_coords = {
        'A': (18.9220, 72.8347), 'B': (18.9548, 72.8377), 'C': (18.9449, 72.8259),
        'D': (18.9647, 72.8130), 'E': (18.9696, 72.8423), 'F/N': (19.0238, 72.8550),
        'F/S': (19.0014, 72.8452), 'G/N': (19.0330, 72.8475), 'G/S': (19.0103, 72.8262),
        'H/E': (19.0700, 72.8468), 'H/W': (19.0657, 72.8310), 'K/E': (19.1136, 72.8697),
        'K/W': (19.1197, 72.8464), 'L': (19.0759, 72.8877), 'M/E': (19.0473, 72.9158),
        'M/W': (19.0596, 72.8958), 'N': (19.1417, 72.9331), 'P/N': (19.1874, 72.8484),
        'P/S': (19.1551, 72.8464), 'R/C': (19.2215, 72.8556), 'R/N': (19.2804, 72.8597),
        'R/S': (19.2094, 72.8126), 'S': (19.1306, 72.9375), 'T': (19.1735, 72.9495)
    }

    df_wards['latitude'] = df_wards['Ward_Alphabet'].map(lambda x: mumbai_ward_coords.get(x, (19.0760, 72.8777))[0])
    df_wards['longitude'] = df_wards['Ward_Alphabet'].map(lambda x: mumbai_ward_coords.get(x, (19.0760, 72.8777))[1])

    # Spatial calculation using cKDTree
    transit_coords = df_transit[['latitude', 'longitude']].values
    ward_coords = df_wards[['latitude', 'longitude']].values
    tree = cKDTree(transit_coords)
    distances, _ = tree.query(ward_coords)
    df_wards['distance_km'] = distances * 111

    # Calculations for poor access & priority
    df_wards['Accessibility_Gap_Score'] = df_wards['distance_km'] * df_wards['TOT_P_DEN']
    
    # Poor access dataframe
    df_poor = df_wards[df_wards['distance_km'] > 0.30].sort_values(by='distance_km', ascending=False)
    poor_access_list = []
    for _, row in df_poor.iterrows():
        poor_access_list.append({
            'ward': row['Ward_Alphabet'],
            'name': row['Ward_Names'],
            'distance': round(row['distance_km'], 2)
        })

    # Prioritized dataframe
    df_prio = df_wards.sort_values(by='Accessibility_Gap_Score', ascending=False).head(5)
    prioritized_list = []
    for _, row in df_prio.iterrows():
        prioritized_list.append({
            'ward': row['Ward_Alphabet'],
            'name': row['Ward_Names'],
            'distance': round(row['distance_km'], 2),
            'score': round(row['Accessibility_Gap_Score'], 2)
        })

    return poor_access_list, prioritized_list

@app.route('/')
def home():
    poor_access, prioritized = process_transit_data()
    return render_template('index.html', poor_access=poor_access, prioritized=prioritized)

if __name__ == '__main__':
    app.run(debug=True)
