from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np

# saut de ligne

df = pd.read_csv('MoroccoPostalcode (1).csv')
df.head

# saut de ligne

from geopy.geocoders import Nominatim

# saut de ligne

address = 'Morocco'
geolocator = Nominatim(user_agent="morocco_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The coordinates of Morocco are {}, {}.'.format(latitude, longitude))

# saut de ligne

pip install folium

# saut de ligne

import folium

# saut de ligne

map_Morocco = folium.Map(location=[latitude, longitude], zoom_start=11)

# saut de ligne

for latitude, longitude, Pays, Ville in zip(df['Latitude'], df['Longitude'], df['Pays'], df['Ville']):
    label = '{}, {}'.format(Pays, Ville)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [latitude, longitude],
        radius=5,
        popup=label,
        color='black',
        fill=True
        ).add_to(map_Morocco)
map_Morocco
