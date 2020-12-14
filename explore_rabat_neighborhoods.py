# Part 1 

import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')

# Part 2

rabat_postal_code_list = [('Riad–Agdal','Agdal','33.958469','-6.869386'),
('Souissi','Souissi','33.9688418','-6.8348164'),
('Hassan','Hassan','34.0227868','-6.8238124'),
('El Youssoufia','El Youssoufia','33.9920961','-6.8103975'),
('Yacoub El Manssour','Yacoub El Manssour','33.976447','-6.888387')]


# Part 3

#Create a DataFrame object
neighborhoodsRabat = pd.DataFrame(rabat_postal_code_list, columns = ['Borough', 'Neighborhood', 'Latitude', 'Longitude']) 
# Get names of indexes for which column Borough has value Not assigned
indexNames = neighborhoodsRabat[ neighborhoodsRabat['Neighborhood'] == 'Not assigned' ].index
# Delete these row indexes from dataFrame
neighborhoodsRabat.drop(indexNames , inplace=True)
neighborhoodsRabat.head()

# Part 4

neighborhoodsRabat['Longitude'] = neighborhoodsRabat['Longitude'].replace(r'\s+', np.nan, regex=True)
neighborhoodsRabat['Longitude'] = neighborhoodsRabat['Longitude'].replace(r'^$', np.nan, regex=True)
neighborhoodsRabat['Longitude'] = neighborhoodsRabat['Longitude'].fillna(-0.99999)
neighborhoodsRabat['Longitude'] = pd.to_numeric(neighborhoodsRabat['Longitude'])
neighborhoodsRabat['Latitude'] = neighborhoodsRabat['Latitude'].replace(r'\s+', np.nan, regex=True)
neighborhoodsRabat['Latitude'] = neighborhoodsRabat['Latitude'].replace(r'^$', np.nan, regex=True)
neighborhoodsRabat['Latitude'] = neighborhoodsRabat['Latitude'].fillna(-0.99999)
neighborhoodsRabat['Latitude'] = pd.to_numeric(neighborhoodsRabat['Latitude'])

# Part 5 

print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(neighborhoodsRabat['Borough'].unique()),
        neighborhoodsRabat.shape[0]
    )
)

# Part 6 

address = 'Rabat, Maroc'

geolocator = Nominatim(user_agent="mo_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Rabat are {}, {}.'.format(latitude, longitude))

# Part 7

neighborhoodsRabat.head()

# Part 8 

# create map of Rabat en utilisant les valeurs : latitude and longitude values
map_rabat = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhoodRabat in zip(neighborhoodsRabat['Latitude'], neighborhoodsRabat['Longitude'], neighborhoodsRabat['Borough'], neighborhoodsRabat['Neighborhood']):
    label = '{}, {} Latitude {} Longitude {}'.format(neighborhoodRabat, borough, lat, lng)
    print(label)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        # [34.0227868, -6.8238124],
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_rabat)
    
map_rabat

# Part 9

CLIENT_ID = 'PVBTYHASBMH1NGZUITEIGVZO3FUTBPBL1WSYMA010JUN0QXV' # your Foursquare ID
CLIENT_SECRET = '3ZNVZWN4Y4C31WHVD2KOUNVQWJPGSNSR2QJU1XY2KY51W4AQ' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value
radius = 500
LIMIT = 100
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

# Part 10 

hassan_data = neighborhoodsRabat[neighborhoodsRabat['Borough'] == 'Hassan'].reset_index(drop=True)
hassan_data

# Part 11 

hassan_data.loc[0, 'Neighborhood']

# Part 12


neighborhood_latitude = hassan_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = hassan_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = hassan_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))

# Part 13 




url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            neighborhood_latitude, 
            neighborhood_longitude, 
            radius, 
            LIMIT)

# Part 14 


results = requests.get(url).json()
results


# Part 15


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# Part 16 


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues


# Part 17 

print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


#Part 18

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

# Part 19

hassan_venues = getNearbyVenues(names=hassan_data['Neighborhood'],
                                   latitudes=hassan_data['Latitude'],
                                   longitudes=hassan_data['Longitude']
                                  )

# Part 20

print(hassan_venues.shape)
hassan_venues


# Part 21

hassan_venues.groupby('Neighborhood').count()


# Part 22

print('There are {} uniques categories.'.format(len(hassan_venues['Venue Category'].unique())))

# Part 23



# one hot encoding
hassan_onehot = pd.get_dummies(hassan_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
hassan_onehot['Neighborhood'] = hassan_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [hassan_onehot.columns[-1]] + list(hassan_onehot.columns[:-1])
hassan_onehot = hassan_onehot[fixed_columns]

hassan_onehot


# Part 24

hassan_onehot.shape


# Part 25


hassan_grouped = hassan_onehot.groupby('Neighborhood').mean().reset_index()
hassan_grouped


# Part 26

hassan_grouped.shape

# Part 27


num_top_venues = 5

for hood in hassan_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = hassan_grouped[hassan_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')

# Part 28


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Part 29


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = hassan_grouped['Neighborhood']

for ind in np.arange(hassan_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(hassan_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()

# Part 30


# set number of clusters
kclusters = 1

hassan_grouped_clustering = hassan_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(hassan_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# Part 31


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

hassan_merged = hassan_data

# merge hassan_grouped with hassan_data to add latitude/longitude for each neighborhood
hassan_merged = hassan_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

hassan_merged.head() # check the last columns!



# Part 32


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(hassan_merged['Latitude'], hassan_merged['Longitude'], hassan_merged['Neighborhood'], hassan_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

# Part 33

hassan_merged.loc[hassan_merged['Cluster Labels'] == 0, hassan_merged.columns[[1] + list(range(5, hassan_merged.shape[1]))]]

