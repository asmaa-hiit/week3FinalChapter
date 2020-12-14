#Part 1

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

#Part 2

rabat_postal_code_list = [('Agdal-Ryad','Sector 10','33.9621246','-6.8733537'),
('Agdal-Ryad','Sector 11','33.9502218','-6.8699741'),
('Agdal-Ryad','Sector 12','33.9525536','-6.8699527'),
('Agdal-Ryad','Sector 13','33.9501151','-6.8752742'),
('Agdal-Ryad','Sector 14','33.9497769','-6.8809926'),
('Agdal-Ryad','Sector 15','33.9529985','-6.8835569'),
('Agdal-Ryad','Sector 16','33.9539731','-6.8725222'),
('Agdal-Ryad','Sector 17','33.9577864','-6.8778598'),
('Agdal-Ryad','Sector 18','33.9564383','-6.8832779'),
('Agdal-Ryad','Sector 19','33.9599932','-6.8876875'),
('Agdal-Ryad','Sector 20','33.9602202','-6.8801451'),
('Agdal-Ryad','Sector 21','33.9630633','-6.8779671'),
('Agdal-Ryad','Sector 22','33.9665337','-6.8723613'),
('Agdal-Ryad','Sector 23','33.945972','-6.86903'),
('Agdal-Ryad','Sector 24','33.9447483','-6.8738311'),
('Agdal-Ryad','Sector 25','33.9537817','-6.8602591'),
('Agdal-Ryad','Sector 5','33.9541409','-6.8762598'),
('Agdal-Ryad','Sector 6','33.9546583','-6.8625444'),
('Agdal-Ryad','Sector 7','33.9661689','-6.8657791'),
('Agdal-Ryad','Sector 8','33.9567274','-6.8676782'),
('Yacoub El Manssour','El Fath','33.9677281','-6.8991504'),
('Yacoub El Manssour','El Manzah','33.9733277','-6.8969166'),
('Yacoub El Manssour','El Massira','33.9699542','-6.8989031'),
('Yacoub El Manssour','El Amal','33.9842425','-6.8815221'),
('Hassan','Oudayas','33.9830532','-6.9422365'),
('Hassan','Diour Jamaa','34.0164256','-6.8510221'),
('Hassan','Mellah','34.0262614','-6.8317017'),
('Hassan','Hassan','34.0202577','-6.8373125'),
('Hassan','L’ocean','34.0237006','-6.8516452'),
('Hassan','Administratif','34.0118645','-6.8312216'),
('Touarga','Touarga','34.0032231','-6.8471289'),
('El Youssoufia','Mabella','33.9923799','-6.8743145'),
('El Youssoufia','Industriel','33.9878554','-6.8033158'),
('El Youssoufia','Linbiaat','33.9933484','-6.8161208'),
('El Youssoufia','Takaddoum','33.9836411','-6.9446876'),
('Souissi','Chellah','33.9834403','-6.9429407'),
('Souissi','Sector 1','33.9616841','-6.8534919'),
('Souissi','Sector 2','33.962654','-6.8585051'),
('Souissi','Sector 3','33.9634103','-6.8655324'),
('Souissi','Sector 4','33.9602603','-6.861965'),
('Souissi','Sector 9','33.9589077','-6.8671523')
]


#Part 3

# Part 3

#Create a DataFrame object
neighborhoodsRabat = pd.DataFrame(rabat_postal_code_list, columns = ['Borough', 'Neighborhood', 'Latitude', 'Longitude']) 
# Get names of indexes for which column Borough has value Not assigned
indexNames = neighborhoodsRabat[ neighborhoodsRabat['Neighborhood'] == 'Not assigned' ].index
# Delete these row indexes from dataFrame
neighborhoodsRabat.drop(indexNames , inplace=True)
neighborhoodsRabat.tail()



#Part 4

neighborhoodsRabat['Longitude'] = neighborhoodsRabat['Longitude'].replace(r'\s+', np.nan, regex=True)
neighborhoodsRabat['Longitude'] = neighborhoodsRabat['Longitude'].replace(r'^$', np.nan, regex=True)
neighborhoodsRabat['Longitude'] = neighborhoodsRabat['Longitude'].fillna(-0.99999)
neighborhoodsRabat['Longitude'] = pd.to_numeric(neighborhoodsRabat['Longitude'])
neighborhoodsRabat['Latitude'] = neighborhoodsRabat['Latitude'].replace(r'\s+', np.nan, regex=True)
neighborhoodsRabat['Latitude'] = neighborhoodsRabat['Latitude'].replace(r'^$', np.nan, regex=True)
neighborhoodsRabat['Latitude'] = neighborhoodsRabat['Latitude'].fillna(-0.99999)
neighborhoodsRabat['Latitude'] = pd.to_numeric(neighborhoodsRabat['Latitude'])



#Part 5

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
print('The geograpical coordinate of {} are {}, {}.'.format(address, latitude, longitude))



#Part 7

# create map of Rabat en utilisant les valeurs : latitude and longitude values
map_rabat = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhoodRabat in zip(neighborhoodsRabat['Latitude'], neighborhoodsRabat['Longitude'], neighborhoodsRabat['Borough'], neighborhoodsRabat['Neighborhood']):
    label = '{}, {} Latitude {} Longitude {}'.format(neighborhoodRabat, borough, lat, lng)
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


#Part 8


agdal_riad_data = neighborhoodsRabat[neighborhoodsRabat['Borough'] == 'Agdal-Ryad'].reset_index(drop=True)
agdal_riad_data



#Part 9

address = 'Hay Riad, Rabat'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geographical coordinate of {} are {}, {}.'.format(address, latitude, longitude))



#Part 10



# create map of Agdal Riad using latitude and longitude values
map_agdal_riad = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(agdal_riad_data['Latitude'], agdal_riad_data['Longitude'], agdal_riad_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_agdal_riad)  
    
map_agdal_riad


#Part 11



CLIENT_ID = 'PVBTYHASBMH1NGZUITEIGVZO3FUTBPBL1WSYMA010JUN0QXV' # your Foursquare ID
CLIENT_SECRET = '3ZNVZWN4Y4C31WHVD2KOUNVQWJPGSNSR2QJU1XY2KY51W4AQ' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value
radius = 500
LIMIT = 100
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


#Part 12


agdal_riad_data


#Part 13


agdal_riad_data.loc[0, 'Neighborhood']


#Part 14



neighborhood_latitude = agdal_riad_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = agdal_riad_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = agdal_riad_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


#Part 15


url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            neighborhood_latitude, 
            neighborhood_longitude, 
            radius, 
            LIMIT)


#Part 16


results = requests.get(url).json()
results


#Part 17



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

#Part 18

venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


#Part 19


print('{} venues were returned.'.format(nearby_venues.shape[0]))


#Part 20


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




#Part 21

agdal_riad_venues = getNearbyVenues(names=agdal_riad_data['Neighborhood'],
                                   latitudes=agdal_riad_data['Latitude'],
                                   longitudes=agdal_riad_data['Longitude']
                                  )

#Part 22

print(agdal_riad_venues.shape)
agdal_riad_venues.head()


#Part 23


agdal_riad_venues.groupby('Neighborhood').count()


#Part 24

print('There are {} uniques categories.'.format(len(manhattan_venues['Venue Category'].unique())))


#Part 25




# one hot encoding
agdal_riad_onehot = pd.get_dummies(agdal_riad_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
agdal_riad_onehot['Neighborhood'] = agdal_riad_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [agdal_riad_onehot.columns[-1]] + list(agdal_riad_onehot.columns[:-1])
agdal_riad_onehot = agdal_riad_onehot[fixed_columns]

agdal_riad_onehot.head()






#Part 26


agdal_riad_onehot.shape

#Part 27



agdal_riad_grouped = agdal_riad_onehot.groupby('Neighborhood').mean().reset_index()
agdal_riad_grouped

#Part 28




agdal_riad_grouped.shape


#Part 29



num_top_venues = 32

for hood in agdal_riad_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = agdal_riad_grouped[agdal_riad_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')

#Part 30

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


#Part 31


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
neighborhoods_venues_sorted['Neighborhood'] = agdal_riad_grouped['Neighborhood']

for ind in np.arange(agdal_riad_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(agdal_riad_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
s


#Part 32



# set number of clusters
kclusters = 5

agdal_riad_grouped_clustering = agdal_riad_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(agdal_riad_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 

#Part 33


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

agdal_riad_merged = agdal_riad_data

# merge agdal_riad_grouped with agdal_riad_data to add latitude/longitude for each neighborhood
agdal_riad_merged = agdal_riad_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

agdal_riad_merged.head() # check the last columns!


#Part 34

agdal_riad_merged['Cluster Labels'] = agdal_riad_merged['Cluster Labels'].astype(int)
agdal_riad_merged


#Part 35

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(agdal_riad_merged['Latitude'], agdal_riad_merged['Longitude'], agdal_riad_merged['Neighborhood'], agdal_riad_merged['Cluster Labels']):
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


#Part 36



agdal_riad_merged.loc[agdal_riad_merged['Cluster Labels'] == 0, agdal_riad_merged.columns[[1] + list(range(5, agdal_riad_merged.shape[1]))]]

#Part 37

agdal_riad_merged.loc[agdal_riad_merged['Cluster Labels'] == 1, agdal_riad_merged.columns[[1] + list(range(5, agdal_riad_merged.shape[1]))]]

#Part 38

agdal_riad_merged.loc[agdal_riad_merged['Cluster Labels'] == 2, agdal_riad_merged.columns[[1] + list(range(5, agdal_riad_merged.shape[1]))]]

#Part 39

agdal_riad_merged.loc[agdal_riad_merged['Cluster Labels'] == 3, agdal_riad_merged.columns[[1] + list(range(5, agdal_riad_merged.shape[1]))]]


#Part 40

agdal_riad_merged.loc[agdal_riad_merged['Cluster Labels'] == 4, agdal_riad_merged.columns[[1] + list(range(5, agdal_riad_merged.shape[1]))]]

