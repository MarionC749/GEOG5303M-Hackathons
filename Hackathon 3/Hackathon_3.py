#Submitted as part of coursework for GEOG5303M: Creative Coding for Real World Problems
# %%
pip install folium matplotlib pysal geopandas

# %%
#Import packages
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats


# %% [markdown]
# # Defining High Streets in Liverpool

# %% [markdown]
# ## Data exploration

# %%
#Load POI Liverpool file
POILiverpool = gpd.read_file('/Users/marion/Desktop/Etudes - Leeds/Semester 2/Creative Coding for Real World Problems/Topic 3/Hackathon 3 Data/poi_Liverpool.gpkg')
POILiverpool.info()

# %%
POILiverpool.head()

# %%
#Upload Liverpool Boundary
LiverpoolPol = gpd.read_file('/Users/marion/Desktop/Etudes - Leeds/Semester 2/Creative Coding for Real World Problems/Topic 3/Hackathon 3 Data/Liverpool Boundary/LiverpoolPolygon.shp')
LiverpoolPol.plot()

# %%
LiverpoolPol.info()

# %%
#Check CRS of Liverpool files
print(LiverpoolPol.crs)
print(POILiverpool.crs)

# %%
#Clip the POI to Liverpool boundary
LiverpoolPol['outline'] = 1
LiverpoolPol_outline = LiverpoolPol.dissolve(by = 'outline')
Liverpool_POI = gpd.clip(POILiverpool, LiverpoolPol_outline)
Liverpool_POI.plot()

# %%
Liverpool_POI['groupname'].unique()

# %%
Liverpool_POI['categoryname'].unique()

# %%
Liverpool_POI['classname'].unique()

# %%
#Upload Liverpool Roads
Liverpool_Mer = gpd.read_file('/Users/marion/Desktop/Etudes - Leeds/Semester 2/Creative Coding for Real World Problems/Topic 3/Hackathon 3 Data/Roads Liverpool/Meridian2_250305_124659.shp')
Liverpool_Mer.plot()

# %%
print(Liverpool_Mer.crs)

# %%
#Reproject data to EPSG:27700
Liverpool_Mer = Liverpool_Mer.to_crs(epsg=27700)
print(LiverpoolPol.crs)

# %%
#Clip the Liverpool Roads to Liverpool boundary
LiverpoolPol['outline2'] = 1
LiverpoolPol_outline = LiverpoolPol.dissolve(by = 'outline2')
Liverpool_Roads = gpd.clip(Liverpool_Mer, LiverpoolPol_outline)
Liverpool_Roads.plot()

# %%
Liverpool_Roads.head()

# %%
#Create unique identifier for the roads
Liverpool_Roads['ID'] = range(1, len(Liverpool_Roads) + 1)
Liverpool_Roads.head()

# %%
#Extract retail POIs
Liverpool_Retail = Liverpool_POI[Liverpool_POI['groupname'] == 'Retail']
#Get rid of columns with NAs which do not matter here
Liverpool_Retail = Liverpool_Retail.drop(columns=['uprn',
                                         'address_detail',
                                         'street_name',
                                         'locality',
                                         'telephone_number',
                                         'url',
                                         'brand',
                                         'qualifier_type',
                                         'qualifier_data'])
Liverpool_Retail.info()

# %% [markdown]
# ## Use DBSCAN to create clusters of at least 10 POIs that are maximum 150m away from each other

# %%
#Extract the coordinates from geometries
coords = Liverpool_Retail.geometry.apply(lambda geom: [geom.x, geom.y]).tolist()

# %%
#DBSCAN
from sklearn.cluster import DBSCAN

#eps = how near points need to be to be included in the same cluster = 150m
#min_samples = smallest number of points we need to be able to call the collection of points a 'cluster' = 10
#metric='euclidean' = for the distance to be in meters
dbscan = DBSCAN(eps=150, min_samples=10, metric='euclidean')

# %%
#Fit the model to the coordinates of POIs
dbscan.fit(coords)

#assign the labels to a new variable
dbscan_labels = dbscan.labels_

#Add the clusters IDs as a new column and view the clusters
Liverpool_Retail['dClusters'] = dbscan_labels
Liverpool_Retail['dClusters'].value_counts()
#Result: 961 POIs have not been clustered

# %%
#Drop all the POIs which are not in clusters (are in the -1 cluster)
Liverpool_Retail_Clusters = Liverpool_Retail[Liverpool_Retail['dClusters'] != -1]

# %%
#Plot clusters
plt.scatter(Liverpool_Retail_Clusters.geometry.x, Liverpool_Retail_Clusters.geometry.y, c=Liverpool_Retail_Clusters['dClusters'], cmap='viridis', marker='o')
plt.title("DBSCAN Clusters")
plt.show()

# %%
Liverpool_Retail_Clusters.info()

# %%
Liverpool_Retail_Clusters['dClusters'].unique()
#Result: Have 33 clusters

# %%
#Visualize where the clusters are on a map
import folium

Liverpool_Retail_Clusters.explore(
    column="dClusters",
    cmap="hsv",
)

# %% [markdown]
# ## Identify streets with clusters and having at least 10 retail shops

# %%
#Create buffer column of roads (30m)
Liverpool_Roads['buffered_geometry'] = Liverpool_Roads.geometry.buffer(30)
#Assign the column buffered_geometry as the geometry to use
Liverpool_Roads2 = Liverpool_Roads.set_geometry('buffered_geometry')
Liverpool_Roads2 = Liverpool_Roads2.drop(columns= 'geometry')
Liverpool_Roads2.head()

# %%
#Visualize where the buffered roads are on a map
import folium

Liverpool_Roads2.explore(
    column="ID",
    cmap="hsv",
)

# %%
Liverpool_Roads2.info()

# %%
print(Liverpool_Retail_Clusters.crs)
print(Liverpool_Roads2.crs)

# %%
#Assign a road to the POIs in clusters
intersect_POI = gpd.sjoin(Liverpool_Retail_Clusters, Liverpool_Roads2, how="inner", predicate="intersects")
intersect_POI.info()

# %%
#Group by road ID and DBSCAN cluster and count the number of shops in each group
count_POI = intersect_POI.groupby(['dClusters', 'ID']).size().reset_index(name='shop_count')
count_POI.info()

# %%
count_POI.describe()
#result: None of the streets have 10 shops of the same cluster (max is 9), decide on 5

# %%
#Filter to only include clusters with at least 5 shops on the same road
valid_streets = count_POI[count_POI['shop_count'] >= 5]
valid_streets.info()

# %% [markdown]
# ## Continue High Streets Identification

# %% [markdown]
# **Step 1:** Mark the high streets in the overall road data

# %%
#Create new column indicating whether or not a road is a high street
Liverpool_Roads2['is_high_street'] = Liverpool_Roads2['ID'].isin(valid_streets['ID'])

#Check 161 roads have been marked as 'high streets'
Liverpool_Roads2['is_high_street'].value_counts()

# %%
Liverpool_Roads2.head()

# %%
#Visualize the high streets

import folium

#Filter rows where 'is_high_street' is True
filtered_roads = Liverpool_Roads2[Liverpool_Roads2['is_high_street'] == True]

# Visualize the filtered roads using .explore
filtered_roads.explore(
    column="ID",
    cmap="hsv",)

# %% [markdown]
# **Step 2:** Eliminate streets that don't include the following categories:
# * 1 'Gambling'
# * 1 'Bus Transport'
# * 1 'Legal and Financial'
# * 1 ‘Historical and Cultural’
# * 1 ‘Health Practitioners and Establishments’

# %% [markdown]
# Issue after data exploration:
# * only  51 high streets identified with the clustering having 'Gambling' points in them
# * only have 104 high streets with the clustering having 'Bus Transport' points in them
# * only have 140 high streets with the clustering having 'Legal and Financial' points in them
# * only have 2 high streets with the clustering having 'Historical and Cultural' points in them
# * only have 63 high streets with the clustering having 'Health Practitioners and Establishments' points in them
# 
# Considering this we decide to keep High Streets with at least 1 Bus Transport point, 1 Legal and Financial, 1 Health Practitioners and Establishments

# %%
Liverpool_POI['categoryname'].unique()

# %%
#Filter POIs for Bus Transport
bus_data = Liverpool_POI[Liverpool_POI['categoryname'] == 'Bus Transport']

# %%
#Spatially join, to have just the road with Bus Transport points in them
roads_with_bus = gpd.sjoin(Liverpool_Roads2, bus_data, how='inner',predicate='intersects')

# %%
#Get list of IDs of roads with Bus Transport
roads_with_bus_ids = roads_with_bus['ID'].unique()

# %%
#Filter POIs for Health Practitioners and Establishments
health_data = Liverpool_POI[Liverpool_POI['categoryname'] == 'Health Practitioners and Establishments']

# %%
health_data['name'].unique()

# %%
#Spatially join, to have just the road with Health Practitioners and Establishments points in them
roads_with_health = gpd.sjoin(Liverpool_Roads2, health_data, how='inner',predicate='intersects')

# %%
#Get list of IDs of roads with Bus Transport
roads_with_health_ids = roads_with_health['ID'].unique()

# %%
#Filter POIs for Legal and Financial
legal_data = Liverpool_POI[Liverpool_POI['categoryname'] == 'Legal and Financial']

# %%
legal_data['name'].unique()

# %%
#Spatially join, to have just the road with Legal and Financial points in them
roads_with_legal = gpd.sjoin(Liverpool_Roads2, legal_data, how='inner',predicate='intersects')

# %%
#Get list of IDs of roads with Legal and Financial
roads_with_legal_ids = roads_with_legal['ID'].unique()
roads_with_legal_ids

# %%
#Update 'is_high_street' column

#Keep True for roads that are already marked as True and are present in all three lists
#Set to False for roads that are already marked as True but are not in all three lists
#Keep False for roads that are already marked as False regardless of the lists

Liverpool_Roads2['is_high_street'] = Liverpool_Roads2.apply(
    lambda row: True if row['is_high_street'] and
                (row['ID'] in roads_with_bus_ids and
                 row['ID'] in roads_with_health_ids and
                 row['ID'] in roads_with_legal_ids)
                else False if row['is_high_street'] else row['is_high_street'], axis=1)

# Check how many high streets we have now
print(Liverpool_Roads2['is_high_street'].value_counts())

# %% [markdown]
# Conclusion: there are 33 roads in Liverpool that we can label as 'High Streets' as they have:
# * at least 5 retail shops that are less than 150m away from ecah other
# * at least 1 'Bus Transport' facility
# * at least 1 'Legal and Financial'facility
# * at least 1 ‘Health Practitioners and Establishments’ facility

# %%
#Extract the high streets from Liverpool_Roads2
High_streets = Liverpool_Roads2[Liverpool_Roads2['is_high_street'] == True]
High_streets.info()

# %%
High_streets_ids = High_streets['ID'].unique()

# %%
#Create final dataset containing Liverpool's High Streets
Liverpool_High_Streets = Liverpool_Roads[Liverpool_Roads['ID'].isin(High_streets_ids)]
Liverpool_High_Streets = Liverpool_High_Streets.drop(columns= 'buffered_geometry')
# Check the result
Liverpool_High_Streets.info()

# %%
#Save final dataset
Liverpool_High_Streets.to_file("LiverPool_High_Streets.shp")

# %%
#plot
fig, ax = plt.subplots(1,1, figsize=(10,10), ) #set figure size
LiverpoolPol.plot(ax = ax, facecolor = 'lightgray') #plot Liverpool map
Liverpool_Roads.plot(ax=ax, color='lightblue')
Liverpool_High_Streets.plot(ax = ax, color = 'red')
ax.set_axis_off() #remove axes
fig.suptitle('High Streets in Liverpool') #add a title
plt.show()

# %% [markdown]
# ## Identification of High Streets in Bradford

# %% [markdown]
# ## Data exploration

# %%
#Load POI Bradford file
POIBradford = gpd.read_file('/Users/marion/Desktop/Etudes - Leeds/Semester 2/Creative Coding for Real World Problems/Topic 3/Hackathon 3 Data/poi_Bradford.gpkg')
POIBradford.info()

# %%
POIBradford.head()

# %%
#Upload Bradford Boundary
BradfordPol = gpd.read_file('/Users/marion/Desktop/Etudes - Leeds/Semester 2/Creative Coding for Real World Problems/Topic 3/Hackathon 3 Data/bradford_boundary/bradford_boundary.shp')
BradfordPol.plot()

# %%
#Check CRS of Bradford files
print(BradfordPol.crs)
print(POIBradford.crs)

# %%
#Change CRS
BradfordPol = BradfordPol.to_crs(epsg=27700)
print(BradfordPol.crs)

# %%
#Clip the POI to Bradford boundary
BradfordPol['outline'] = 1
BradfordPol_outline = BradfordPol.dissolve(by = 'outline')
Bradford_POI = gpd.clip(POIBradford, BradfordPol_outline)
Bradford_POI.plot()

# %%
Bradford_POI['groupname'].unique()

# %%
Bradford_POI['categoryname'].unique()

# %%
Bradford_POI['classname'].unique()

# %%
#Upload Bradford Roads
Bradford_Mer = gpd.read_file('/Users/marion/Desktop/Etudes - Leeds/Semester 2/Creative Coding for Real World Problems/Topic 3/Hackathon 3 Data/Roads Bradford/Meridian2_250305_125023.shp')
Bradford_Mer.plot()

# %%
print(Bradford_Mer.crs)

# %%
#Reproject data to EPSG:27700
Bradford_Mer = Bradford_Mer.to_crs(epsg=27700)
print(Bradford_Mer.crs)

# %%
#Clip the Bradford Roads to Bradford boundary
BradfordPol['outline2'] = 1
BradfordPol_outline = BradfordPol.dissolve(by = 'outline2')
Bradford_Roads = gpd.clip(Bradford_Mer, BradfordPol_outline)
Bradford_Roads.plot()

# %%
Bradford_Roads.head()

# %%
#Create unique identifier for the roads
Bradford_Roads['ID'] = range(1, len(Bradford_Roads) + 1)
Bradford_Roads.head()

# %%
#Extract retail POIs
Bradford_Retail = Bradford_POI[Bradford_POI['groupname'] == 'Retail']
#Get rid of columns with NAs which do not matter here
Bradford_Retail = Bradford_Retail.drop(columns=['uprn',
                                         'address_detail',
                                         'street_name',
                                         'locality',
                                         'telephone_number',
                                         'url',
                                         'brand',
                                         'qualifier_type',
                                         'qualifier_data'])
Bradford_Retail.info()

# %%
Bradford_Retail['categoryname'].unique()

# %% [markdown]
# ## Use DBSCAN to create clusters of at least 10 POIs that are maximum 150m away from each other

# %%
#Extract the coordinates from geometries
coords = Bradford_Retail.geometry.apply(lambda geom: [geom.x, geom.y]).tolist()

# %%
#DBSCAN
from sklearn.cluster import DBSCAN

#eps = how near points need to be to be included in the same cluster = 150m
#min_samples = smallest number of points we need to be able to call the collection of points a 'cluster' = 10
#metric='euclidean' = for the distance to be in meters
dbscan = DBSCAN(eps=150, min_samples=10, metric='euclidean')

# %%
#Fit the model to the coordinates of POIs
dbscan.fit(coords)

#assign the labels to a new variable
dbscan_labels = dbscan.labels_

#Add the clusters IDs as a new column and view the clusters
Bradford_Retail['dClusters'] = dbscan_labels
Bradford_Retail['dClusters'].value_counts()
#Result: 1228 POIs have not been clustered

# %%
#Drop all the POIs which are not in clusters (are in the -1 cluster)
Bradford_Retail_Clusters = Bradford_Retail[Bradford_Retail['dClusters'] != -1]

# %%
#Plot clusters
plt.scatter(Bradford_Retail_Clusters.geometry.x, Bradford_Retail_Clusters.geometry.y, c=Bradford_Retail_Clusters['dClusters'], cmap='viridis', marker='o')
plt.title("DBSCAN Clusters")
plt.show()

# %%
Bradford_Retail_Clusters.info()

# %%
Bradford_Retail_Clusters['dClusters'].unique()
#Result: Have 40 clusters

# %%
#Visualize where the clusters are on a map
import folium

Bradford_Retail_Clusters.explore(
    column="dClusters",
    cmap="hsv",
)

# %% [markdown]
# ## Identify streets with clusters and having at least 10 retail shops

# %%
#Create buffer column of roads (30m)
Bradford_Roads['buffered_geometry'] = Bradford_Roads.geometry.buffer(30)
#Assign the column buffered_geometry as the geometry to use
Bradford_Roads2 = Bradford_Roads.set_geometry('buffered_geometry')
Bradford_Roads2 = Bradford_Roads2.drop(columns= 'geometry')
Bradford_Roads2.head()

# %%
#Visualize where the buffered roads are on a map
import folium

Bradford_Roads2.explore(
    column="ID",
    cmap="hsv",
)

# %%
Bradford_Roads2.info()

# %%
print(Bradford_Retail_Clusters.crs)
print(Bradford_Roads2.crs)

# %%
#Assign a road to the POIs in clusters
intersect_POI = gpd.sjoin(Bradford_Retail_Clusters, Bradford_Roads2, how="inner", predicate="intersects")
intersect_POI.info()

# %%
#Group by road ID and DBSCAN cluster and count the number of shops in each group
count_POI = intersect_POI.groupby(['dClusters', 'ID']).size().reset_index(name='shop_count')
count_POI.info()

# %%
count_POI.describe()
#result: the majority (75%) of roads have less than 4 shops, decide on 5 for consistency

# %%
#Filter to only include clusters with at least 5 shops on the same road
valid_streets = count_POI[count_POI['shop_count'] >= 5]
valid_streets.info()

# %% [markdown]
# ## Continue High Streets Identification

# %% [markdown]
# **Step 1:** Mark the high streets in the overall road data

# %%
#Create new column indicating whether or not a road is a high street
Bradford_Roads2['is_high_street'] = Bradford_Roads2['ID'].isin(valid_streets['ID'])

#Check 149 roads have been marked as 'high streets'
Bradford_Roads2['is_high_street'].value_counts()

# %%
Bradford_Roads2.head()

# %%
#Visualize the high streets

import folium

#Filter rows where 'is_high_street' is True
filtered_roads = Bradford_Roads2[Bradford_Roads2['is_high_street'] == True]

# Visualize the filtered roads using .explore
filtered_roads.explore(
    column="ID",
    cmap="hsv",)

# %% [markdown]
# **Step 2:** Eliminate roads that don't include the following categories:
# * 1 'Bus Transport'
# * 1 'Legal and Financial'
# * 1 ‘Health Practitioners and Establishments’

# %%
Bradford_POI['categoryname'].unique()

# %%
#Filter POIs for Bus Transport
bus_data = Bradford_POI[Bradford_POI['categoryname'] == 'Bus Transport']

# %%
#Spatially join, to have just the roads with Bus Transport points in them
roads_with_bus = gpd.sjoin(Bradford_Roads2, bus_data, how='inner',predicate='intersects')

# %%
#Get list of IDs of roads with Bus Transport
roads_with_bus_ids = roads_with_bus['ID'].unique()

# %%
#Filter POIs for Health Practitioners and Establishments
health_data = Bradford_POI[Bradford_POI['categoryname'] == 'Health Practitioners and Establishments']

# %%
#Spatially join, to have just the roads with Health Practitioners and Establishments points in them
roads_with_health = gpd.sjoin(Bradford_Roads2, health_data, how='inner',predicate='intersects')

# %%
#Get list of IDs of roads with Bus Transport
roads_with_health_ids = roads_with_health['ID'].unique()

# %%
#Filter POIs for Legal and Financial
legal_data = Bradford_POI[Bradford_POI['categoryname'] == 'Legal and Financial']

# %%
#Spatially join, to have just the roads with Legal and Financial points in them
roads_with_legal = gpd.sjoin(Bradford_Roads2, legal_data, how='inner',predicate='intersects')

# %%
#Get list of IDs of roads with Legal and Financial
roads_with_legal_ids = roads_with_legal['ID'].unique()
roads_with_legal_ids

# %%
#Update 'is_high_street' column

#Keep True for roads that are already marked as True and are present in all three lists
#Set to False for roads that are already marked as True but are not in all three lists
#Keep False for roads that are already marked as False regardless of the lists

Bradford_Roads2['is_high_street'] = Bradford_Roads2.apply(
    lambda row: True if row['is_high_street'] and
                (row['ID'] in roads_with_bus_ids and
                 row['ID'] in roads_with_health_ids and
                 row['ID'] in roads_with_legal_ids)
                else False if row['is_high_street'] else row['is_high_street'], axis=1)

# Check how many high streets we have now
print(Bradford_Roads2['is_high_street'].value_counts())

# %% [markdown]
# Conclusion: there are 42 roads in Bradford that we can label as 'High Streets' as they have:
# * at least 5 retail shops that are less than 150m away from ecah other
# * at least 1 'Bus Transport' facility
# * at least 1 'Legal and Financial'facility
# * at least 1 ‘Health Practitioners and Establishments’ facility

# %%
#Extract the high streets from Bradford_Roads2
High_streets = Bradford_Roads2[Bradford_Roads2['is_high_street'] == True]
High_streets.info()

# %%
High_streets_ids = High_streets['ID'].unique()

# %%
#Create final dataset containing Bradford's High Streets
Bradford_High_Streets = Bradford_Roads[Bradford_Roads['ID'].isin(High_streets_ids)]
Bradford_High_Streets = Bradford_High_Streets.drop(columns= 'buffered_geometry')
# Check the result
Bradford_High_Streets.info()

# %%
#Save final dataset
Bradford_High_Streets.to_file("Bradford_High_Streets.shp")

# %%
#plot
fig, ax = plt.subplots(1,1, figsize=(10,10), ) #set figure size
BradfordPol.plot(ax = ax, facecolor = 'lightgray') #plot Bradford map
Bradford_Roads.plot(ax=ax, color='lightblue')
Bradford_High_Streets.plot(ax = ax, color = 'red')
ax.set_axis_off() #remove axes
fig.suptitle('High Streets in Bradford')
plt.show()

# %% [markdown]
# # Plots for Presentation Slides

# %%
#Convert CRS from 27700 to 4326 to extract coordinates
Liverpool_High_Streets_4326 = Liverpool_High_Streets.to_crs(epsg=4326)
print(Liverpool_High_Streets_4326.crs)

# %%
import folium

#Create map centered on Liverpool coordinates (53.403553, -2.986844)
#cartodb positron to have neutral background map
m = folium.Map(location=[53.403553, -2.986844], zoom_start=13, tiles="cartodb positron")

#Extract high streets LineString geometries
streets_geom = Liverpool_High_Streets_4326[Liverpool_High_Streets_4326.geometry.geom_type == 'LineString']

#Loop through each high street geometry and extract coordinates
for idx, geometry in streets_geom.iterrows():
    coordinates = list(geometry.geometry.coords)  # Extract coordinates from LineString
    #Invert coordinates
    coordinates = [(lat, lon) for lon, lat in coordinates]
    #Debugging: print coordinates to check if they are correct
    print(f"Street {idx} Coordinates: {coordinates}")

    #Add lines to map
    folium.PolyLine(coordinates, color="red", weight=2.5, opacity=1).add_to(m)

#Add Liverpool boundaries
folium.GeoJson(LiverpoolPol,
    style_function=lambda x: {
        'fillColor': 'lightblue',
        'weight': 2,
        'opacity': 0.2}
).add_to(m)




#Show map
m

# %%
#Convert CRS from 27700 to 4326 to extract coordinates
Bradford_High_Streets_4326 = Bradford_High_Streets.to_crs(epsg=4326)
print(Bradford_High_Streets_4326.crs)

# %%
import folium

#Create map centered on Bradford coordinates (53.403553, -2.986844)
#cartodb positron to have neutral background map
m = folium.Map(location=[53.799999, -1.750000], zoom_start=13, tiles="cartodb positron")

#Extract high streets LineString geometries
streets_geom = Bradford_High_Streets_4326[Bradford_High_Streets_4326.geometry.geom_type == 'LineString']

#Loop through each high street geometry and extract coordinates
for idx, geometry in streets_geom.iterrows():
    coordinates = list(geometry.geometry.coords)  # Extract coordinates from LineString
    #Invert coordinates
    coordinates = [(lat, lon) for lon, lat in coordinates]
    #Debugging: print coordinates to check if they are correct
    print(f"Street {idx} Coordinates: {coordinates}")

    #Add lines to map
    folium.PolyLine(coordinates, color="red", weight=2.5, opacity=1).add_to(m)

#Add Liverpool boundaries
folium.GeoJson(BradfordPol,
    style_function=lambda x: {
        'fillColor': 'lightblue',
        'weight': 2,
        'opacity': 0.2}
).add_to(m)




#Show map
m

# %% [markdown]
# ## Make example high street map (North Parade in Bradford)

# %%
bus_data_4326 = bus_data.to_crs(epsg=4326)

# %%
legal_data_4326 = legal_data.to_crs(epsg=4326)

# %%
health_data_4326 = health_data.to_crs(epsg=4326)

# %%
retail_data_4326 = Bradford_Retail.to_crs(epsg=4326)

# %%
#Color blind colors: '#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00'

# %%
import folium

#Create map centered on Bradford coordinates (53.403553, -2.986844)
#cartodb positron to have neutral background map
m = folium.Map(location=[53.799999, -1.750000], zoom_start=13, tiles="cartodb positron")

#Extract high streets LineString geometries
streets_geom = Bradford_High_Streets_4326[Bradford_High_Streets_4326.geometry.geom_type == 'LineString']

#Loop through each high street geometry and extract coordinates
for idx, geometry in streets_geom.iterrows():
    coordinates = list(geometry.geometry.coords)  # Extract coordinates from LineString
    #Invert coordinates
    coordinates = [(lat, lon) for lon, lat in coordinates]

    #Add lines to map
    folium.PolyLine(coordinates, color="red", weight=2.5, opacity=1).add_to(m)


#Add Retail POIs to map
for idx, geometry in retail_data_4326.iterrows():
    point = geometry.geometry
    latitude, longitude = point.y, point.x  #Extract coordinates

    #Add point
    folium.Marker(location=[latitude, longitude],
        popup=f"Point {idx}",
        icon=folium.Icon(color='pink', icon='fa-cart-plus', prefix='fa')).add_to(m)
    
#Add Bus Transport POIs to map
for idx, geometry in bus_data_4326.iterrows():
    point = geometry.geometry
    latitude, longitude = point.y, point.x  #Extract coordinates

    #Add point
    folium.Marker(location=[latitude, longitude],
        popup=f"Point {idx}",
        icon=folium.Icon(color='orange', icon='fa-bus', prefix='fa')).add_to(m)


#Add Financial-Legal POIs to map
for idx, geometry in legal_data_4326.iterrows():
    point = geometry.geometry
    latitude, longitude = point.y, point.x  #Extract coordinates
    #Add point
    folium.Marker(location=[latitude, longitude],
        popup=f"Point {idx}",
        icon=folium.Icon(color='blue', icon='fa-university', prefix='fa')).add_to(m)

#Add Health POIs to map
for idx, geometry in health_data_4326.iterrows():
    point = geometry.geometry
    latitude, longitude = point.y, point.x  #Extract coordinates

    #Add point
    folium.Marker(location=[latitude, longitude],
        popup=f"Point {idx}",
        icon=folium.Icon(color='green', icon='fa-medkit', prefix='fa')).add_to(m)
    

#Show map
m


