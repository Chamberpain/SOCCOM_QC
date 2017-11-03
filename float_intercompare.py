import numpy as np
import sys,os
sys.path.append(os.path.abspath("../"))
import soccom_proj_settings
import pandas as pd
from LatLon import LatLon
import oceans 

#need to add functionality to read in and compare WOES cruise data and SOCCAT data 



def assign_pos(row):
	row['LatLon'] = LatLon(row['Lat'],row['Lon']) #this is necessary because LatLon must be fed one tuple at a time
	return row

def distance_calc(row):
	row['distance'] = (row['LatLon']-pos_token).magnitude #calculate the distance of each profile to the position in the loop.
	return row

def subsample_df(holder_):
# we subsmaple the dataframe to make more efficient work of our position compares. It is relatively expensive to calculate the distance from one profile to all other profiles in the 
# dataset, so instead we bin them into 5 degree boxes and only take the bins on either side.
	lon_index = (lon_bins == holder_['Lon Cut'].values[0]).tolist().index(True)
	try:
		lon_list = [lon_bins[lon_index-1],lon_bins[lon_index],lon_bins[lon_index+1]]
	except IndexError:
		lon_list = [lon_bins[lon_index-1],lon_bins[lon_index],lon_bins[0]] #this is included because we will exceed index dimensions if it is at the end of the list

	lat_index = (lat_bins == holder_['Lat Cut'].values[0]).tolist().index(True)
	try:
		lat_list = [lat_bins[lat_index-1],lat_bins[lat_index],lat_bins[lat_index+1]]
	except IndexError:
		lat_list = [lat_bins[lat_index-1],lat_bins[lat_index],lat_bins[0]] #this is included because we will exceed index dimensions if it is at the end of the list

	return df[(df['Lat Cut'].isin(lat_list))&(df['Lon Cut'].isin(lon_list))]


df = pd.read_pickle(soccom_proj_settings.soccom_drifter_file) #read in all the SOCCOM profiles
df = df.drop_duplicates(subset=['Lat','Lon','Cruise','Date'])[['Lat','Lon','Cruise','Date']] #drop everything else but one position value so that the code runs faster
df = df.dropna(subset=['Lat','Lon'])
df.Lon = oceans.wrap_lon180(df.Lon)

# bin the data, so that we can more efficiently check the distances
df['Lon Cut'] = pd.cut(df.Lon,range(-180,185,5),include_lowest=True)
df['Lat Cut'] = pd.cut(df.Lat,range(-90,0,5),include_lowest=True)
lon_bins = (df['Lon Cut'].values).categories
lat_bins = (df['Lat Cut'].values).categories


frames = []
df = df.apply(assign_pos,axis=1)
for cruise in df.Cruise.unique():
	print cruise
	holder = df[df.Cruise==cruise] #select only the looped cruise from the dataframe
	df = df[df.Cruise!=cruise] # remove the looped cruise (not interested in comparing the selected float profiles to each other)
	for date in holder.Date.unique():
		holder_token = holder[holder.Date==date]	#loop through all dates of each SOCCOM profile
		pos_token = holder_token.LatLon.values[0]
		df_truth = subsample_df(holder_token) #returns a dataframe geographically sliced in 5 degree bins on either side north-south and east-west
		if not df_truth.empty:
			df_output = df_truth.apply(distance_calc,axis=1) #if there is anything in this bin, calculate the distance to all the floats
			df_output['Date Compare']=holder_token['Date'].values[0] #and save the dates, float names, and lat and lons
			df_output['Cruise Compare']=holder_token['Cruise'].values[0]
			df_output['Lat Compare']=holder_token['Lat'].values[0]
			df_output['Lon Compare']=holder_token['Lon'].values[0]
			frames.append(df_output[['Date','Cruise','Lat','Lon','Date Compare','Cruise Compare','Lat Compare','Lon Compare','distance']]) 
			#then append to frames (this is done because it is orders of magnitude faster for pandas)
df_save = pd.concat(frames) #create large dataframe from the list of dataframes 
df_save.to_pickle('./intercompare_floats.pickle') # and save. 
