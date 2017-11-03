import matplotlib.pyplot as plt
import pandas as pd
import sys,os
sys.path.append(os.path.abspath("../"))
import soccom_proj_settings
import numpy as np 
from scipy import interpolate

pressure_list = np.array(range(20,2000,40))
variable_list = [('Temperature',0.3),('Salinity',0.1),('Nitrate',20),('Oxygen',20),('pH25C',0.03)] # the second value in each of these tuples is chosen as half a std of the field at 500m

def interpolator(dataframe):
     interpolation_list = pressure_list[(pressure_list>dataframe['Pressure'].min())&(pressure_list<dataframe['Pressure'].max())]
     lat,lon,cruise,date = dataframe[:1][['Lat','Lon','Cruise','Date']].values[0]
     df_holder = pd.DataFrame() #populate the index rows with the desired depths
     df_holder['Pressure']=interpolation_list
     for variable,dummy in variable_list:
          dataframe_token = dataframe.dropna(subset=[variable])
          if dataframe_token.empty:
               continue
          try:
               dummy_func = interpolate.interp1d(dataframe_token['Pressure'].values,dataframe_token[variable].values, kind = 'slinear')
               df_holder[variable] = dummy_func(interpolation_list)
          except ValueError:
               interp_list_token = pressure_list[(pressure_list>dataframe_token['Pressure'].min())&(pressure_list<dataframe_token['Pressure'].max())]
               df_holder.loc[df_holder.Pressure.isin(interp_list_token),variable]=dummy_func(interp_list_token)
     df_holder['Date']=date
     df_holder['Cruise'] = cruise
     df_holder['Lat']=lat
     df_holder['Lon']=lon
     return df_holder




df = pd.read_pickle(soccom_proj_settings.soccom_drifter_file)
df_save = pd.read_pickle('./intercompare_floats.pickle')
df_save['distance'].hist(bins=20) 
plt.xlabel('Distance (km)')
plt.xlim([0,600])
plt.title('Number of occurrances of float intersection by distance')
plt.figure()
df_plot = df_save[df_save.distance<50]
(df_plot['Date']-df_plot['Date Compare']).dt.days.abs().hist(bins=20)
plt.xlabel('Days')
plt.title('Difference in days for all floats less than 50km')


float_list = df_plot.Cruise.tolist()+df_plot['Cruise Compare'].tolist()
date_list = df_plot.Date.tolist()+df_plot['Date Compare'].tolist()
df = df[(df.Date.isin(date_list))&(df.Cruise.isin(float_list))]
df_plot = df_plot[(df_plot['Date']-df_plot['Date Compare']).dt.days.abs().mod(365)<45] #this makes the maximum seasonal time difference 45 days

frames = []
for cruise in df.Cruise.unique(): 
     print cruise
     for date in df[df.Cruise==cruise].Date.unique():
          print date
          frames.append(interpolator(df[(df.Date==date)&(df.Cruise==cruise)]))
df_compare=pd.concat(frames)


for row in df_plot.iterrows():
     date1 = row[1]['Date']
     date2 = row[1]['Date Compare']
     cruise1 = row[1]['Cruise']
     cruise2 = row[1]['Cruise Compare']
     
     df1 = df_compare[(df_compare.Date==date1)&(df_compare.Cruise==cruise1)]
     df2 = df_compare[(df_compare.Date==date2)&(df_compare.Cruise==cruise2)]

     for variable,difference_threshold in variable_list:
          compare_series = (df1[df1.Pressure>400][variable]-df2[df2.Pressure>400][variable])
          compare_series = compare_series.abs()
          if (compare_series>difference_threshold).any():
               plt.figure()
               df_holder = df[(df.Date==date1)&(df.Cruise==cruise1)]
               plt.plot(df_holder.dropna(subset=[variable])[variable].values,df_holder.dropna(subset=[variable]).Pressure.values,label=cruise1)
               df_holder = df[(df.Date==date2)&(df.Cruise==cruise2)]
               plt.plot(df_holder.dropna(subset=[variable])[variable].values,df_holder.dropna(subset=[variable]).Pressure.values,label=cruise2)
               plt.gca().invert_yaxis()
               plt.title(variable)
               plt.legend()
               plt.show()