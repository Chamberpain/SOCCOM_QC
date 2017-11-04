import matplotlib.pyplot as plt
import pandas as pd
import sys,os
sys.path.append(os.path.abspath("../"))
import soccom_proj_settings
import numpy as np 
from scipy import interpolate
from mpl_toolkits.basemap import Basemap
#Need to program a more reasonable number of pressure values to interpolate

plot_float_stats = True
pressure_list = np.array(range(20,2000,40))
variable_list = [('Temperature',0.3),('Salinity',0.1),('Nitrate',20),('Oxygen',20),('pH25C',0.03)] # the second value in each of these tuples is chosen as half a std of the field at 500m

def interpolator(dataframe):
     interpolation_list = pressure_list[(pressure_list>dataframe['Pressure'].min())&(pressure_list<dataframe['Pressure'].max())] 
# The interpolation list must have a smaller range than the pressures that are used to create the interpolation function (no extrapolation)
     lat,lon,cruise,date = dataframe[:1][['Lat','Lon','Cruise','Date']].values[0] #strip out the lat and lon
     df_holder = pd.DataFrame() 
     df_holder['Pressure']=interpolation_list 
     for variable,dummy in variable_list: #loop through all variables
          dataframe_token = dataframe.dropna(subset=[variable])
          if dataframe_token.empty: #some floats only contain a subset of the sensors. if dataframe_token is empty, just continue on to other variables
               continue
          try:
               dummy_func = interpolate.interp1d(dataframe_token['Pressure'].values,dataframe_token[variable].values, kind = 'slinear')
               df_holder[variable] = dummy_func(interpolation_list)
          except ValueError: #because some of the BGC variables are not collected at all depths, there is occasionally a size mismatch. We add this excemption so that we dont have to 
          # create an interpolation list for every depth value (which would be expensive)
               interp_list_token = pressure_list[(pressure_list>dataframe_token['Pressure'].min())&(pressure_list<dataframe_token['Pressure'].max())]
               df_holder.loc[df_holder.Pressure.isin(interp_list_token),variable]=dummy_func(interp_list_token)
     df_holder['Date']=date
     df_holder['Cruise'] = cruise
     df_holder['Lat']=lat
     df_holder['Lon']=lon
     return df_holder

df = pd.read_pickle(soccom_proj_settings.soccom_drifter_file) #this is the full SOCCOM data file with all profiles
df_save = pd.read_pickle('./intercompare_floats.pickle') #this is the list of all float profiles that occur in proximity to one another

if plot_float_stats:
     df_save['distance'].hist(bins=20) #histogram of all the distances
     plt.xlabel('Distance (km)')
     plt.xlim([0,600]) # because we are binning the profiles in 5 degree boxes the area of each bin decreases after a certain threshhold. We limit the plot to 600km because otherwise the plot gives a non-intuitive result
     plt.title('Number of occurrances of float intersection by distance')
     plt.figure()
     df_plot = df_save[df_save.distance<50] #restrict to profiles that are within 50km of one another
     (df_plot['Date']-df_plot['Date Compare']).dt.days.abs().hist(bins=20)
     plt.xlabel('Days')
     plt.title('Difference in days for all floats less than 50km')

df_plot = df_plot[(df_plot['Date']-df_plot['Date Compare']).dt.days.abs().mod(365)<45] #this makes the maximum seasonal time difference 45 days
float_list = df_plot.Cruise.tolist()+df_plot['Cruise Compare'].tolist()
date_list = df_plot.Date.tolist()+df_plot['Date Compare'].tolist()
df = df[(df.Date.isin(date_list))&(df.Cruise.isin(float_list))] # whittle down the dataframe a little for easier processing. Only floats and dates that are in intercomparison list.

frames = []
for cruise in df.Cruise.unique(): 
     print cruise
     for date in df[df.Cruise==cruise].Date.unique():
          print date
          frames.append(interpolator(df[(df.Date==date)&(df.Cruise==cruise)])) #interpolate to common pressure values so that we can effectively compare values and filter outliers
df_compare=pd.concat(frames)


local_root = '/Users/paulchamberlain/Data/SOSE/'
ssh = np.load(os.path.join(local_root,'sshave.npy'))
mat = scipy.io.loadmat(os.path.join(local_root,'grid.mat'))
plot_data = np.ma.masked_equal(ssh,0.)
XC = mat['XC'][:,0]
YC = mat['YC'][0,:]


for row in df_plot.iterrows():
     date1 = row[1]['Date']
     date2 = row[1]['Date Compare']
     cruise1 = row[1]['Cruise']
     cruise2 = row[1]['Cruise Compare']
     
     df1 = df_compare[(df_compare.Date==date1)&(df_compare.Cruise==cruise1)] 
     df2 = df_compare[(df_compare.Date==date2)&(df_compare.Cruise==cruise2)] #grab the interpoalted profiles for the specified dates and cruises

     for variable,difference_threshold in variable_list: 
          compare_series = (df1[df1.Pressure>400][variable]-df2[df2.Pressure>400][variable])
          compare_series = compare_series.abs() #absolute value of the difference below 400 m (this should be relatively stable, so differences will indicate differences in sensors.)
          if (compare_series>difference_threshold).any():
               print '###### I found a questionable one #########'
               plt.figure(figsize=(13,5))
               plt.subplot(1,2,1)
               df_holder = df[(df.Date==date1)&(df.Cruise==cruise1)]
               plt.plot(df_holder.dropna(subset=[variable])[variable].values,df_holder.dropna(subset=[variable]).Pressure.values,label=cruise1+' on '+date1.strftime(format='%Y-%m-%d'))
               df_holder = df[(df.Date==date2)&(df.Cruise==cruise2)]
               plt.plot(df_holder.dropna(subset=[variable])[variable].values,df_holder.dropna(subset=[variable]).Pressure.values,label=cruise2+' on '+date2.strftime(format='%Y-%m-%d'))
               plt.gca().invert_yaxis()
               plt.title(variable)
               plt.legend()
               plt.subplot(1,2,2)
               m = Basemap(projection='spstere',boundinglat=-50,lon_0=180,resolution='l')
               m.drawcoastlines(linewidth=1.5)
               m.fillcontinents(zorder=8)
               line_space = 20
               fontsz=10
               parallels = np.arange(-90,0,line_space/2)
               m.drawparallels(parallels,labels=[1,0,0,0],fontsize=fontsz)
               meridians = np.arange(-360.,360.,line_space)
               m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fontsz)
               xx,yy = np.meshgrid(XC,YC)
               XX,YY = m(xx,yy)
               lev = np.max([abs(np.nanmin(plot_data)),abs(np.nanmax(plot_data))])
               levels = np.linspace(-lev,lev,20)
               cs = m.contour(XX,YY,plot_data,levels,linewidths=0.5,colors='k',animated=True,alpha=0.4)
               ca = m.contourf(XX,YY,plot_data,levels,cmap=plt.cm.RdBu_r,animated=True,alpha=0.4)
               y = [row[1]['Lat'],row[1]['Lon Compare']]
               x = [row[1]['Lon'],row[1]['Lat Compare']]

               X,Y = m(x,y)
               m.scatter(X,Y,marker='*',c='gold',s=260,linewidths=2,edgecolors='k')


               # plt.colorbar(ca)
               # plt.clabel(cs,inline=1,fontsize=6)




               plt.savefig('./float_intercompare_plots/'+str(row[0])+variable)
               plt.close()