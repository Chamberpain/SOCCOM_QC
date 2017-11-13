import matplotlib.pyplot as plt
import pandas as pd
import sys,os
sys.path.append(os.path.abspath("../"))
import soccom_proj_settings
import numpy as np 
from scipy import interpolate
from mpl_toolkits.basemap import Basemap
import scipy.io
from collections import OrderedDict

#Need to program a more reasonable number of pressure values to interpolate

plot_float_stats = True
plot_profile_compare = False
pressure_list = np.array(range(20,2000,40))
variable_list = [('Temperature',0.4),('Salinity',0.2),('Nitrate',8),('Oxygen',30),('OxygenSat',15),('pH25C',0.03),('pHinsitu',0.03),('TALK_MLR',15),('DIC_MLR',30)] # the second value in each of these tuples is chosen as half a std of the field at 500m
quality_cruise_list = ['P14S','P15S','P16S','P18','P17E','SR03','SR04','I09S','I08S','S04I','S04P','I07S','I06S','SR01','A23','A12','A13.5']
distance_threshold = 35
day_threshold = 45

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
               try:
                    interp_list_token = pressure_list[(pressure_list>dataframe_token['Pressure'].min())&(pressure_list<dataframe_token['Pressure'].max())]
                    df_holder.loc[df_holder.Pressure.isin(interp_list_token),variable]=dummy_func(interp_list_token)
               except NameError:
                    return pd.DataFrame()
     df_holder['Date']=date
     df_holder['Cruise'] = cruise
     df_holder['Lat']=lat
     df_holder['Lon']=lon
     return df_holder

# df = pd.concat([pd.read_pickle(soccom_proj_settings.soccom_drifter_file),pd.read_pickle(soccom_proj_settings.goship_file)]) #this is the full SOCCOM data file with all profiles
df_soccom = pd.read_pickle(soccom_proj_settings.soccom_drifter_file)
df_goship = pd.read_pickle(soccom_proj_settings.goship_file)
df_goship['TALK_MLR']=df_goship['Alkalinity']
df_goship.loc[df_goship.Oxygen<40,'Oxygen']=np.nan  #oxygen = 0 is the nan case for a subset of these data. Also, there are a bunch of seemingly unphysical oxygen values (different units perhaps??)

# cruise_list = df_goship.Cruise.unique()
# cruise_list = [i for e in quality_cruise_list for i in cruise_list if e in i] 
# df_goship = df_goship[df_goship.Cruise.isin(cruise_list)]
df = pd.concat([df_soccom,df_goship])

df_save = pd.read_pickle('./intercompare_floats.pickle') #this is the list of all float profiles that occur in proximity to one another
df = df[df.Pressure<2000]

if plot_float_stats:
     df_plot = df_save[df_save.distance<60]
     df_plot['distance'].hist(bins=50,cumulative=True,label='All Occurances') #histogram of all the distances
     df_plot[(df_plot['Date']-df_plot['Date Compare']).dt.days.abs().mod(365)<day_threshold]['distance'].hist(bins=50,cumulative=True,label='Within Seasonal Criteria')
     plt.xlabel('Distance (km)')
     plt.title('Number of occurrances of profile intersections by distance')
     plt.legend()
     plt.savefig('./float_intercompare_plots/number_floats_distance')
     plt.close()

     plt.figure()
     df_plot = df_save[df_save.distance<distance_threshold] #restrict to profiles that are within distance threshold of one another
     (df_plot['Date']-df_plot['Date Compare']).dt.days.abs().mod(365).hist(bins=20,cumulative=True)
     plt.xlabel('Days')
     plt.title('Difference in days for all floats less than '+str(distance_threshold)+'km')
     plt.savefig('./float_intercompare_plots/number_floats_days')
     plt.close()

     # plt.figure()
     # m = Basemap(projection='spstere',boundinglat=-40,lon_0=180,resolution='l')
     # m.drawcoastlines(linewidth=1.5)
     # m.fillcontinents(zorder=8)
     # line_space = 20
     # fontsz=10
     # parallels = np.arange(-90,0,line_space/2)
     # m.drawparallels(parallels,labels=[1,0,0,0],fontsize=fontsz)
     # meridians = np.arange(-360.,360.,line_space)
     # m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fontsz)
     # xx,yy = np.meshgrid(XC,YC)
     # XX,YY = m(xx,yy)
     # lev = np.max([abs(np.nanmin(plot_data)),abs(np.nanmax(plot_data))])
     # levels = np.linspace(-lev,lev,20)
     # cs = m.contour(XX,YY,plot_data,levels,linewidths=0.5,colors='k',animated=True,alpha=0.4)
     # ca = m.contourf(XX,YY,plot_data,levels,cmap=plt.cm.RdBu_r,animated=True,alpha=0.4)

     # lat_list, lon_list = zip(*df_save[['Lat Compare','Lon Compare']].drop_duplicates().values)
     # lat_list1, lon_list1 = zip(*df_save[['Lat','Lon']].drop_duplicates().values)

     # y = lat_list+lat_list1 
     # x = lon_list+lon_list1

     # X,Y = m(x,y)
     # m.scatter(X,Y,marker='*',c='gold',s=100,linewidths=1,edgecolors='k')
     # plt.title('SSH and Float Location')

     # plt.savefig('./float_intercompare_plots/map_all_floats')
     # plt.close()


df_plot = df_save[df_save.distance<distance_threshold] #restrict to profiles that are within distance_threshold of one another
df_plot = df_plot[(df_plot['Date']-df_plot['Date Compare']).dt.days.abs().mod(365)<day_threshold] #this makes the maximum seasonal time difference
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

frames = []
frames_goship = []
plot_num = 0
for cruise in df_plot['Cruise Compare'].unique():
     print cruise
     for date in df_plot[df_plot['Cruise Compare']==cruise]['Date Compare'].unique():
          mask = (df_plot['Date Compare']==date)&(df_plot['Cruise Compare']==cruise)
          plot_flag = {}
          for row in df_plot[mask].iterrows():
               date1 = row[1]['Date']
               date2 = row[1]['Date Compare']
               cruise1 = row[1]['Cruise']
               cruise2 = row[1]['Cruise Compare']

               df1 = df_compare[(df_compare.Date==date1)&(df_compare.Cruise==cruise1)] 
               df2 = df_compare[(df_compare.Date==date2)&(df_compare.Cruise==cruise2)] #grab the interpoalted profiles for the specified dates and cruises

               df1 = df1[df1.Pressure.isin(df2.Pressure)]
               df2 = df2[df2.Pressure.isin(df1.Pressure)]

               df_dummy = df1[list(zip(*variable_list)[0])]-df2[list(zip(*variable_list)[0])]
               df_dummy['Pressure']=df1.Pressure
               frames.append(df_dummy)

               if row[1]['Type']=='GOSHIP':
                    frames_goship.append(df_dummy)

               df_dummy = df_dummy[df_dummy.Pressure>400].max()
               for variable,tolerance in variable_list:
                    if df_dummy.abs()[variable]>tolerance:
                         plot_flag[variable]=True
          if plot_profile_compare:
               if plot_flag:

                    cruise_list = df_plot[mask]['Cruise'].tolist()+[df_plot[mask]['Cruise Compare'].tolist()[0]]
                    date_list = df_plot[mask]['Date'].tolist()+[df_plot[mask]['Date Compare'].tolist()[0]]

                    lat_list = df_plot[mask]['Lat'].tolist()+[df_plot[mask]['Lat Compare'].tolist()[0]]
                    lon_list = df_plot[mask]['Lon'].tolist()+[df_plot[mask]['Lon Compare'].tolist()[0]]

                    for variable in plot_flag:
                         plt.figure(figsize=(13,5))
                         plt.subplot(1,2,1)
                         for date,cruise in zip(date_list,cruise_list):
                              df_holder = df[(df.Date==date)&(df.Cruise==cruise)]
                              if (df_holder[variable].isnull()).all():
                                   continue
                              if type(cruise) is int:
                                   plt.plot(df_holder.dropna(subset=[variable])[variable].values,df_holder.dropna(subset=[variable]).Pressure.values,color='wheat',label='GOSHIP',alpha=0.65)
                              else:
                                   plt.plot(df_holder.dropna(subset=[variable])[variable].values,df_holder.dropna(subset=[variable]).Pressure.values,linewidth=3,label=cruise+' on '+date.strftime(format='%Y-%m-%d'))

                         plt.gca().invert_yaxis()
                         plt.title(variable)
                         handles, labels = plt.gca().get_legend_handles_labels()
                         by_label = OrderedDict(zip(labels, handles))
                         plt.legend(by_label.values(), by_label.keys())
                         
                         plt.subplot(1,2,2)
                         m = Basemap(projection='spstere',boundinglat=-40,lon_0=180,resolution='l')
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
                         y = lat_list
                         x = lon_list

                         X,Y = m(x,y)
                         m.scatter(X,Y,marker='*',c='gold',s=260,linewidths=2,edgecolors='k')
                         plt.title('SSH and Float Location')

                         plt.savefig('./float_intercompare_plots/'+str(plot_num)+variable)
                         plt.close()
                         plot_num +=1

if plot_float_stats:
     df = pd.concat(frames)  #this is the dataframe with the total fleet differences
     plt.figure(figsize=(13,13))
     for n,variable in enumerate(list(zip(*variable_list)[0])):          
          plt.subplot(3,3,(n+1))
          for pres in df.Pressure.unique():
               series_plot = df[df.Pressure==pres][variable]
               plt.errorbar(series_plot.mean(), pres, xerr=series_plot.std(),fmt='o',color='b')
               plt.title(variable)
          plt.gca().invert_yaxis()
          if n in [0,3,6]:
               plt.ylabel('Pressure (db)')
     plt.suptitle('Average and Standard Deviation of All Profile Discrepancy')
     plt.savefig('./float_intercompare_plots/intercompare_stats')
     plt.close()

     df = pd.concat(frames_goship)  #this is the dataframe with the total fleet differences
     plt.figure(figsize=(13,13))
     for n,variable in enumerate(list(zip(*variable_list)[0])):          
          plt.subplot(3,3,(n+1))
          for pres in df.Pressure.unique():
               series_plot = df[df.Pressure==pres][variable]
               plt.errorbar(series_plot.mean(), pres, xerr=series_plot.std(),fmt='o',color='b')
               plt.title(variable)
          plt.gca().invert_yaxis()
          if n in [0,3,6]:
               plt.ylabel('Pressure (db)')
     plt.suptitle('Average and Standard Deviation of SOCCOM Float to GOSHIP Profile Discrepancy')
     plt.savefig('./float_intercompare_plots/intercompare_stats_goship')
     plt.close()