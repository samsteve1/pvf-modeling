# Create pvlib
# python code to calculate and plot CS (clear sky) and
# TMY3 (typical meterological year) POA irradiance – daily
# single plots with both CS and TMY3.  Use the data for
# Toledo, 35°tilt and 180°azimuth, to generate a
# histogram in 10% intervals showing number of days vs.
# % fractional insolation for TMY3 vs. CS.
# Determine the
# full, annual fractional POA for TMY3/CS to determine
# the impact of clouds on PV energy generation.

import os
import numpy as np
import pandas as pd
from ClearSkyIrradiance import ClearSkyIrradiance
from pvlib.location import Location
from TMYIrradiance import TMYIrradiance
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler

plt.rc('ytick', labelsize=20)

year = "2022"
period = 24
frequency = '1h'
surface_tilt = 35  # set tilt to latitude in the class file
surface_azimuth = 180
model = 'simplified_solis'
coordinates = [
    (41.651031, -83.541939, 614, 'America/Detroit', 'Toledo', 'OH'),
    (44.98, -93.27, 256, 'America/Chicago', 'Minneapolis', 'MN'),
    (35.22, -101.71, 1098, 'America/Chicago', 'Amarillo', 'TX'),
    (42.49, -76.46, 330, 'America/Detroit',  'Ithaca', 'NY'),
    (43.615, -116.20, 823, 'America/Boise', 'Boise', 'ID'),
    (34.992, -117.54, 755, 'America/Los_Angeles', 'Kramer Junction', 'CA'),
    (21.310, -157.86, 5, 'Pacific/Honolulu', 'Honolulu', 'HI'),
    (64.84, -147.72, 132, 'America/Anchorage', 'Fairbanks', 'AK'),
    (31.025837, -87.506462, 86, 'America/Chicago', 'Atmore', 'AL'),
    (39.742043, -104.991531, 1607, 'America/Denver', 'Denver', 'CO'),
    (33.448376, -112.074036, 332, 'America/Phoenix', 'Phoenix', 'AZ'),
    (36.082157, -94.171852, 374, 'America/Chicago', 'Fayetteville', 'AR'),
    (41.76371110, -72.68509320, 18, 'America/Detroit', 'Hartford', 'CT'),
    (39.6293, -75.6583, 21, 'America/Detroit', 'Bear', 'DE'),
    (38.889805, -77.009056, 67, 'America/Detroit', 'Washington', 'DC'),
    (25.761681, -80.191788, 1.8, 'America/Detroit', 'Miami', 'FL'),
    (33.950001, -83.383331, 184, 'America/Detroit', 'Athens', 'GA'),
    (41.506699, -90.515137, 117, 'America/Chicago', 'Moline', 'IL'),
    (39.168804, -86.536659, 245, 'America/Detroit', 'Bloomington', 'IN'),
    (41.619549, -93.598022, 266, 'America/Chicago', 'Des Moines', 'IA'),
    (38.867, -101.964, 1174, 'America/Denver', 'Weskan', 'KS'),
    (37.88641, -84.65704, 295, 'America/Detroit', 'Lexington', 'KY'),
    (29.951065, -90.071533, -2, 'America/Chicago', 'New Orleans', 'LA'),
    (43.680031, -70.310425, 62, 'America/Detroit', 'Portland', 'ME'),
    (38.978443, -76.492180, 13, 'America/Detroit', 'Annapolis', 'MD'),
    (42.640999, -71.316711, 31, 'America/Detroit', 'Lowell', 'MA'),
    (44.7631, -85.6206, 140, 'America/Detroit', 'Traverse City', 'MI'),
    (32.298756, -90.184807, 85, 'America/Chicago', 'Jackson', 'MS'),
    (39.099724, -94.578331, 292, 'America/Chicago', 'Kansas City', 'MO'),
    (45.787636, -108.489304, 952, 'America/Denver', 'Billings', 'MT'),
    (40.806862, -96.681679, 347, 'America/Chicago', 'Lincoln', 'NE'),
    (36.188110, -115.176468, 620, 'America/Los_Angeles', 'Las Vegas', 'NV'),
    (43.2040, -71.5362, 260, 'America/Detroit', 'Concord', 'NH'),
    (39.370121, -74.438942, 2, 'America/Detroit', 'Atlantic City', 'NJ'),
    (35.106766, -106.629181, 2100, 'America/Chicago', 'Albuquerque', 'NM'),
    (35.250538, -75.528824, 8, 'America/Detroit', 'Cape Hatteras', 'NC'),
    (46.825905, -100.778275, 352, 'America/Chicago', 'Bismarck', 'ND'),
    (35.402485, -99.446182, 590, 'America/Chicago', 'Elk City', 'OK'),
    (42.224869, -121.781670, 1248, 'America/Los_Angeles', 'Klamath Falls', 'OR'),
    (40.613953, -75.477791, 103, 'America/Detroit', 'Allentown', 'PA'),
    (41.7175, -71.4092, 6, 'America/Detroit', 'Warwick', 'RI'),
    (32.776566, -79.930923, 6, 'America/Detroit', 'Charleston', 'SC'),
    (44.080544, -103.231018, 1023, 'America/Denver', 'Rapid City', 'SD'),
    (36.601, -86.717, 219, 'America/Chicago', 'Orlinda', 'TN'),
    (40.759926,  -111.884888, 1300, 'America/Denver', 'Salt Lake City', 'UT'),
    (44.475883, -73.212074, 61, 'America/Detroit', 'Burlington', 'VT'),
    (36.698750, -78.901398, 131, 'America/Detroit', 'South Boston', 'VA'),
    (46.602070, -120.505898, 325, 'America/Los_Angeles', 'Yakima', 'WA'),
    (38.349819, -81.632622, 182, 'America/Detroit', 'Charleston', 'WV'),
    (44.959633, -89.630600, 368, 'America/Chicago', 'Wausau', 'WI'),
    (42.89750, -106.47306, 1560, 'America/Denver', 'Casper', 'WY')
]

states = []
names = []
total_fractional_insolation = []

for location in coordinates:
    latitude, longitude, altitude, timezone, name, state = location

    file_path = 'C:/Users/abp6q/Documents/PVModeling/HW_5/files/tmy_clearsky_analysis/' + \
        state + '_' + name + '/'

    location = Location(latitude, longitude, tz=timezone,
                        altitude=altitude, name=name)

    clear_sky_irradiance = ClearSkyIrradiance(
        latitude, longitude, altitude, timezone, name, surface_tilt, surface_azimuth)

    tmy_irradiance = TMYIrradiance(
        latitude, longitude, altitude, timezone, name, surface_tilt, surface_azimuth)

    names.append(name)
    states.append(state)
    annual_TMY_POA = []
    annual_clearsky_POA = []
    fractional_insolation = []

    annual_clear_sky_irradiance = clear_sky_irradiance.get_annual_clear_sky_irradiance(
        year, frequency, period)

    annual_tmy_irradiance = tmy_irradiance.get_annual_tmy_irradiance(
        year, frequency, period)

    daily_TMY_POA_sum = []
    daily_CS_POA_sum = []

    for daily_clearsky_irradiance, daily_TMY3_irradiance in zip(annual_clear_sky_irradiance, annual_tmy_irradiance):
        days = daily_clearsky_irradiance.index
        daily_clearsky_irradiance.index = daily_clearsky_irradiance.index.strftime(
            "%H:%M")
        daily_TMY3_irradiance.index = daily_TMY3_irradiance.index.strftime(
            "%H:%M")

        # normalize TMY data
        nonzero_index = np.flatnonzero(daily_TMY3_irradiance['POA'].values)[0]

        daily_TMY_POA_sum.append(daily_TMY3_irradiance['POA'].sum())
        daily_CS_POA_sum.append(daily_clearsky_irradiance['POA'].sum())

        fig, (ax1) = plt.subplots(1, sharey=True)
        ax = daily_clearsky_irradiance['POA'].plot(
            ax=ax1, label='Clear Sky POA')

        daily_TMY3_irradiance['POA'].plot(ax=ax1, label='TMY3 POA')

        ax1.set_xlabel('Time of day (' + days[0].strftime("%m-%d") + ')')
        ax1.set_ylabel('Irradiance ($Wh/m^2$)')
        ax1.legend()
        title = name + ', ' + state + '. Tilt Angle:' + \
            str(round(latitude, 2)) + ' dec deg'
        ax1.title.set_text(title)
        plt.text(0, (round(daily_clearsky_irradiance["POA"].max())), "Clearsky POA: " + str(round(daily_clearsky_irradiance["POA"].sum(
        ))) + ' $Wh/m^2$', ha='left', va='top', size='medium', color='blue', weight='light')
        ax.text(0, (round(daily_clearsky_irradiance["POA"].max())), "TMY3 POA: " + str(round(daily_TMY3_irradiance["POA"].sum(
        ))) + ' $Wh/m^2$', ha='left', size='medium', color='orange', weight='light')

        print(round(daily_TMY3_irradiance["POA"].max()))
        # check if directory exists
        isExist = os.path.exists(file_path)
        if not isExist:
            os.makedirs(file_path)
        file_name = file_path + \
            days[0].strftime("%m-%d") + '_daily_poa_irradiance.png'
        plt.savefig(file_name, metadata=None, dpi=300, bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto')
        print("Saved " + file_name)
        plt.close()

        # Calculate daily fractional insolation
        total_daily_clearsky_POA = daily_clearsky_irradiance["POA"].sum()
        total_daily_TMY_POA = daily_TMY3_irradiance["POA"].sum()
        percentage_fractional_insolation = round(
            (total_daily_TMY_POA/total_daily_clearsky_POA) * 100, 2)
        fractional_insolation.append(percentage_fractional_insolation)

        annual_TMY_POA.append(total_daily_TMY_POA)
        annual_clearsky_POA.append(total_daily_clearsky_POA)
    fractional_insolation = pd.DataFrame(
        {'daily_fractional_insolation': fractional_insolation})
    scaler = MinMaxScaler(feature_range=(10,100))
    #df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    fractional_insolation = pd.DataFrame(scaler.fit_transform(fractional_insolation), columns=['daily_fractional_insolation'])
    
    plt.hist(fractional_insolation['daily_fractional_insolation'],
             bins=10,  edgecolor='black')  # format BIN
    plt.xlabel('Fractional Insolation (%)')
    plt.ylabel('Number of Days')
    plt.title('Fractional Insolation Distribution -' + name + ', ' + state)
    file_name = file_path + str(year) + '_fractional_insolation.png'
    plt.savefig(file_name, metadata=None, dpi=300, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto')
    print("Saved " + file_name)
    plt.close()
    annual_TMY_POA_sum = round(sum(annual_TMY_POA), 2)
    annual_clearsky_POA_sum = round(sum(annual_clearsky_POA), 2)
    annual_fractional_insolation = round(
        (annual_TMY_POA_sum/annual_clearsky_POA_sum) * 100, 2)

    total_fractional_insolation.append(annual_fractional_insolation)

    print("Total annual TMY POA: ", annual_TMY_POA_sum, "Wh/m^2")
    print("Total annual clearsky POA: ", annual_clearsky_POA_sum, "Wh/m^2")
    print("Full annual fractional insolation:",
          annual_fractional_insolation, "%")

    daily_CS_POA_sum_df = pd.DataFrame({'CS': daily_CS_POA_sum})
    daily_TMY_POA_sum_df = pd.DataFrame({'TMY': daily_TMY_POA_sum})
    fig, (ax1) = plt.subplots(1, sharey=True)
    ax = daily_CS_POA_sum_df.plot(ax=ax1, label='Clear Sky POA')

    ax1.plot("TMY", data=daily_TMY_POA_sum_df)
    ax1.set_xlabel('Day number')
    ax1.set_ylabel('Insolation ($Wh/m^2$)')
    ax1.legend()
    title = name + ', ' + state + '. Tilt Angle: ' + \
        str(round(latitude, 2)) + ' dec deg'
    ax1.title.set_text(title)
    file_name = file_path + str(year) + '_' + model + '_daily_insolation.png'
    plt.savefig(file_name, metadata=None, dpi=300, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto')
    print("Saved " + file_name)
    plt.close()

    # write fractional insolation to file
    print("Writing results to file...")
    fractional_insolation.to_csv(file_path + 'daily_fractional_insolation.csv')
    with open(file_path + "results.txt", "w") as outfile:
        outfile.write("Total annual TMY POA: " +
                      str(annual_TMY_POA_sum) + " Wh/m^2" + "\n")
        outfile.write("Total annual Clear Sky POA: " +
                      str(annual_clearsky_POA_sum) + " Wh/m^2" + "\n")
        outfile.write("Full annual fractional insolation: " +
                      str(annual_fractional_insolation) + "%" + "\n")
        outfile.close()
    

# save total % fractional insolation for all locations
data = {'state': state, name: names,
        'fractional_insolation': total_fractional_insolation}
data = pd.DataFrame(data)
data.to_csv(os.getcwd()+'xdata2.csv')

# Plot heat map and bar chart
# load the excel file into a pandas dataframe & skip header rows
df = pd.read_csv(os.getcwd()+'/xdata2.csv',
                  usecols=['state', 'name', 'fractional_insolation'])
df = df.sort_values(by=['state'], ascending=True)
gdf = gpd.read_file(os.getcwd()+'/files/cb_2018_us_state_500k')
gdf = gdf.merge(df, left_on='STUSPS', right_on='state')

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
polygon = Polygon([(-175, 50), (-175, 72), (-140, 72), (-140, 50)])

# clip alaska
poly_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=world.crs)
# fig, ax1 = plt.subplots(1, figsize=(8, 18))
polygon = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
gdf.to_crs({'init': 'epsg:2163'})

# Apply this to the gdf to ensure all states are assigned colors by the same func


def makeColorColumn(gdf, variable, vmin, vmax):
    # apply a function to a column to create a new column of assigned colors & return full frame
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlOrBr)
    gdf['value_determined_color'] = gdf[variable].apply(
        lambda x: mcolors.to_hex(mapper.to_rgba(x)))
    return gdf


# set the value column that will be visualised
variable = 'fractional_insolation'

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
# math.ceil(gdf.pct_food_insecure.max())
vmin, vmax = gdf.fractional_insolation.min(), gdf.fractional_insolation.max()
# Choose the continuous colorscale "YlOrBr" from https://matplotlib.org/stable/tutorials/colors/colormaps.html
colormap = "YlOrBr"
gdf = makeColorColumn(gdf, variable, vmin, vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init': 'epsg:2163'})


# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

# set the font for the visualization to Helvetica
hfont = {'fontname': 'Helvetica'}

# add a title and annotation
ax.set_title('Annual Fractional Insolation\n Distribution by Location.',
              **hfont, fontdict={'fontsize': '42', 'fontweight': '1'})

# Create colorbar legend
fig = ax.get_figure()
# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])

cbax.set_title('Annual Percentage fractional\n insolation for U.S. States.',
                **hfont, fontdict={'fontsize': '15', 'fontweight': '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap,
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
comma_fmt = FuncFormatter(lambda x, p: format(x/100, '.0%'))
fig.colorbar(sm, cax=cbax, format=comma_fmt)
tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)
# annotate the data source, date of access, and hyperlink
ax.annotate("Data: CAMS McClear Service for irradiation under clear-sky [1]\n Photovoltaic Geographical Information System (PVGIS) [2]", xy=(
    0.22, .085), xycoords='figure fraction', fontsize=14, color='#555555')


# create map
# Note: we're going state by state here because of unusual coloring behavior when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state not in ['AK', 'HI']:
        vf = visframe[visframe.state == row.state]
        c = gdf[gdf.state == row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')


# add Alaska
akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])
akax.axis('off')
# polygon to clip western islands
polygon = Polygon([(-170, 50), (-170, 72), (-140, 72), (-140, 50)])
alaska_gdf = gdf[gdf.state == 'AK']
alaska_gdf.clip(polygon).plot(
    color=gdf[gdf.state == 'AK'].value_determined_color, linewidth=0.8, ax=akax, edgecolor='0.8')


# add Hawaii
hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])
hiax.axis('off')
# polygon to clip western islands
hipolygon = Polygon([(-160, 0), (-160, 90), (-120, 90), (-120, 0)])
hawaii_gdf = gdf[gdf.state == 'HI']
hawaii_gdf.clip(hipolygon).plot(column=variable,
                                color=hawaii_gdf['value_determined_color'], linewidth=0.8, ax=hiax, edgecolor='0.8')


fig.savefig(os.getcwd()+'/files/tmy_clearsky_analysis/percentage_fractional_insolation_distribution.png',
            dpi=400, bbox_inches="tight")

# Make a horizontal bar chart of state food insecurity rates using Seaborn
visFrame = gdf.sort_values('fractional_insolation', ascending=False)[:100]

# Horizontal barchart
# Create subplot
sns.set_style('whitegrid')  # set theme
fig, ax = plt.subplots(figsize=(16, 30))
#ax.update_yaxes({"tick_font_size": 2})
# Create barplot
chart1 = sns.barplot(
    x=visFrame['fractional_insolation'], y=visFrame['name'], color='lightblue')
chart1.set_title('Percentage Fractional Insolation by Location.\n',
                  weight='bold', fontsize=20)

chart1.set_xlabel('% Factional Insolation', weight='bold', fontsize=20)
chart1.set_ylabel('Location', weight='bold', fontsize=20)

for p in ax.patches:
    width = p.get_width()    # get bar length
    ax.text(width + .9,       # set the text at .5 unit right of the bar
            p.get_y() + p.get_height() / 2,  # get Y coordinate + X coordinate / 2
            '{:}%'.format(round(width, 1)),  # set variable to display
            fontsize=25,
            ha='left',   # horizontal alignment
            va='center')  # vertical alignment

fig.savefig(os.getcwd()+'/files/tmy_clearsky_analysis/percentage_fractional_insolation.png',
            dpi=400, bbox_inches="tight")
