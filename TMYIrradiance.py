import pvlib
import datetime
from pvlib.location import Location
from pvlib import irradiance
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.iotools import get_pvgis_tmy
from pvlib.pvsystem import PVSystem, Array, FixedMount


class TMYIrradiance:
    def __init__(self, latitude, longitude, altitude, timezone, name, surface_tilt, surface_azimuth):
        self.location = Location(
            latitude, longitude, tz=timezone, altitude=altitude, name=name)
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.mount = mount = FixedMount(
            surface_tilt=surface_tilt, surface_azimuth=surface_azimuth)

    def get_irradiance_data(self, year, frequency, period):
        url = r'https://re.jrc.ec.europa.eu/api/v5_2/'
        pvgis_data = get_pvgis_tmy(self.location.latitude, self.location.longitude,
                                   outputformat='csv', url=url, map_variables=True)

        weather_data = pvgis_data[0]
        weather_data.index = weather_data.index.map(
            lambda t: t.replace(year=1990)
        )

        # convert to local time from tz
        weather_data = weather_data.tz_convert(self.location.tz)

        weather_data.index = weather_data.index.map(
            lambda t: t.replace(year=1990)
        )

        weather_data = weather_data.sort_index()

        return weather_data

    def get_annual_tmy_irradiance(self, year, frequency, period):

        annual_tmy_irradiance = []
        tmy_data = self.get_irradiance_data(year, frequency, period)
        # tmy_daily_irradiance = tmy_data.groupby(['Month', 'Day'])
        tmy_daily_irradiance = tmy_data.groupby(
            [tmy_data.index.month, tmy_data.index.day])
        for (month, day), group in tmy_daily_irradiance:
            date = group.index[0]
            times = pd.date_range(date, freq=frequency,
                                  periods=period, tz=self.location.tz)
            # group.index = times
            solar_position = self.location.get_solarposition(times=times)

            tmy_POA_irradiance = irradiance.get_total_irradiance(
                surface_tilt=self.location.latitude,
                surface_azimuth=self.surface_azimuth,

                solar_zenith=solar_position['apparent_zenith'],
                solar_azimuth=solar_position['azimuth'],
                dni=group['dni'],
                ghi=group['ghi'],
                dhi=group['dhi']
            )

            annual_tmy_irradiance.append(pd.DataFrame(
                {'POA': tmy_POA_irradiance['poa_global'], 'GHI': group['ghi']}))

        return annual_tmy_irradiance
