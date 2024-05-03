import pvlib
import datetime
from pvlib.location import Location
from pvlib import irradiance
import pandas as pd
import matplotlib.pyplot as plt


class ClearSkyIrradiance:
    def __init__(self, latitude, longitude, altitude, timezone, name, surface_tilt, surface_azimuth):
        self.location = Location(
            latitude, longitude, tz=timezone, altitude=altitude, name=name)
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth

    def get_annual_clear_sky_irradiance(self, year, frequency, period):
        print('Getting clear sky irradiance...')
        email = 'idowusteve.idowu@gmail.com'
        self.year = int(year)
        annual_clear_sky_irradiance = []

        print('Getting the irradiance for each day in ' + str(year) + '...')
        # model = 'simplified_solis'
        start_date = datetime.date(self.year, 1, 1)

        times = pd.date_range(start=start_date, freq=frequency,
                              periods=8760, tz=self.location.tz)
        print('Getting Solar Data for day # ' + str(times[0]) + '...')
        clear_sky_data = pvlib.iotools.get_cams(latitude=self.location.latitude, longitude=self.location.longitude,
                                                start=times[0], end=times[-1], email=email, identifier='mcclear', time_step=frequency)[0]
        clear_sky_data = clear_sky_data.tz_convert(self.location.tz)

        clear_sky_data.index = clear_sky_data.index.map(
            lambda t: t.replace(year=self.year)
        )
        clear_sky_data = clear_sky_data.sort_index()

        clear_sky_daily_irradiance = clear_sky_data.groupby(
            [clear_sky_data.index.month, clear_sky_data.index.day])

        for (month, day), group in clear_sky_daily_irradiance:
            date = group.index[0]
            times = pd.date_range(date, freq=frequency,
                                  periods=period, tz=self.location.tz)
            solar_position = self.location.get_solarposition(times=times)

            clear_sky_POA_irradiance = irradiance.get_total_irradiance(
                surface_tilt=self.location.latitude,
                surface_azimuth=self.surface_azimuth,

                solar_zenith=solar_position['apparent_zenith'],
                solar_azimuth=solar_position['azimuth'],
                dni=group['dni_clear'],
                ghi=group['ghi_clear'],
                dhi=group['dhi_clear']
            )

            annual_clear_sky_irradiance.append(pd.DataFrame(
                {'POA': clear_sky_POA_irradiance['poa_global'], 'GHI': group['ghi_clear']}))

        return annual_clear_sky_irradiance
