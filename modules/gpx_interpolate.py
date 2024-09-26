# Copyright (c) 2019 Remi Salmon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# imports
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, splprep, splev

# constants
EARTH_RADIUS = 6371e3 # meters

# functions
def gpx_interpolate(gpx_data, res, deg = 1, use_ele=False):
    # input: gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    #        res = float
    #        deg = int
    # output: gpx_data_interp = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}

    if not type(deg) is int:
        raise TypeError('deg must be int')

    if not 1 <= deg <= 5:
        raise ValueError('deg must be in [1-5]')

    if not len(gpx_data['lat']) > deg:
        raise ValueError('number of data points must be > deg')

    # interpolate spatial data
    _gpx_data = gpx_remove_duplicate(gpx_data)

    _gpx_dist = gpx_calculate_distance(_gpx_data, use_ele = use_ele)

    x = [_gpx_data[i] for i in ('lat', 'lon', 'ele') if (i in _gpx_data)]

    tck, _ = splprep(x, u = np.cumsum(_gpx_dist), k = deg, s = 0)

    u_interp = np.linspace(0, np.sum(_gpx_dist), num = 1+int(np.sum(_gpx_dist)/res))
    x_interp = splev(u_interp, tck)

    # interpolate time data linearly to preserve monotonicity
    if ('tstamp' in _gpx_data):
        f = interp1d(np.cumsum(_gpx_dist), _gpx_data['tstamp'], fill_value = 'extrapolate')

        tstamp_interp = f(u_interp)

    gpx_data_interp = {'lat':list(x_interp[0]),
                       'lon':list(x_interp[1]),
                       'ele':list(x_interp[2]) if ('ele' in gpx_data) else None,
                       'tstamp':list(tstamp_interp) if ('tstamp' in gpx_data) else None}

    return gpx_data_interp

def interpolate_points_latlon(df, deg = 2, res = 10, add_vars=[]):
    """
    @param res: spatial resolution in metres
    @param deg: degree of spline (if 1, then linear) used for interpolation
    """
    
    gpx_data = {'lat':df['Latitude'].to_numpy(),
                'lon':df['Longitude'].to_numpy(),
                'tstamp': df['Timestamp'].to_numpy()}
    names_dict = {0:'Latitude', 1:'Longitude'}
    names_tuple = tuple(['lat', 'lon'] + add_vars)
    for i, v in enumerate(add_vars, 2):
        gpx_data.update({v:df[v].to_numpy()})
        names_dict.update({i:v})
    
    gpx_dist = gpx_calculate_distance(gpx_data, use_ele=False)
    x = [gpx_data[i] for i in names_tuple if (i in gpx_data)]
    tck, _ = splprep(x, u = np.cumsum(gpx_dist), k = deg, s = 0)
    u_interp = np.linspace(0, np.sum(gpx_dist), num = 1+int(np.sum(gpx_dist)/res))
    x_interp = splev(u_interp, tck)
    latlon_interp = pd.DataFrame(x_interp).T.rename(names_dict, axis = 1)
    
    return latlon_interp


def gpx_calculate_distance(gpx_data, use_ele = True):
    # input: gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    #        use_ele = bool
    # output: gpx_dist = numpy.ndarray[float]

    gpx_dist = np.zeros(len(gpx_data['lat']))

    for i in range(len(gpx_dist)-1):
        lat1 = np.radians(gpx_data['lat'][i])
        lon1 = np.radians(gpx_data['lon'][i])
        lat2 = np.radians(gpx_data['lat'][i+1])
        lon2 = np.radians(gpx_data['lon'][i+1])

        delta_lat = lat2-lat1
        delta_lon = lon2-lon1

        c = 2.0*np.arcsin(np.sqrt(np.sin(delta_lat/2.0)**2+np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2.0)**2)) # haversine formula

        dist_latlon = EARTH_RADIUS*c # great-circle distance

        if ('ele' in gpx_data) and use_ele:
            dist_ele = gpx_data['ele'][i+1]-gpx_data['ele'][i]

            gpx_dist[i+1] = np.sqrt(dist_latlon**2+dist_ele**2)
        else:
            gpx_dist[i+1] = dist_latlon

    return gpx_dist


def gpx_remove_duplicate(gpx_data):
    # input: gpx_data = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}
    # output: gpx_data_nodup = dict{'lat':list[float], 'lon':list[float], 'ele':list[float], 'tstamp':list[float], 'tzinfo':datetime.tzinfo}

    gpx_dist = gpx_calculate_distance(gpx_data)

    i_dist = np.concatenate(([0], np.nonzero(gpx_dist)[0])) # keep gpx_dist[0] = 0.0

    if not len(gpx_dist) == len(i_dist):
        print('Removed {} duplicate trackpoint(s)'.format(len(gpx_dist)-len(i_dist)))

    gpx_data_nodup = {'lat':[], 'lon':[], 'ele':[], 'tstamp':[]}

    for k in ('lat', 'lon', 'ele', 'tstamp'):
        gpx_data_nodup[k] = [gpx_data[k][i] for i in i_dist] if (k in gpx_data) else None

    return gpx_data_nodup

