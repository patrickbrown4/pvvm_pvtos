"""
MIT License

Copyright (c) 2019 Patrick Richard Brown

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
"""

import pandas as pd
import numpy as np
import sys, os, site, zipfile, math, time, json
import googlemaps, urllib, shapely, urllib.request
import xml.etree.ElementTree as ET
from glob import glob
from urllib.error import HTTPError
from tqdm import tqdm, trange
from warnings import warn

#################
### VARIABLES ###
#################

### ISO timezone
timezone_iso = {
    'CAISO': -8, 'caiso': -8, 'c': -8, 'C': -8,
    'ERCOT': -6, 'ercot': -6, 'e': -6, 'E': -6,
    'MISO':  -5, 'miso':  -5, 'm': -5, 'M': -5,
    'PJM':   -5, 'pjm':   -5, 'p': -5, 'P': -5,
    'NYISO': -5, 'nyiso': -5, 'n': -5, 'N': -5,
    'ISONE': -5, 'isone': -5, 'i': -5, 'I': -5,
    'APS': -7, 'AZ': -7, 'NV': -8,
    'PACE': -7, 'ID': -7, 'UT': -7, 'WY': -7,
    'PACW': -8, 'PGE': -8, 'PSE': -8,
    'PACW_PGE_PSE': -8,
    'BPA': -8,
    'SMEPA': -5, 'Entergy': -5, 'CLECO': -5,
    'LAGN': -5, 'LEPA': -5,
}

def timezone_to_tz(timezone):
    return 'Etc/GMT{:+}'.format(-timezone)

tz_iso = {key: timezone_to_tz(timezone_iso[key]) for key in timezone_iso}

### Daylight savings time
### Ref: www.timeanddate.com/time/change/usa/new-york
### Ref: en.wikipedia.org/wiki/History_of_time_in_the_United_States
dst_springforward = {
    1990: '19900401',
    1991: '19910407',
    1992: '19920405',
    1993: '19930404',
    1994: '19940403',
    1995: '19950402',
    1996: '19960407',
    1997: '19970406',
    1998: '19980405',
    1999: '19990404',
    2000: '20000402',
    2001: '20010401',
    2002: '20020407',
    2003: '20030406',
    2004: '20040404',
    2005: '20050403',
    2006: '20060402',
    2007: '20070311',
    2008: '20080309',
    2009: '20090308',
    2010: '20100314',
    2011: '20110313',
    2012: '20120311',
    2013: '20130310',
    2014: '20140309',
    2015: '20150308',
    2016: '20160313',
    2017: '20170312',
    2018: '20180311',
    2019: '20190310',
    2020: '20200308'
}

dst_fallback = {
    1990: '19901028',
    1991: '19911027',
    1992: '19921025',
    1993: '19931031',
    1994: '19941030',
    1995: '19951029',
    1996: '19961027',
    1997: '19971026',
    1998: '19981025',
    1999: '19991031',
    2000: '20001029',
    2001: '20011028',
    2002: '20021027',
    2003: '20031026',
    2004: '20041031',
    2005: '20051030',
    2006: '20061029',
    2007: '20071104',
    2008: '20081102',
    2009: '20091101',
    2010: '20101107',
    2011: '20111106',
    2012: '20121104',
    2013: '20131103',
    2014: '20141102',
    2015: '20151101',
    2016: '20161106',
    2017: '20171105',
    2018: '20181104',
    2019: '20191103',
    2020: '20201101'
}

states = [
    'Alabama',
    'Alaska',
    'Arizona',
    'Arkansas',
    'California',
    'Colorado',
    'Connecticut',
    'Delaware',
    'Florida',
    'Georgia',
    'Hawaii',
    'Idaho',
    'Illinois',
    'Indiana',
    'Iowa',
    'Kansas',
    'Kentucky',
    'Louisiana',
    'Maine',
    'Maryland',
    'Massachusetts',
    'Michigan',
    'Minnesota',
    'Mississippi',
    'Missouri',
    'Montana',
    'Nebraska',
    'Nevada',
    'New Hampshire',
    'New Jersey',
    'New Mexico',
    'New York',
    'North Carolina',
    'North Dakota',
    'Ohio',
    'Oklahoma',
    'Oregon',
    'Pennsylvania',
    'Rhode Island',
    'South Carolina',
    'South Dakota',
    'Tennessee',
    'Texas',
    'Utah',
    'Vermont',
    'Virginia',
    'Washington',
    'West Virginia',
    'Wisconsin',
    'Wyoming',
    'District of Columbia',
]

usps = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC',
]

state2usps = dict(zip(states, usps))
usps2state = dict(zip(usps, states))


### FIPS codes
### https://www.census.gov/geographies/reference-files/2017/demo/popest/2017-fips.html

fips2state = {
    2: 'AK', 1: 'AL', 5: 'AR', 4: 'AZ', 6: 'CA', 8: 'CO', 9: 'CT', 
    11: 'DC', 10: 'DE', 12: 'FL', 13: 'GA', 15: 'HI', 19: 'IA', 16: 'ID', 
    17: 'IL', 18: 'IN', 20: 'KS', 21: 'KY', 22: 'LA', 25: 'MA', 24: 'MD', 
    23: 'ME', 26: 'MI', 27: 'MN', 29: 'MO', 28: 'MS', 30: 'MT', 37: 'NC', 
    38: 'ND', 31: 'NE', 33: 'NH', 34: 'NJ', 35: 'NM', 32: 'NV', 36: 'NY', 
    39: 'OH', 40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 
    47: 'TN', 48: 'TX', 49: 'UT', 51: 'VA', 50: 'VT', 53: 'WA', 55: 'WI', 
    54: 'WV', 56: 'WY', 
 }

state2fips = {
    'AK': 2, 'AL': 1, 'AR': 5, 'AZ': 4, 'CA': 6, 'CO': 8, 'CT': 9, 
    'DC': 11, 'DE': 10, 'FL': 12, 'GA': 13, 'HI': 15, 'IA': 19, 'ID': 16, 
    'IL': 17, 'IN': 18, 'KS': 20, 'KY': 21, 'LA': 22, 'MA': 25, 'MD': 24, 
    'ME': 23, 'MI': 26, 'MN': 27, 'MO': 29, 'MS': 28, 'MT': 30, 'NC': 37, 
    'ND': 38, 'NE': 31, 'NH': 33, 'NJ': 34, 'NM': 35, 'NV': 32, 'NY': 36, 
    'OH': 39, 'OK': 40, 'OR': 41, 'PA': 42, 'RI': 44, 'SC': 45, 'SD': 46, 
    'TN': 47, 'TX': 48, 'UT': 49, 'VA': 51, 'VT': 50, 'WA': 53, 'WI': 55, 
    'WV': 54, 'WY': 56,
}


###############
### GENERAL ###
###############

def yearhours(year):
    if year == 'tmy':
        hours = 365*24
    elif year % 4 != 0:
        hours = 365*24
    elif year % 100 != 0:
        hours = 366*24
    elif year % 400 != 0:
        hours = 365*24
    else:
        hours = 366*24
    return hours

def monthhours(year, month):
    """
    Inputs
    ------
    year: integer
    month: integer in range(1,13)
    """
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31 * 24
    elif month in [4, 6, 9, 11]:
        return 30 * 24
    elif month == 2 and yearhours(year) == 8760:
        return 28 * 24
    elif month == 2 and yearhours(year) == 8784:
        return 29 * 24
    else:
        raise Exception("Bad query. Use integer year and month.")

def makedays(year, style='yyyymmdd'):
    """
    Returns a full list of calendar dates for the specified year.
    Allowed styles: ['yyyymmdd', 'yyyy-mm-dd', 'dd/mm/yyyy']
    Default style is 'yyyymmdd'.
    """
    if style not in ['yyyymmdd', 'yyyy-mm-dd', 'dd/mm/yyyy']:
        raise Exception('Invalid style. Check help.')
    yr = str(year)
    days = int(yearhours(year)/24)
    monthdays = {
        '01': 31, 
        '02': 28 if days == 365 else 29,
        '03': 31, '04': 30, '05': 31, '06': 30, '07': 31, 
        '08': 31, '09': 30, '10': 31, '11': 30, '12': 31}
    daysout = []
    for month in range(12):
        mon = '{:02d}'.format(month + 1)
        for day in range(monthdays[mon]):
            d = '{:02d}'.format(day + 1)
            if style == 'yyyymmdd':
                daysout.append(yr + mon + d)
            elif style == 'yyyy-mm-dd':
                daysout.append(yr + '-' + mon + '-' + d)
            elif style == 'mm/dd/yyyy':
                daysout.append(mon + '/' + d + '/' + yr)
    return daysout

def pathify(path=None, add='', make=False):
    if path is None:
        path = ''
    path, add = str(path), str(add)
    if len(path) != 0 and path[-1] != '/':
        path = path + '/'
    
    path = path + add
    if path != '':
        if path[-1] != '/': 
            path = path + '/'
    
    if make==True:
        if os.path.isdir(path):
            pass
        else:
            os.mkdir(path)

    return path

def nowtime():
    now = time.localtime()
    out = (str(now.tm_year) +
        '{:02d}'.format(now.tm_mon) + 
        '{:02d}'.format(now.tm_mday) + 
        ' ' +
        '{:02d}'.format(now.tm_hour) + 
        ':' +
        '{:02d}'.format(now.tm_min) + 
        ':' + 
        '{:02d}'.format(now.tm_sec))
    return out

def undodst(datetime, keepfallbackhour=True):
    if type(datetime) != pd.Timestamp:
        raise TypeError('datetime must be pd.Timestamp')
    year = datetime.year
    dststart = dst_springforward[year] + ' 03:00'
    dststart = pd.to_datetime(dststart)
    if keepfallbackhour:
        dstend = dst_fallback[year] + ' 01:00'
    else:
        dstend = dst_fallback[year] + ' 02:00'
    dstend = pd.to_datetime(dstend)

    if (datetime >= dststart) and (datetime <= dstend):
        datetimeout = datetime - pd.Timedelta('1H')
    else:
        datetimeout = datetime

    return datetimeout

def testsave(savename):
    if os.path.isfile(savename):
        print("WARNING: A file already exists for name '{}'.".format(savename))
        gonogo = input("Do you want to overwrite it? y/[n]")
        if gonogo.lower() == 'y':
            return savename
        else:
            newname = input("Do you want to create a new savename? [y]/n ")
            if newname.lower() == 'n':
                quit()
            else:
                savename = input("Enter new filename (include folder and extension, no quotes): ")
                return testsave(savename)
    else:
        return savename

def safesave(savename):
    goodstuff = len(os.path.splitext(savename)[0])
    if os.path.isfile(savename):
        if savename[goodstuff - 5] == '(' and savename[goodstuff - 1] == ')':
            oldfiller = int(savename[goodstuff - 4:goodstuff - 1])
            newfiller = oldfiller + 1
            newname = (savename[:goodstuff - 4]
                       + '{:03d}'.format(newfiller) 
                       + savename[goodstuff - 1:])
        else:
            newname = (savename[:goodstuff] + ' (001)' + savename[goodstuff:])
        return safesave(newname)
    else: 
        return savename

##################
### GEOGRAPHIC ###
##################

def get_latlonlabels(df, latlonlabels=None, columns=None):
    if latlonlabels is not None:
        latlabel, lonlabel = latlonlabels[0], latlonlabels[1]
    if columns is None:
        columns = df.columns
        
    if ('latitude' in columns) and ('longitude' in columns):
        latlabel, lonlabel = 'latitude', 'longitude'
    elif ('Latitude' in columns) and ('Longitude' in columns):
        latlabel, lonlabel = 'Latitude', 'Longitude'
    elif ('lat' in columns) and ('lon' in columns):
        latlabel, lonlabel = 'lat', 'lon'
    elif ('lat' in columns) and ('long' in columns):
        latlabel, lonlabel = 'lat', 'long'
    elif ('x' in columns) and ('y' in columns):
        latlabel, lonlabel = 'x', 'y'
    
    return latlabel, lonlabel

def closestpoint_simple(
    querylons, querylats, reflons, reflats,
    verbose=False):
    """
    For each (lon,lat) in zip(querylons,querylats), find 
    index of closest point in zip(reflons,reflats) using
    simple lat/lon cartesian coordinates. So only gives
    relatively accurate outputs for small distances.
    """
    ### Make sure lons and lats are same length
    assert len(querylons) == len(querylats)
    assert len(reflons) == len(reflats)
    
    ### Make lookup table
    dfref = pd.DataFrame({'reflon':reflons,'reflat':reflats})
    
    ### Loop over query latlons
    if verbose is True:
        iterator = trange(len(querylons))
    elif verbose is False:
        iterator= range(len(querylons))
    out = []
    for i in iterator:
        ### Get query point (could instead do this with 
        ### zip and enumerate)
        lon, lat = querylons[i], querylats[i]
        ### Calculate squared distance to each ref point
        sqdistances = (reflons-lon)**2 + (reflats-lat)**2
        ### Save the index
        out.append(np.argmin(sqdistances))
    return out

def closestpoint(
    dfquery, dfdata, dfquerylabel=None, 
    dfqueryx='longitude', dfqueryy='latitude', 
    dfdatax='longitude', dfdatay='latitude',
    method='cartesian', return_distance=False):
    """
    For each row of dfquery, returns closest label from dfdata
    """
    import geopy.distance
    closestindexes = []
    closest_distances = []
    if method in ['cartesian', 'xy', None, 'equirectangular', 'latlon']:
        lons, lats = dfdata[dfdatax].values, dfdata[dfdatay].values
        for i in dfquery.index:
            lon, lat = dfquery.loc[i, dfqueryx], dfquery.loc[i, dfqueryy]
            sqdistances = (lons - lon)**2 + (lats - lat)**2
            closestindex = np.argmin(sqdistances)
            closestindexes.append(closestindex)
            
            if return_distance is True:
                closest_distances.append(
                    geopy.distance.distance(
                        (lat, lon), 
                        (dfdata[dfdatay].values[closestindex], 
                         dfdata[dfdatax].values[closestindex])).km
                )
            
    elif method in ['geopy', 'geodesic']:
        lons, lats = dfdata[dfdatax].values, dfdata[dfdatay].values
        for i in tqdm(dfquery.index):
            lon, lat = dfquery.loc[i, dfqueryx], dfquery.loc[i, dfqueryy]
            distances = dfdata.apply(
                lambda row: geopy.distance.distance((lat, lon), (row[dfdatay], row[dfdatax])).km,
                axis=1).values
            closestindex = np.argmin(distances)
            closestindexes.append(closestindex)
            
            if return_distance is True:
                closest_distances.append(
                    geopy.distance.distance(
                        (lat, lon), 
                        (dfdata[dfdatay].values[closestindex], 
                         dfdata[dfdatax].values[closestindex])).km
                )
            
    if return_distance is True:
        return closestindexes, closest_distances
    else:
        return closestindexes

def pointpolymap(
    dfpointsin, dfpolyin, 
    x='Longitude', y='Latitude', zone='Zone',
    resetindex=True, verbose=True, progressbar=True,
    multipleassignment='raise'):
    """
    Inputs
    ------
    dfpointsin: dataframe of points with separate x and y columns
    dfpolyin: geopandas dataframe with polygons in 'geometry' column
    x: label for x column. default 'Longitude'
    y: label for y column. default 'Latitude'
    zone: label for polygon column. default 'Zone'
    multipleassignment: str in ['debug', 'raise', 'pass', 'flatlists']
    """
    if resetindex:
        dfpoints = dfpointsin.reset_index(drop=True)
        dfpoly = dfpolyin.reset_index(drop=True)
    else:
        ### changed 20190620
        # dfpoints, dfpoly = dfpointsin.copy(), dfpolyin.copy()
        dfpoints, dfpoly = dfpointsin.copy(), dfpolyin
        
    point_lonlats = list(zip(dfpoints[x].values, dfpoints[y].values))
    ###### added 20190620
    if type(dfpoly) == shapely.geometry.polygon.Polygon:
        namelist = [True]
        geoms = [dfpoly]
        ###############################
    else:
        namelist = dfpoly[zone].copy()
        geoms = dfpoly.geometry.copy()

    polybools = []
    polynames = []
    ### Loop over points
    if progressbar is True:
        iterator = trange(len(point_lonlats))
    else:
        iterator = range(len(point_lonlats))
    for i in iterator:
        point = shapely.geometry.Point(point_lonlats[i])
        polyboolsout = []
        polynamesout = []
        ### Loop over polys
        for j in range(len(geoms)):
            if point.within(geoms[j]):
                polyboolsout.append(True)
                polynamesout.append(namelist[j])
        polybools.append(polyboolsout)
        polynames.append(polynamesout)
    
    ### Check for overlaps
    foo = []
    for i in range(len(polybools)):
        foo.append(len(polybools[i]))
        if len(polybools[i]) > 1:
            if verbose:
                print(i)
                print(polybools[i])
                print(polynames[i])
    
    ### Write output list
    polyboolsflat = []
    polynamesflat = []
    for i in range(len(polybools)): 
        try:
            polyboolsflat.append(polybools[i][0])
            polynamesflat.append(polynames[i][0])
        except IndexError:
            polyboolsflat.append(False)
            polynamesflat.append('')
    
    ### Deal with multiply-assigned points if necessary
    if pd.Series(foo).max() > 1:
        print('max overlap = {}'.format(pd.Series(foo).max()))
        if multipleassignment=='flatlists':
            print('returning (polynamesflat, polyboolsflat)')
            return polynamesflat, polyboolsflat
        elif multipleassignment in ['debug', 'originallist']:
            print('returning polynames')
            return polynames
        elif multipleassignment in ['raise', 'except']:
            raise Exception('Some points are mulitply assigned')
        elif multipleassignment in ['silent', 'pass']:
            print('returning polynamesflat')
            return polynamesflat
    
    return polynamesflat


def voronoi_polygons(dfpoints):
    """
    Inputs
    ------
    dfpoints: pd.DataFrame with latitude and longitude columns
    
    Ouputs
    ------
    dfpoly: dataframe with Voronoi polygons and descriptive parameters
    
    Sources
    -------
    ('https://stackoverflow.com/questions/27548363/'
     'from-voronoi-tessellation-to-shapely-polygons)
    
    """
    import shapely, scipy.spatial, pyproj
    import geopandas as gpd
    
    ### Get latitude and longitude column names
    latlabel, lonlabel = get_latlonlabels(dfpoints)
    
    ### Get polygons
    points = dfpoints[[lonlabel,latlabel]].values
    vor = scipy.spatial.Voronoi(points)

    ### Make shapely linestrings 
    lines = [
        shapely.geometry.LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]

    ### Make shapely polygons, coords, and bounds
    ###### CHANGED 20190911 - Intersect each poly with region bounds
    ### Original
    # polys = [poly for poly in shapely.ops.polygonize(lines)]
    ### New
    regionhull = (
        shapely.geometry.MultiPoint(dfpoints[[lonlabel,latlabel]].values)
        .convex_hull)
    polys = [poly.intersection(regionhull) for poly in shapely.ops.polygonize(lines)]
    ###### Continue
    coords = [list(poly.exterior.coords) for poly in polys]
    bounds = [list(poly.bounds) for poly in polys] ### (minx, miny, maxx, maxy)
    centroid_x = [poly.centroid.x for poly in polys]
    centroid_y = [poly.centroid.y for poly in polys]
    centroids = [[poly.centroid.x, poly.centroid.y] for poly in polys]

    ### Calculate areas in square kilometers
    areas = []
    for i, (poly, coord, bound) in enumerate(list(zip(polys, coords, bounds))):
        pa = pyproj.Proj("+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(
            bound[1], bound[3], (bound[1]+bound[3])/2, (bound[0]+bound[2])/2))
        lon,lat = zip(*coord)
        x,y = pa(lon,lat)
        cop = {'type':'Polygon','coordinates':[zip(x,y)]}
        areas.append(shapely.geometry.shape(cop).area/1000000)

    ### Make and return output dataframe
    dfpoly = gpd.GeoDataFrame(pd.DataFrame(
        {'coords':coords, 'bounds':bounds,
         'centroid':centroids,'centroid_lon':centroid_x,'centroid_lat':centroid_y,
         'area':areas, 'geometry':polys,}))
    
    return dfpoly


def get_area_latlon(shape):
    """
    Notes
    -----
    * shape must be a shapely Polygon or MultiPolygon
    * results are return in km^2
    """
    import shapely, pyproj
    ### Easy way, if simple polygon
    if type(shape) == shapely.geometry.polygon.Polygon:
        ### Get coords, bounds, and centroid
        coord = shape.exterior.coords
        bound = shape.bounds
        ### Project into equal-area coordinates
        pa = pyproj.Proj("+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(
            bound[1], bound[3], (bound[1]+bound[3])/2, (bound[0]+bound[2])/2))
        lon,lat = zip(*coord)
        x,y = pa(lon,lat)
        ### Make new polygon in equal-area coordinates
        cop = {'type':'Polygon','coordinates':[zip(x,y)]}
        ### Calculate the area [km^2]
        area = shapely.geometry.shape(cop).area/1000000
    ### If multipolygon
    elif type(shape) == shapely.geometry.multipolygon.MultiPolygon:
        subareas = []
        for subpoly in shape:
            ### Get coords, bounds, and centroid
            coord = subpoly.exterior.coords
            bound = subpoly.bounds
            ### Project into equal-area coordinates
            pa = pyproj.Proj("+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(
                bound[1], bound[3], (bound[1]+bound[3])/2, (bound[0]+bound[2])/2))
            lon,lat = zip(*coord)
            x,y = pa(lon,lat)
            ### Make new polygon in equal-area coordinates
            cop = {'type':'Polygon','coordinates':[zip(x,y)]}
            ### Calculate the area
            subarea = shapely.geometry.shape(cop).area/1000000
            subareas.append(subarea)
        ### Sum the area
        area = sum(subareas)
    return area


def polyoverlaparea_km2(poly1, poly2=None, method='intersection', returnpoly=False):
    """
    """
    import shapely, pyproj
    ### Get output poly
    if (method in ['first']) or (poly2 is None):
        polyout = poly1
    elif method in ['difference']:
        polyout = poly1.difference(poly2)
    elif method in ['intersection']:
        polyout = poly1.intersection(poly2)
        
    ### If no overlap
    if polyout.is_empty:
        area = 0

    ### Easy way, if polyout is simple polygon
    elif type(polyout) == shapely.geometry.polygon.Polygon:
        ### Get coords, bounds, and centroid
        coord = polyout.exterior.coords
        bound = polyout.bounds
        ### Project into equal-area coordinates
        pa = pyproj.Proj("+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(
            bound[1], bound[3], (bound[1]+bound[3])/2, (bound[0]+bound[2])/2))
        lon,lat = zip(*coord)
        x,y = pa(lon,lat)
        ### Make new polygon in equal-area coordinates
        cop = {'type':'Polygon','coordinates':[zip(x,y)]}
        ### Calculate the area
        area = shapely.geometry.shape(cop).area/1000000

    ### If multipolygon
    elif type(polyout) in [shapely.geometry.multipolygon.MultiPolygon, 
                           shapely.geometry.collection.GeometryCollection]:
        subareas = []
        for subpoly in polyout:
            ### Pass over if item is a point
            if type(subpoly) == shapely.geometry.point.Point:
                continue
            ### Get coords, bounds, and centroid
            coord = subpoly.exterior.coords
            bound = subpoly.bounds
            ### Project into equal-area coordinates
            pa = pyproj.Proj("+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(
                bound[1], bound[3], (bound[1]+bound[3])/2, (bound[0]+bound[2])/2))
            lon,lat = zip(*coord)
            x,y = pa(lon,lat)
            ### Make new polygon in equal-area coordinates
            cop = {'type':'Polygon','coordinates':[zip(x,y)]}
            ### Calculate the area
            subarea = shapely.geometry.shape(cop).area/1000000
            subareas.append(subarea)
        ### Sum the area
        area = sum(subareas)
    
        
    ### Return results
    if returnpoly is False:
        return area
    else:
        return area, polyout


def voronoi_polygon_overlap(
        dfcoords, polyoverlap, index_coords=None, polybuffer=0.5,
        returnfull=False):
    """
    Inputs
    ------
    index_coords: column of dfcoords to use as label. If None, use dfcoords.index.

    Notes
    -----
    * dfcoords should extendpast the boundaries of polyoverlap

    """
    import shapely

    ### Get latitude and longitude column names
    latlabel, lonlabel = get_latlonlabels(dfcoords)
    ### Get index label if necessary
    if index_coords is None:
        indexlabel = 'index'
    else:
        indexlabel = index_coords

    ###### Get bounding box for region, add buffer
    regionbounds = {
        'longitude':[polyoverlap.bounds[0]-polybuffer, polyoverlap.bounds[2]+polybuffer],
        'latitude':[polyoverlap.bounds[1]-polybuffer, polyoverlap.bounds[3]+polybuffer],
    }

    ### Get region bounding box
    regionbox = shapely.geometry.Polygon([
        (regionbounds['longitude'][0], regionbounds['latitude'][0]),
        (regionbounds['longitude'][1], regionbounds['latitude'][0]),
        (regionbounds['longitude'][1], regionbounds['latitude'][1]),
        (regionbounds['longitude'][0], regionbounds['latitude'][1]),
    ])

    ### Get subset of points in dfcoords that lie within bounding box
    ### (note that the original index is kept and labeled as 'index')
    regioncoords = dfcoords.loc[
        (dfcoords[latlabel] <= regionbounds['latitude'][1])
        & (dfcoords[latlabel] >= regionbounds['latitude'][0])
        & (dfcoords[lonlabel] <= regionbounds['longitude'][1])
        & (dfcoords[lonlabel] >= regionbounds['longitude'][0])
    ].reset_index().copy()

    ###### Get Voronoi polygons
    dfpoly = voronoi_polygons(regioncoords)
    ### Clip off the funky edge polygons
    dfpoly['poly'] = dfpoly.intersection(regionbox)

    ###### Get overlap of Voronoi polygons with polyavailable
    sites_tomodel = set()
    indices = dfpoly.index.copy()
    polys_zoneoverlap =  {index: [] for index in indices}
    areas_zoneoverlap =  {index: [] for index in indices}
    regionpoly2site = {index: None for index in indices}
    area_total = 0

    ### Get hull of available region - maybe slightly speeds up intersections?
    hull = polyoverlap.convex_hull

    ### Loop over Voronoi polygons
    for index in tqdm(dfpoly.index):
        poly = dfpoly.loc[index, 'poly']

        ### First check if poly bounding box overlaps convex hull of zone
        if poly.envelope.intersection(hull).is_empty:
            area = 0
            overlap = shapely.geometry.Polygon()
        else:
            ### Calculate overlap area
            area, overlap = polyoverlaparea_km2(
                poly, polyoverlap, method='intersection', returnpoly=True)
        
        ### Add the area to the total
        area_total += area
        polys_zoneoverlap[index].append(overlap)
        areas_zoneoverlap[index].append(area)
        
        ### If there is overlap, pull out additional data
        if area > 0:
            querylon, querylat = dfpoly.loc[
                index, ['centroid_lon','centroid_lat']].values
            ### Get closest point in regioncoords
            ### (Note that this returns the iloc index of regioncoords,
            ###  which is not the same as the dfcoords index)
            closest_index = closestpoint_simple(
                querylons=[querylon], querylats=[querylat],
                reflons=regioncoords[lonlabel].values, reflats=regioncoords[latlabel].values,
            )[0]
            ### Save the poly-to-dfcoords lineup. The saved value will either be 
            ### the original loc index of dfcoords (if index_coords is None)
            ### or the value of the index_coords column.
            regionpoly2site[index] = regioncoords.loc[closest_index, indexlabel]
            sites_tomodel.add(regionpoly2site[index])

    ### Create key for overlap area by site
    polyweights = pd.DataFrame(areas_zoneoverlap, index=['km2']).T
    polyweights = polyweights.merge(
        pd.DataFrame(regionpoly2site, index=[indexlabel]).T,
        left_index=True, right_index=True)[[indexlabel,'km2']]

    if returnfull == False:
        pass
    else:
        import geopandas as gpd
        polyoverlaps = pd.DataFrame(polys_zoneoverlap, index=['geometry']).T
        polyoverlaps = polyoverlaps.merge(
            pd.DataFrame(regionpoly2site, index=[indexlabel]).T,
            left_index=True, right_index=True)[[indexlabel,'geometry']]
        polyweights = gpd.GeoDataFrame(polyweights.merge(
            polyoverlaps, on=indexlabel)[[indexlabel, 'km2', 'geometry']])
    
    return polyweights
    # return polyweights, polyoverlaps



def partition_jenks(dfin, column, bins, returnbreaks=False):
    """
    """
    import jenkspy
    ### Test inputs
    assert type(dfin) is pd.DataFrame
    assert type(bins) is int
    ### Copy input dataframe
    df = dfin.copy()
    
    ### Calculate breaks
    breaks = jenkspy.jenks_breaks(df[column].values, bins)
    
    ### Return the breaks if that's all you want
    if returnbreaks == True:
        return breaks
    
    ###### Assign the labels
    df['jbin'] = 'empty'
    
    ### Start with bottom bin
    jbin = 0; jbreak = breaks[1]
    df.loc[(df[column] <= jbreak), 'jbin'] = jbin
    
    ### Higher bins
    for jbin, jbreak in enumerate(breaks[2:]):
        jbin += 1 ### Correct it since we're starting one spot into the list
        df.loc[(df[column] <= jbreak)
               & (df[column] > breaks[jbin]),
               'jbin'
              ] = jbin
    
    ### Check to make sure we've labeled every point
    if df['jbin'].dtype != int:
        if any(df['jbin'] == 'empty'):
            print(df.loc[df['jbin']=='empty'])
            raise Exception('Missing points')
        
    return df['jbin']

