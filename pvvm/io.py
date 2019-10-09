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

###########################
### IMPORT PROJECT PATH ###
import pvvm.settings
revmpath = pvvm.settings.revmpath
datapath = pvvm.settings.datapath
apikeys = pvvm.settings.apikeys
nsrdbparams = pvvm.settings.nsrdbparams

###################
### IMPORT DATA ###

##################
### Solar and LMPs

def getNSRDBfile(filepath, filename, year, resolution=30, forcemidnight=False):
    '''
    '''
    if resolution not in [30, 60]:
        raise Exception("resolution must be 30 or 60.")
    if forcemidnight not in [True, False]:
        raise Exception("forcemidnight must be True or False.")
    if type(year) != int and year != 'tmy':
        raise Exception("Bad year query. Must be integer or 'tmy'.")

    filepath = pvvm.toolbox.pathify(filepath)
    
    info = pd.read_csv(filepath+filename, nrows=1)
    timezone, elevation = int(info['Local Time Zone'][0]), float(info['Elevation'][0])
    tz = 'Etc/GMT{0:+}'.format(-timezone) ### <-- {:+} is also ok

    dfsun = pd.read_csv(filepath+filename, skiprows=2)
    
    if type(year) == int and forcemidnight == True:
        dfsun = dfsun.set_index(pd.date_range('1/1/{yr}'.format(yr=year), 
            freq=str(resolution)+'Min', 
            periods=yearhours(year)*60/resolution))
    else:
        dfsun.index = pd.to_datetime(dfsun[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        if year == 'tmy': 
            ### Changed on 20180105 to work with site-solar-cf-opt-3-regeo.py
            # del dfsun['Unnamed: 10']
            dfsun.drop('Unnamed: 10', axis=1, errors='ignore', inplace=True)

    dfsun.index = dfsun.index.tz_localize(tz)
    
    return dfsun, info, tz, elevation

def getNSRDBinfo(filepath=None, filename=None, **kwargs):
    """
    """
    if (filepath is None) and (filename is not None):
        inpath = filename
    elif (filename is None) and (filepath is not None):
        inpath = filepath
    elif (filepath is not None) and (filename is not None):
        inpath = os.path.join(filepath, filename)

    # filepath = pvvm.toolbox.pathify(filepath)
    info = pd.read_csv(inpath, nrows=1).T.to_dict()[0]
    return info


def queryNSRDBfile(filename=None, year=None, filepath=None, resolution=None, 
    forcemidnight=False, save=False, download=True, filetype=None,
    attributes=None, leap_day='true', utc='false', psmversion=3,
    returnfilename=False):
    '''
    Notes
    -----
    * filepath must be None if doing arbitrary query
    '''
    ### Set defaults
    if year in [None, 'tmy']:
        year = 'tmy'
    else:
        year = int(year)

    if resolution is None:
        if year == 'tmy': resolution = 60
        else: resolution = 30

    ### Clean up inputs
    filepath = pvvm.toolbox.pathify(filepath)
    fullpath = os.path.join(filepath, filename)

    ### Set filenames in case function is being used for query
    if filepath == '':
        querypath = revmpath + 'Data/NSRDB/'
    else:
        querypath = filepath
    if filetype is None:
        filetype = '.csv'
    queryname = filename.replace(' ','_')
    fullquerypath = os.path.join(
        querypath, queryname+'-{}{}'.format(year, filetype))
    
    ### Get NSRDB file for arbitrary query
    if (not os.path.exists(fullpath)) and (not os.path.exists(fullquerypath)):
        if download == False:
            raise Exception('Cannot locate {}'.format(fullpath))

        ###### Get lat/lon from googlemaps if type(query) is str
        ### Set up googlemaps
        import googlemaps
        gmaps = googlemaps.Client(key=apikeys['googlemaps'])

        ### Get latitude and longitude of query
        location = gmaps.geocode(filename)
        lat = location[0]['geometry']['location']['lat']
        lon = location[0]['geometry']['location']['lng']

        ### Set defaults if necessary
        if attributes is None:
            attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed'
        if year == 2016:
            psmversion = 3
        if filepath == '':
            filepath = revmpath + 'Data/NSRDB/'
        os.makedirs(filepath, exist_ok=True)

        ### Download file from NSRDB
        dfin, fullpath = pvvm.data.downloadNSRDBfile(
            lat=lat, lon=lon, year=year, filepath=filepath,
            nodename=filename.replace(' ','_'), filetype=filetype,
            attributes=attributes, leap_day=leap_day, interval=str(resolution),
            utc=utc, psmversion=psmversion, write=True, return_savename=True)

    elif (not os.path.exists(fullpath)) and (os.path.exists(fullquerypath)):
        fullpath = fullquerypath

    ### Load NSRDB file    
    info = pd.read_csv(fullpath, nrows=1)
    timezone, elevation = int(info['Local Time Zone'][0]), float(info['Elevation'][0])
    tz = 'Etc/GMT{0:+}'.format(-timezone) ### <-- {:+} is also ok

    dfsun = pd.read_csv(fullpath, skiprows=2)
    
    if type(year) == int and forcemidnight == True:
        dfsun = dfsun.set_index(pd.date_range('1/1/{yr}'.format(yr=year), 
            freq=str(resolution)+'Min', 
            periods=yearhours(year)*60/resolution))
    else:
        dfsun.index = pd.to_datetime(dfsun[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        if year == 'tmy': 
            ### Changed on 20180105 to work with site-solar-cf-opt-3-regeo.py
            # del dfsun['Unnamed: 10']
            dfsun.drop('Unnamed: 10', axis=1, errors='ignore', inplace=True)

    dfsun.index = dfsun.index.tz_localize(tz)
    
    if returnfilename:
        return dfsun, info, tz, elevation, fullpath
    else:
        return dfsun, info, tz, elevation

def getLMPfile(filepath, filename, tz, squeeze=False, product='lmp'):
    """
    To use for MCC, MCL, MCE (instead of LMP): 
    * Keep default filepath, to io/lmp-nodal/da/, and specify
      product='mcc' (or 'mcl' or 'mce')
    * OR keep product as 'lmp' or None, and set filepath to the folder
      containing the MCC/MCL/MCE file (i.e. to 'io/mcc-nodal/da/')
    """
    ### filepathify filepath
    if filepath[-1] != '/': 
        filepath = str(filepath+'/')

    ### Normalize product
    if (product is None) or product.lower() in ['lmp']:
        product = 'lmp'
    elif product.lower() in ['mcl','mlc','loss','losses']:
        product = 'mcl'
    elif product.lower() in ['mcc','congestion']:
        product = 'mcc'
    elif product.lower() in ['mce','mec','energy']:
        product = 'mce'
    else:
        raise Exception("invalid product: {}".format(product))

    ### Modifity lmp path to product path, if necessary
    if product in ['mcc', 'mce', 'mcl']:
        if filepath.count('lmp') != 1:
            print(filepath)
            print(product)
            raise Exception("No 'lmp' found in filepath to replace")
        filepath = filepath.replace('lmp', product)

    ### Load file
    dflmp = pd.read_csv(filepath+filename, index_col=0, header=None,
        names=[product], squeeze=squeeze)
    dflmp.index.names = [None]
    dflmp.index = pd.to_datetime(dflmp.index).tz_convert(tz) ###### Changed 20190412
    # dflmp.index = dflmp.index.tz_localize('UTC')
    # dflmp.index = dflmp.index.tz_convert(tz)
    return dflmp

##################
### Solar capacity

def geteia(datafile, datayear=2017, columns='all', states='all',
    status='operable', listcolumns=False, download=True):
    """
    Load dataframe of data from EIA 860 or 923 report

    Parameters
    ----------
    datayear: int
    datafile: str in ['plant', 'generator', 'solar', 'wind', '923', '860m']
    columns: str in ['all', 'important'] or list of column names
    status: str in ['operable', 'proposed', 'retired']

    Returns
    -------
    if not listcolumns: pd.DataFrame
    if listcolumns: list of all columns in pd.DataFrame

    General info
    ------------
    solartechs = ['Solar Photovoltaic', 
                  'Solar Thermal without Energy Storage',
                  'Solar Thermal with Energy Storage']

    windtechs = ['Onshore Wind Turbine',
                 'Offshore Wind Turbine']
    """
    ### Warn if mismatched inputs
    if (datafile in ['solar', 'wind']) and (status == 'proposed'):
        warn("No 'Proposed' sheet for {}; returning 'Operable'.".format(datafile))

    ### Define files
    ## Form 860
    if datayear in [2013, 2014, 2015, 2016, 2017]:
        files = {
            'plant': '{}EIA/EIA860/eia860{y}/2___Plant_Y{y}.xlsx'.format(
                datapath, y=datayear),
            'generator': '{}EIA/EIA860/eia860{y}/3_1_Generator_Y{y}.xlsx'.format(
                datapath, y=datayear),
            'solar': '{}EIA/EIA860/eia860{y}/3_3_Solar_Y{y}.xlsx'.format(
                datapath, y=datayear),
            'wind': '{}EIA/EIA860/eia860{y}/3_2_Wind_Y{y}.xlsx'.format(
                datapath, y=datayear),
        }
        sheets = {'operable': 'Operable', 'proposed': 'Proposed',
                  'retired': 'Retired and Canceled'}
    elif datayear == 2012:
        files = {
            'plant': '{}EIA/EIA860/eia860{y}/PlantY{y}.xlsx'.format(
                datapath, y=datayear),
            'generator': '{}EIA/EIA860/eia860{y}/GeneratorY{y}.xlsx'.format(
                datapath, y=datayear),
        }
        sheets = {'operable': 'Operable', 'proposed': 'Proposed',
                  'retired': 'Retired & Canceled'}
    elif datayear == 2011:
        files = {
            'plant': '{}EIA/EIA860/eia860{y}/Plant.xlsx'.format(
                datapath, y=datayear),
            'generator': '{}EIA/EIA860/eia860{y}/GeneratorY{y}.xlsx'.format(
                datapath, y=datayear),
        }
        sheets = {'operable': 'operable', 'proposed': 'proposed',
                  'retired': 'retired & canceled'}
    elif datayear == 2010:
        files = {
            'plant': '{}EIA/EIA860/eia860{y}/PlantY{y}.xls'.format(
                datapath, y=datayear),
            'generator': '{}EIA/EIA860/eia860{y}/GeneratorsY{y}.xls'.format(
                datapath, y=datayear),
            '923': '{}EIA/EIA923/f923_{y}/EIA923 SCHEDULES 2_3_4_5 Final {y}.xls'.format(
                datapath, y=datayear),
        }
        sheets = {'operable': 'Exist', 'proposed': 'Prop', 'retired': 'Ret_IP'}
    elif datayear in [2007, 2008, 2009]:
        files = {
            'plant': '{}EIA/EIA860/eia86020{y}/PlantY{y}.xls'.format(
                datapath, y=str(datayear)[2:]),
            'generator': '{}EIA/EIA860/eia86020{y}/GeneratorY{y}.xls'.format(
                datapath, y=str(datayear)[2:]),
            '923': (
                '{}EIA/EIA923/f923_{y}/EIA923 SCHEDULES 2_3_4_5 M '
                'Final {y} REVISED 05252011.XLS').format(
                    datapath, y=datayear),
        }
        sheets = {'operable': 'Exist', 'proposed': 'Prop', 'retired': 'Ret_IP'}

    ## Form 923
    if datayear in [2012, 2014, 2015, 2016]:
        files['923'] = (
            '{}EIA/EIA923/f923_{y}/EIA923_Schedules_2_3_4_5_M_12_{y}_Final_Revision.xlsx'.format(
                datapath, y=datayear))
    elif datayear in [2017]:
        files['923'] = (
            '{}EIA/EIA923/f923_{y}/EIA923_Schedules_2_3_4_5_M_12_{y}_Final.xlsx'.format(
                datapath, y=datayear))
        files['storage'] = '{}EIA/EIA860/eia860{y}/3_4_Energy_Storage_Y{y}.xlsx'.format(
                datapath, y=datayear)
    elif datayear in [2011, 2013]:
        files['923'] = (
            '{}EIA/EIA923/f923_{y}/EIA923_Schedules_2_3_4_5_{y}_Final_Revision.xlsx'.format(
                datapath, y=datayear))
    elif datayear in [2008]:
        files['923'] = '{}EIA/EIA923/f923_{y}/eia923December{y}.xls'.format(datapath, y=datayear)
    elif datayear in [2007]:
        files['923'] = '{}EIA/EIA923/f906920_{y}/f906920_{y}.xls'.format(datapath, y=datayear)

    ## Form 860M
    ## Pick the latest file
    if datafile.lower() == '860m':
        outpath = datapath + 'EIA/EIA860M/'
        files['860m'] = outpath+'january_generator2018.xlsx'
        ### Downlad it if it doesn't exist
        if not os.path.exists(files['860m']):
            outpath = datapath + 'EIA/EIA860M/'
            os.makedirs(outpath, exist_ok=True)
            urlbase = 'https://www.eia.gov/electricity/data/eia860m/xls/'
            urllib.request.urlretrieve(
                urlbase+'january_generator2018.xlsx',
                outpath+'january_generator2018.xlsx')


    ### Define important columns
    columns_important = {
        'plant': [
            'Utility ID', 'Utility Name', 
            'Plant Code', 'Plant Name',
            'Street Address', 'City', 'State', 'Zip', 'County',
            'Latitude', 'Longitude', 
            'NERC Region',
            'Balancing Authority Code', 'Balancing Authority Name',
        ],
        'generator': [
            'Utility ID', 'Utility Name', 
            'Plant Code', 'Plant Name', 
            'State', 'County', 
            'Generator ID', 
            'Status', 'Operating Month', 'Operating Year',
            'Technology', 'Prime Mover', 'Unit Code',
            'Nameplate Capacity (MW)', 
            'Nameplate Power Factor', 'Minimum Load (MW)',
            'Associated with Combined Heat and Power System', 
            'Time from Cold Shutdown to Full Load', 
        ],
        'solar': [
            'Utility ID', 'Utility Name', 
            'Plant Code', 'Plant Name', 
            'State', 'County', 
            'Generator ID', 
            'Status', 'Operating Month', 'Operating Year', 
            'Technology', 'Prime Mover', 'Nameplate Capacity (MW)', 
            'Single-Axis Tracking?', 'Dual-Axis Tracking?', 'Fixed Tilt?', 
            'DC Net Capacity (MW)'
        ],
        'wind': [
            'Utility ID', 'Utility Name',
            'Plant Code', 'Plant Name', 
            'State', 'County',
            'Generator ID', 
            'Status', 'Operating Month', 'Operating Year',
            'Technology', 'Nameplate Capacity (MW)',
            'Number of Turbines', 'Predominant Turbine Model Number',
            'Wind Quality Class', 'Turbine Hub Height (Feet)'
        ],
        '923': slice(None),
        '860m': slice(None),
    }
    if status == 'proposed':
        columns_important['generator'] = [
            'Utility ID', 'Utility Name',
            'Plant Code', 'Plant Name', 
            'State', 'County', 
            'Generator ID', 
            'Status', 'Current Month', 'Current Year', 
            'Technology', 'Prime Mover', 'Unit Code', 
            'Nameplate Capacity (MW)',
            'Nameplate Power Factor', 'Minimum Load (MW)',
            'Associated with Combined Heat and Power System', 
        ]
    if status == 'retired':
        columns_important['generator'] = [
            'Utility ID', 'Utility Name',
            'Plant Code', 'Plant Name', 
            'State', 'County', 
            'Generator ID', 
            'Status', 'Operating Month', 'Operating Year',
            'Retirement Month', 'Retirement Year',
            'Technology', 'Prime Mover', 'Unit Code', 
            'Nameplate Capacity (MW)',
            'Nameplate Power Factor', 'Minimum Load (MW)',
            'Associated with Combined Heat and Power System', 
            'Time from Cold Shutdown to Full Load', 
        ]
        columns_important['solar'] = [
            'Utility ID', 'Utility Name', 
            'Plant Code', 'Plant Name', 
            'State', 'County', 
            'Generator ID', 
            'Status', 'Retirement Month', 'Retirement Year', 
            'Technology', 'Prime Mover', 'Nameplate Capacity (MW)', 
            'Single-Axis Tracking?', 'Dual-Axis Tracking?', 'Fixed Tilt?', 
            'DC Net Capacity (MW)'
        ]

    ### Set skiprows / skipfooters (currently only works for 'generator')
    skipfooter = {
        (2009, 'operable'): 0, (2009, 'proposed'): 0, (2009, 'retired'): 0, 
        (2010, 'operable'): 0, (2010, 'proposed'): 0, (2010, 'retired'): 0, 
        (2011, 'operable'): 0, (2011, 'proposed'): 0, (2011, 'retired'): 0, 
        (2012, 'operable'): 0, (2012, 'proposed'): 0, (2012, 'retired'): 0, 
        (2013, 'operable'): 0, (2013, 'proposed'): 0, (2013, 'retired'): 0, 
        (2014, 'operable'): 0, (2014, 'proposed'): 0, (2014, 'retired'): 0, 
        (2015, 'operable'): 1, (2015, 'proposed'): 1, (2015, 'retired'): 0, 
        (2016, 'operable'): 1, (2016, 'proposed'): 1, (2016, 'retired'): 0, 
        (2017, 'operable'): 1, (2017, 'proposed'): 1, (2017, 'retired'): 1, 
    }
    skiprows = {
        (2009, 'operable'): 0, (2009, 'proposed'): 0, (2009, 'retired'): 0, 
        (2010, 'operable'): 0, (2010, 'proposed'): 0, (2010, 'retired'): 0, 
        (2011, 'operable'): 1, (2011, 'proposed'): 1, (2011, 'retired'): 1, 
        (2012, 'operable'): 1, (2012, 'proposed'): 1, (2012, 'retired'): 1, 
        (2013, 'operable'): 1, (2013, 'proposed'): 1, (2013, 'retired'): 1, 
        (2014, 'operable'): 1, (2014, 'proposed'): 1, (2014, 'retired'): 1, 
        (2015, 'operable'): 1, (2015, 'proposed'): 1, (2015, 'retired'): 1, 
        (2016, 'operable'): 1, (2016, 'proposed'): 1, (2016, 'retired'): 1, 
        (2017, 'operable'): 1, (2017, 'proposed'): 1, (2017, 'retired'): 1, 
    }
    plantsheet = {
        2007: 'PlantY07', 2008: 'PlantY08', 2009: 'PlantY09', 2010: 'PlantY2010', 
        2011: 'plant2011', 2012: 'plant2012', 2013: 'Plant', 2014: 'Plant', 
        2015: 'Plant', 2016: 'Plant', 2017: 'Plant', 2018: 'Plant', 
    }
    skiprows_plant = {
        2007: 0, 2008: 0, 2009: 0, 2010: 0, 2011: 1, 2012: 1, 2013: 1, 
        2014: 1, 2015: 1, 2016: 1, 2017: 1,
    }
    ### Load dataframe
    for i in range(5): # 5 is the maximum number of retries
        try:
            if datafile == 'plant':
                df = pd.read_excel(
                    files['plant'],
                    sheet_name=plantsheet[datayear], 
                    skiprows=skiprows_plant[datayear], skipfooter=0)
            elif datafile == 'generator':
                df = pd.read_excel(
                    files['generator'],
                    sheet_name=sheets[status], 
                    skiprows=skiprows[(datayear, status)], 
                    skipfooter=skipfooter[(datayear, status)])
            elif (datafile == 'solar') and (status == 'operable'):
                df = pd.read_excel(
                    files['solar'], 
                    sheet_name='Operable',
                    skiprows=1)
            elif (datafile == 'solar') and (status == 'retired'):
                df = pd.read_excel(
                    files['solar'], 
                    sheet_name='Retired and Canceled',
                    skiprows=1)
            elif datafile in ['wind','storage']:
                df = pd.read_excel(
                    files[datafile], 
                    sheet_name='Operable',
                    skiprows=1)
            ## 923
            elif datafile == '923':
                if datayear in [2011, 2012, 2013, 2014, 2015, 2016, 2017]:
                    df = pd.read_excel(
                        files['923'],
                        sheet_name='Page 1 Generation and Fuel Data',
                        skiprows=5,
                        header=0)
                elif datayear in [2007,2008,2009,2010]:
                    df = pd.read_excel(
                        files['923'],
                        sheet_name='Page 1 Generation and Fuel Data',
                        skiprows=7,
                        header=0)
            ## 860M
            elif (datafile.lower() == '860m') and (status.lower() in ['operable', 'operating']):
                df = pd.read_excel(files['860m'], sheet_name='Operating', 
                    skiprows=1, skipfooter=1)
            elif (datafile.lower() == '860m') and (status.lower() in ['proposed', 'planned']):
                df = pd.read_excel(files['860m'], sheet_name='Planned', 
                    skiprows=1, skipfooter=1)
            elif (datafile.lower() == '860m') and (status.lower() in ['retired']):
                df = pd.read_excel(files['860m'], sheet_name='Retired', 
                    skiprows=1, skipfooter=1)
            elif (datafile.lower() == '860m') and (status.lower() in ['canceled']):
                df = pd.read_excel(files['860m'], sheet_name='Canceled or Postponed', 
                    skiprows=1, skipfooter=1)

        except FileNotFoundError as err:
            if download == False:
                raise FileNotFoundError(err)
            ### Download it
            if datafile == '923':
                if datayear >= 2008:
                    filename = 'f923_{}'.format(datayear)
                else:
                    filename = 'f906920_{}'.format(datayear)
                url = 'https://www.eia.gov/electricity/data/eia923/archive/xls/{}.zip'.format(
                    filename)
                outpath = datapath+'EIA/EIA923/{}/'.format(filename)
                os.makedirs(outpath, exist_ok=True)
                urllib.request.urlretrieve(
                    url, datapath+'EIA/EIA923/{}.zip'.format(filename))
                ### Unzip it
                zip_ref = zipfile.ZipFile(datapath+'EIA/EIA923/{}.zip'.format(filename), 'r')
                zip_ref.extractall(outpath)
                zip_ref.close()
                print('Downloaded file from {} and saved at {}'.format(url, outpath))
                continue
            
            else:
                url = 'https://www.eia.gov/electricity/data/eia860/xls/eia860{}.zip'.format(
                    datayear)
                outpath = datapath+'EIA/EIA860/eia860{}/'.format(datayear)
                os.makedirs(outpath, exist_ok=True)
                urllib.request.urlretrieve(
                    url, datapath+'EIA/EIA860/eia860{}.zip'.format(datayear))
                ### Unzip it
                zip_ref = zipfile.ZipFile(datapath+'EIA/EIA860/eia860{}.zip'.format(datayear), 'r')
                zip_ref.extractall(outpath)
                zip_ref.close()
                print('Downloaded file from {} and saved at {}'.format(url, outpath))
                continue

        else:
            break

    else:
        raise FileNotFoundError("Can't find {}".format(files[datafile]))

    ### Return column list, if asked for
    if listcolumns:
        return df.columns

    ### Intermediate dataframe with states of interest
    ## Cover different State column labels, and strip whitespace
    statecol = 'State'
    if datafile.lower() == '860m': 
        statecol = 'Plant State'
        df.rename(columns=dict(zip(list(df.columns), [c.strip() for c in list(df.columns)])),
          inplace=True)
    ## Select the data
    if states == 'all':
        dfout = df
    elif (type(states) == str) and (len(states) == 2):
        dfout = df[df[statecol] == states]
    elif type(states) == list:
        dfout = df[df[statecol].map(lambda x: x in states)]

    ### Return dataframe with columns of interest
    if columns == 'all':
        return dfout
    elif columns == 'important':
        return dfout[columns_important[datafile]]
    elif type(columns) == list:
        return dfout[columns]

def getopenpv(states='all', columns='all', tts=True, download=True):
    """
    """
    ### Filepaths
    filepath1 = os.path.join(datapath, 'LBNL/OpenPV/2017/openpv_tts_data/',
                             'TTSX_LBNL_OpenPV_public_file_p1.xlsx')
    filepath2 = os.path.join(datapath, 'LBNL/OpenPV/2017/openpv_tts_data/',
                             'TTSX_LBNL_OpenPV_public_file_p2.xlsx')
    # filepath1csv = os.path.join(
    #     datapath, 
    #     'LBNL/OpenPV/2017/openpv_tts_data/TTSX_LBNL_OpenPV_public_file_p1.csv')
    # filepath2csv = os.path.join(
    #     datapath, 
    #     'LBNL/OpenPV/2017/openpv_tts_data/TTSX_LBNL_OpenPV_public_file_p2.csv')
    filepathraw = os.path.join(datapath, 'LBNL/openpv_all.csv')

    
    ### Columns
    if tts:
        columns_all = [
            'Data Provider', 'System ID (from Data Provider)',
            'System ID (Tracking the Sun)', 'Installation Date', 'System Size',
            'Total Installed Price', 'Appraised Value Flag', 'Sales Tax Cost',
            'Rebate or Grant', 'Performance-Based Incentive (Annual Payment)',
            'Performance-Based Incentives (Duration)',
            'Feed-in Tariff (Annual Payment)', 'Feed-in Tariff (Duration)',
            'Customer Segment', 'New Construction', 'Tracking', 'Tracking Type',
            'Ground Mounted', 'Battery System', 'Zip Code', 'City', 'County',
            'State', 'Utility Service Territory', 'Third-Party Owned',
            'Installer Name', 'Self-Installed', 'Azimuth #1', 'Azimuth #2',
            'Azimuth #3', 'Tilt #1', 'Tilt #2', 'Tilt #3', 'Module Manufacturer #1',
            'Module Manufacturer #2', 'Module Manufacturer #3', 'Module Model #1',
            'Module Model #2', 'Module Model #3', 'Module Technology #1',
            'Module Technology #2', 'Module Technology #3', 'BIPV Module #1',
            'BIPV Module #2', 'BIPV Module #3', 'Module Efficiency #1',
            'Module Efficiency #2', 'Module Efficiency #3', 'Inverter Manufacturer',
            'Inverter Model', 'Microinverter', 'DC Optimizer']
        columns_important = [
            'Installation Date', 'System Size',
            'Tracking', 'Tracking Type',
            'Ground Mounted', 'Customer Segment', 
            'Zip Code', 'City', 'County', 'State',
            'Azimuth #1', 'Tilt #1',
            'Module Technology #1']
    else:
        columns_all = [
            'state', 'date_installed', 'incentive_prog_names', 'type', 'size_kw',
            'appraised', 'zipcode', 'install_type', 'installer', 'cost_per_watt',
            'cost', 'lbnl_tts_version_year', 'lbnl_tts', 'city', 'utility_clean',
            'tech_1', 'model1_clean', 'county', 'annual_PV_prod',
            'annual_insolation', 'rebate', 'sales_tax_cost', 'tilt1',
            'tracking_type', 'azimuth1', 'manuf2_clean', 'manuf3_clean',
            'manuf1_clean', 'inv_man_clean', 'reported_annual_energy_prod',
            'incentivetype', 'year_app_implied', 'year', 'npv_fit_real',
            'application_implied', 'npv_pbi_real', 'other_incentive',
            'appraised_cluster', 'inflation', 'other_incentive_real',
            'zip_available', 'cust_city', 'pbi', 'pbi_real', 'pbi_length',
            'application', 'fit_length', 'fit_rate', 'fit_payment',
            '_3rdparty_implied', 'utility', 'install_price_real_w', 'install_price',
            'installer_clean', 'manuf1_', 'inverter_reported', 'rebate_real',
            'model1', '_3rdparty', 'inv_model_reported', 'microinv_solarhub',
            'bipv_3', 'bipv_2', 'bipv_1', 'sales_tax_rate', 'sales_tax_cost_real',
            'bipv_all', 'thinfilm_all', 'china', 'sys_sizeac', 'pbi_rate',
            'new_constr', 'effic_1', 'cust_county', 'tracking', 'inv_model_clean',
            'mod_cost_real', 'inv_cost_real', 'bos_powerclerk_real',
            'permitting_real', '3rdparty']
        columns_important = [
            'state', 'date_installed', 'size_kw',
            'zipcode', 'install_type', 
            'lbnl_tts_version_year', 'lbnl_tts', 'city', 
            'tech_1', 'county', 
            'tilt1', 'tracking_type', 'azimuth1',
            'tracking', 
        ]
    
    if columns == 'all':
        usecolumns = columns_all
    elif columns == 'important':
        usecolumns = columns_important
    else:
        usecolumns = columns
    
    ### Load data, either TTS cleaned or OpenPV raw
    for i in range(5): # 5 is the maximum number of retries
        try:
            if tts:
                # df1csv = pd.read_csv(
                #     filepath1csv, usecols=usecolumns,
                #     na_values=[-9999, -9999., '-9999'], encoding='latin-1')
                # df2csv = pd.read_csv(
                #     filepath2csv, usecols=usecolumns,
                #     na_values=[-9999, -9999., '-9999'], encoding='latin-1')
                df1 = pd.read_excel(
                    filepath1, #usecols=usecolumns,
                    na_values=[-9999, -9999., '-9999'], #encoding='latin-1'
                    )
                df2 = pd.read_excel(
                    filepath2, #usecols=usecolumns,
                    na_values=[-9999, -9999., '-9999'], #encoding='latin-1'
                    )

                dfin = pd.concat([df1, df2], ignore_index=True)
            else:
                dfin = pd.read_csv(
                    filepathraw,
                    usecols=usecolumns,
                    na_values=[-9999, -9999., '-9999'])

        except FileNotFoundError as err:
            if download == False:
                raise FileNotFoundError(err)
            ### Download it
            url = 'https://openpv.nrel.gov/assets/data/openpv_tts_data.zip'
            outpath = datapath + 'LBNL/OpenPV/2017/openpv_tts_data/'
            os.makedirs(outpath, exist_ok=True)
            urllib.request.urlretrieve(url, datapath+'LBNL/OpenPV/2017/openpv_tts_data.zip')
            ### Unzip it
            zip_ref = zipfile.ZipFile(datapath+'LBNL/OpenPV/2017/openpv_tts_data.zip', 'r')
            zip_ref.extractall(datapath+'LBNL/OpenPV/2017/')
            zip_ref.close()
            print('Downloaded file from {} and saved at {}'.format(url, outpath))
            continue

        else:
            break

    else:            
        raise FileNotFoundError("Can't find {}".format(filepath1))
    
    ### Make and return output dataframe with states of interest
    if tts:
        if states == 'all':
            df = dfin
        elif (type(states) == str) and (len(states) == 2):
            df = dfin[dfin['State'] == states]
        elif type(states) == list:
            df = dfin[dfin['State'].map(lambda x: x in states)]
    else:
        if states == 'all':
            df = dfin
        elif (type(states) == str) and (len(states) == 2):
            df = dfin[dfin['state'] == states]
        elif type(states) == list:
            df = dfin[dfin['state'].map(lambda x: x in states)]
        
    ### Reset index after row selection
    df.reset_index(drop=True, inplace=True)
    
    return df

def getferc(data='demand', datalist=False, form=714):
    """
    Get load from FERC database (note that this is just for full-ISO load)
    https://www.ferc.gov/docs-filing/forms/form-714/data/form714-database.zip
    20190625: downloaded 2018 FERC database and replaced old file
    """
    folderpath = datapath+'FERC/form714-database/'
    ### Download the file if it doesn't exist
    if not os.path.exists(folderpath):
        ### Download it
        url = 'https://www.ferc.gov/docs-filing/forms/form-714/data/form714-database.zip'
        os.makedirs(folderpath, exist_ok=True)
        urllib.request.urlretrieve(url, datapath+'FERC/form714-database.zip')
        ### Unzip it
        zip_ref = zipfile.ZipFile(datapath+'FERC/form714-database.zip', 'r')
        zip_ref.extractall(folderpath)
        zip_ref.close()
        print('Downloaded files from {} and saved at {}'.format(url, folderpath))
    
    filenames = {
        'id':                'Respondent IDs.csv',
        'certification':     'Part 1 Schedule 1 - Identification Certification.csv',
        'ba plants':         'Part 2 Schedule 1 - Balancing Authority Generating Plants.csv',
        'ba monthly demand': 'Part 2 Schedule 2 - Balancing Authority Monthly Demand.csv',
        'ba load':           'Part 2 Schedule 3 - Balancing Authority Net Energy For Load.csv',
        'adjacent ba':       'Part 2 Schedule 4 - Adjacent Balancing Authorities.csv',
        'ba interchange':    'Part 2 Schedule 5 - Balancing Authority Interchange.csv',
        'lambda':            'Part 2 Schedule 6 - Balancing Authority hourly System Lambda.csv',
        'lambda description':'Part 2 Schedule 6 - System Lambda Description.csv',
        'pa description':    'Part 3 Schedule 1 - Planning Area Description.csv',
        'forecast demand':   'Part 3 Schedule 2 - Planning Area Forecast Demand.csv',
        'demand':            'Part 3 Schedule 2 - Planning Area Hourly Demand.csv',
    }
    if datalist is True:
        return filenames
    
    filepath = os.path.join(
        folderpath, filenames[data])
    
    if data == 'demand':
        parse_dates = ['plan_date']
        infer_datetime_format = True
    else:
        parse_dates = False
        infer_datetime_format = False
    try:
        df = pd.read_csv(
            filepath, parse_dates=parse_dates, infer_datetime_format=infer_datetime_format)
    except UnicodeDecodeError:
        df = pd.read_csv(
            filepath, parse_dates=parse_dates, infer_datetime_format=infer_datetime_format,
            encoding='ISO-8859-1')
            # encoding='utf-8')
    
    return df

#############################
### ISO load and full-ISO lmp

def getcaisogenmix(inpath=None, yearend=2018, download=False):
    """
    """
    ### Defaults
    if inpath is None:
        inpath = datapath+'ISO/CAISO/RenewablesWatch/'
    
    ### Set up dicts and load files
    missingfiles = []
    dfout_renew = {}
    dfout_total = {}
    dfsummaries = {}

    for year in range(2010,yearend):

        dictrenew, dicttotal = {}, {}

        for day in pvvm.toolbox.makedays(year):
            try:
                dfrenew = pd.read_csv(
                    '{}{}.txt'.format(inpath, day), 
                    delimiter='\t', engine='python',
                    skiprows=2, skipfooter=28, 
                    header=None,
                    usecols = [1,3,5,7,9,11,13,15],
                    names = [
                        'Hour', 'Geothermal', 'Biomass',
                        'Biogas', 'Small Hydro', 'Wind',
                        'Solar PV', 'Solar Thermal'],
                    na_values=[
                        ('Invalid function argument:Start time and End time '
                         'differ by less than 15 micro seconds'), 
                        '#NAME?', '#REF!', 'Resize to show all values',
                        '[-11059] No Good Data For Calculation',
                        ('The supplied DateTime represents an invalid time.  '
                         'For example, when the clock is adjusted forward, any '
                         'time in the period that is skipped is invalid.\n'
                         'Parameter name: dateTime'),
                        'Connection to the server lost.  '])

                dftotal = pd.read_csv(
                    '{}{}.txt'.format(inpath, day), 
                    delimiter='\t', engine='python',
                    skiprows=30, 
                    header=None,
                    usecols = [1,3,5,7,9,11],
                    names = [
                        'Hour', 'Renewables', 'Nuclear',
                        'Thermal', 'Imports', 'Hydro'],
                    na_values=[
                        ('Invalid function argument:Start time and End time '
                         'differ by less than 15 micro seconds'),
                        '#VALUE!', '#NAME?', '#REF!', 'Resize to show all values',
                        '[-11059] No Good Data For Calculation',
                        ('The supplied DateTime represents an invalid time.  '
                         'For example, when the clock is adjusted forward, any '
                         'time in the period that is skipped is invalid.\n'
                         'Parameter name: dateTime')])

                dictrenew[day] = dfrenew
                dicttotal[day] = dftotal

            except FileNotFoundError as err:
                missingfiles.append(day)
                pass

        dfrenewall = pd.concat(dictrenew)
        dftotalall = pd.concat(dicttotal)

        PVsum = dfrenewall['Solar PV'].astype(float).sum()
        CSPsum = dfrenewall['Solar Thermal'].astype(float).sum()
        solarsum = PVsum + CSPsum
        totalsum = dftotalall[
            ['Renewables', 'Nuclear', 'Thermal', 'Imports', 'Hydro']].astype(float).sum().sum()
        totalnoimportssum = dftotalall[
            ['Renewables', 'Nuclear', 'Thermal', 'Hydro']].astype(float).sum().sum()

        dfout_renew[year] = dfrenewall
        dfout_total[year] = dftotalall
        dfsummaries[year] = pd.Series(
            data=[
                PVsum, CSPsum, solarsum, totalsum, totalnoimportssum,
                solarsum / totalsum, solarsum / totalnoimportssum],
            index=[
                'PVsum', 'CSPsum', 'solarsum', 'totalsum', 'totalnoimportssum',
                'solarsum / totalsum', 'solarsum / totalnoimportssum'],
        )

    dfout_renew = pd.concat(dfout_renew)
    dfout_total = pd.concat(dfout_total)
    dfsummaries = pd.concat(dfsummaries, axis=1).T
    
    ### Combined dataframe and column cleanup
    dfout_renew['Solar'] = dfout_renew['Solar PV'].fillna(0) + dfout_renew['Solar Thermal'].fillna(0)
    
    dfgen = pd.concat([dfout_renew, dfout_total.drop(['Hour'], axis=1)], axis=1)
    dfgen.reset_index(level=0, drop=True, inplace=True)
    dfgen.drop(['20111106'], inplace=True) ## Data is weird on this day
    dfgen['Hour'] = dfgen['Hour'] - 1
    dfgen.reset_index(inplace=True)
    dfgen.drop('level_1', axis=1, inplace=True)
    dfgen.rename(columns={'level_0':'Date'}, inplace=True)
    dfgen['DateTime'] = dfgen.apply(
        lambda row: '{} {:02}:00'.format(int(row['Date']), int(row['Hour'])),
        axis=1)
    dfgen['DateTime'] = pd.to_datetime(dfgen['DateTime'], format='%Y%m%d %H:%M')
    dfgen['DateTimeNoDST'] = dfgen['DateTime'].apply(pvvm.toolbox.undodst)
    dfgen.index = dfgen['DateTimeNoDST']
    dfgen.index = dfgen.index.tz_localize(pvvm.toolbox.tz_iso['CAISO'])
    dfgen = dfgen.drop('DateTimeNoDST', axis=1)
    
    return dfgen

def getload(iso, year, clean=True, units='MW', 
    division='region', filepathin=None):
    """
    """
    iso = iso.upper()
    
    if iso == 'ERCOT':
        ### Set filenames
        loadfile = {}
        for i in range(2002,2015):
            loadfile[i] = '{}_ERCOT_Hourly_Load_Data.xls'.format(i)
        loadfile[2015] = 'native_Load_2015.xls'
        loadfile[2016] = 'native_Load_2016.xlsx'
        loadfile[2017] = 'native_Load_2017.xlsx'

        sheetname = {}
        for i in range(2002,2017):
            sheetname[i] = 'native_Load_{}'.format(i)
        sheetname[2017] = 'Page1_1'

        datecol = {}
        for i in range(2002,2017):
            datecol[i] = 'Hour_End'
        datecol[2017] = 'Hour Ending'
        
        ### Set filepath
        if filepathin is None:
            filepath = datapath + 'ISO/ERCOT/load/raw/'
        else:
            filepath = filepathin
        
        ### Load file
        df = pd.read_excel(
            filepath + loadfile[year],
            sheet_name=sheetname[year],
            # index_col=datecol[year], parse_dates=True, infer_datetime_format=True,
            # parse_dates=[datecol[year]]
        )
        df = df.drop(datecol[year], axis=1)
        df.index = pd.date_range(
            start='{}-01-01 00:00'.format(year),
            end='{}-12-31 23:00'.format(year),
            freq='1H', tz=pvvm.toolbox.tz_iso['ERCOT'])

    elif iso == 'ISONE':
        if clean == True:
            if filepathin is None:
                filepath = datapath + 'ISO/ISONE/load/clean/'
            else:
                filepath = filepathin

            df = pd.read_csv(
                filepath+'ISONE-realdemand-{}.csv'.format(year),
                index_col=0, parse_dates=True, infer_datetime_format=True)
            df = df.tz_localize('UTC').tz_convert(pvvm.toolbox.tz_iso[iso])
    
    elif iso == 'NYISO':
        if clean == True:
            if filepathin is None:
                filepath = datapath + 'ISO/NYISO/load/clean/'
            else:
                filepath = filepathin
        
            df = pd.read_csv(
                filepath+'NYISO-integrated-{}.csv'.format(year),
                index_col=0, parse_dates=True, infer_datetime_format=True)
            df = df.tz_localize('UTC').tz_convert(pvvm.toolbox.tz_iso[iso])

    elif iso == 'PJM':
        if clean == True:
            if filepathin is None:
                filepath = datapath + 'ISO/PJM/load/clean/'
            else:
                filepath = filepathin
        
            df = pd.read_csv(
                filepath+'PJM-metered-{}.csv'.format(year),
                index_col=0, parse_dates=True, infer_datetime_format=True)
            df = df.tz_localize('UTC').tz_convert(pvvm.toolbox.tz_iso[iso])
    
    elif iso == 'MISO':
        if clean == True:
            if filepathin is None:
                filepath = datapath + 'ISO/MISO/load/clean/'
            else:
                filepath = filepathin
        
            df = pd.read_csv(
                filepath+'MISO-actual-region-{}.csv'.format(year),
                index_col=0, parse_dates=True, infer_datetime_format=True)
            df = df.tz_localize('UTC').tz_convert(pvvm.toolbox.tz_iso[iso])
            
    elif iso == 'CAISO':
        # df = pd.read_csv(revmpath+'CAISO/io/caiso-generation-mix-raw.csv',
        #     index_col='DateTimeNoDST', parse_dates=True, infer_datetime_format=True)
        # df = df.tz_localize(tz_iso[iso])
        df = getcaisogenmix()
        df = df.drop(['Date','Hour','DateTime'], axis=1)
        df['Load'] = df.Renewables + df.Nuclear + df.Thermal + df.Imports + df.Hydro
        df = df.loc[str(year)].copy()

    ### Convert units if desired
    convert = {'MW':1, 'GW':1000, 'peak':df.max(),
               'normal':df.max(), 'normalize':df.max()}
    df = df / convert[units] # convert to GW
    
    return df

def getload_ferc(iso, year, dfferc=None, error='raise'):
    """
    Pull load for given iso-year from FERC database
    """
    ### Load FERC dataframe if necessary
    if dfferc is None:
        dfferc = getferc()
    elif type(dfferc) is pd.DataFrame:
        pass
    else:
        raise Exception('Invalid dfferc; load from getferc()')

    ### Label ISO entries
    fercid = {125: 'CAISO', 165: 'ERCOT', 321: 'MISO',
              230: 'PJM',   211: 'NYISO', 185: 'ISONE',
              116: 'APS',   210: 'NV',    228: 'PACE', 
              229: 'PACW',  232: 'PGE',   240: 'PSE', 
              122: 'BPA', ## Bonneville Power Administration
              235: 'CO', ## Public service company of CO
              272: 'WAPA-UME', ## Upper great plains
              273: 'WAPA-CM', ## Rocky mountain region
              274: 'WAPA-LC', ## Lower CO / desert southwest
              275: 'WAPA-UMW', ## Upper great plains
              252: 'SMEPA', ## South Mississippi Electric Power Assoc.
              164: 'Entergy',
              142: 'CLECO',
              196: 'LAGN', ##Louisiana Generating
              195: 'LEPA', ## Louisiana Energy & Power Authority
              }
    dfferc['ISO'] = dfferc.respondent_id.map(
        lambda x: fercid.get(x, 'None')).astype('category')

    ### Pull load for ISO-year from dfferc
    df = dfferc.loc[(dfferc.report_yr == year) & (dfferc.ISO == iso)].copy()
    ### Reshape to long format
    hourcols = ['hour{:02.0f}'.format(i) for i in range(1,25)]
    df = pd.melt(df, id_vars='plan_date', value_vars=hourcols, 
                 var_name='hour', value_name='load')
    df = df.sort_values(['plan_date','hour'])
    try:
        df.index = pd.date_range('{}-01-01'.format(year),'{}-01-01'.format(year+1),
                                 freq='H', closed='left', tz=pvvm.toolbox.tz_iso[iso])
        return df['load']
    except ValueError as err:
        print(err)
        warn('{}\nlen(df) = {}'.format(err, len(df)))
        if error == 'raise':
            raise ValueError(err)
        elif error == 'return':
            return df
        elif error in ['pass', 'ignore']:
            print('{} {}'.format(iso, year))
            pass

def getferc_lambda(ba, year, dfferc=None, error='raise'):
    """
    """
    ### Load FERC dataframe if necessary
    if dfferc is None:
        dfferc = getferc('lambda')
    elif type(dfferc) is pd.DataFrame:
        pass
    else:
        raise Exception("Invalid dfferc; get from getferc('lambda')")

    ### Get converter for respondent_id
    dfferc_id = getferc('id')
    dfferc_id['respondent_name'] = (
        dfferc_id['respondent_name'].map(lambda x: x.strip()))
    fercid2name = dict(zip(
        *dfferc_id[['respondent_id','respondent_name']].T.values))
    fercname2id = dict(zip(
        *dfferc_id[['respondent_name','respondent_id']].T.values))
    ### Specify some extra ferc ids, from pvvm.io.getload_ferc()
    fercid_custom = {
        125: 'CAISO', 165: 'ERCOT', 321: 'MISO',
        230: 'PJM',   211: 'NYISO', 185: 'ISONE',
        116: 'APS',   210: 'NV',    228: 'PACE', 
        229: 'PACW',  232: 'PGE',   240: 'PSE', 
        122: 'BPA', ## Bonneville Power Administration
        235: 'CO', ## Public service company of CO
        272: 'WAPA-UME', ## Upper great plains
        273: 'WAPA-CM', ## Rocky mountain region
        274: 'WAPA-LC', ## Lower CO / desert southwest
        275: 'WAPA-UMW', ## Upper great plains
        252: 'SMEPA', ## South Mississippi Electric Power Assoc.
        164: 'Entergy',
        142: 'CLECO',
        196: 'LAGN', ##Louisiana Generating
        195: 'LEPA', ## Louisiana Energy & Power Authority
    }
    fercname2id = {**fercname2id, **{v:k for k,v in fercid_custom.items()}}

    ### Get the id
    fercid = fercname2id[ba]

    ### Get timezones
    dfeiaferc = pd.read_csv(
        datapath+'EIA/ElectricSystemOperatingData/EIA-BA_FERC-ID.csv',
        dtype={'respondent_id':'Int64','eia_code':'Int64'})
    fercid2timezone = dict(zip(
        dfeiaferc.dropna().respondent_id.values, 
        dfeiaferc.dropna().timezone.values))
    ### Add the extra timezones for MISO south: 
    ### SMEPA (252), Entergy (164), CLECO (142)
    for fid in [252, 164, 142]:
        fercid2timezone[fid] = -6
    tz = pvvm.toolbox.timezone_to_tz(fercid2timezone[fercid])
    
    ### Extract the data
    df = dfferc.loc[
        (dfferc.respondent_id == fercid)& (dfferc.report_yr==year)
    ].copy()
    ### Reshape to long format
    hourcols = ['hour{:02.0f}'.format(i) for i in range(1,25)]
    df = pd.melt(df, id_vars='lambda_date', value_vars=hourcols, 
                 var_name='hour', value_name='lambda')
    df = df.sort_values(['lambda_date','hour'])

    ### Remake the index
    try:
        df.index = pd.date_range(
            '{}-01-01'.format(year),'{}-01-01'.format(year+1),
            freq='H', closed='left', tz=tz)
        return df['lambda']
    except ValueError as err:
        print(err)
        warn('{}\nlen(df) = {}'.format(err, len(df)))
        if error == 'raise':
            raise ValueError(err)
        elif error == 'return':
            return df
        elif error in ['pass', 'ignore']:
            print('{} {}'.format(ba, year))
            pass

def getferc_demand(ba, year, dfferc=None, error='raise'):
    """
    """
    ### Load FERC dataframe if necessary
    if dfferc is None:
        dfferc = getferc('demand')
    elif type(dfferc) is pd.DataFrame:
        pass
    else:
        raise Exception("Invalid dfferc; get from getferc('demand')")

    ### Get converter for respondent_id
    dfferc_id = getferc('id')
    dfferc_id['respondent_name'] = (
        dfferc_id['respondent_name'].map(lambda x: x.strip()))
    fercid2name = dict(zip(
        *dfferc_id[['respondent_id','respondent_name']].T.values))
    fercname2id = dict(zip(
        *dfferc_id[['respondent_name','respondent_id']].T.values))
    ### Specify some extra ferc ids, from pvvm.io.getload_ferc()
    fercid_custom = {
        125: 'CAISO', 165: 'ERCOT', 321: 'MISO',
        230: 'PJM',   211: 'NYISO', 185: 'ISONE',
        116: 'APS',   210: 'NV',    228: 'PACE', 
        229: 'PACW',  232: 'PGE',   240: 'PSE', 
        122: 'BPA', ## Bonneville Power Administration
        235: 'CO', ## Public service company of CO
        272: 'WAPA-UME', ## Upper great plains
        273: 'WAPA-CM', ## Rocky mountain region
        274: 'WAPA-LC', ## Lower CO / desert southwest
        275: 'WAPA-UMW', ## Upper great plains
        252: 'SMEPA', ## South Mississippi Electric Power Assoc.
        164: 'Entergy',
        142: 'CLECO',
        196: 'LAGN', ##Louisiana Generating
        195: 'LEPA', ## Louisiana Energy & Power Authority
    }
    fercname2id = {**fercname2id, **{v:k for k,v in fercid_custom.items()}}

    ### Get the id
    fercid = fercname2id[ba]
    
    ### Get timezones
    dfeiaferc = pd.read_csv(
        datapath+'EIA/ElectricSystemOperatingData/EIA-BA_FERC-ID.csv',
        dtype={'respondent_id':'Int64','eia_code':'Int64'})
    fercid2timezone = dict(zip(
        dfeiaferc.dropna().respondent_id.values, 
        dfeiaferc.dropna().timezone.values))
    ### Add the extra timezones for MISO south: 
    ### SMEPA (252), Entergy (164), CLECO (142)
    for fid in [252, 164, 142]:
        fercid2timezone[fid] = -6
    tz = pvvm.toolbox.timezone_to_tz(fercid2timezone[fercid])
    
    ### Extract the data
    df = dfferc.loc[
        (dfferc.respondent_id == fercid)& (dfferc.report_yr==year)
    ].copy()
    ### Reshape to long format
    hourcols = ['hour{:02.0f}'.format(i) for i in range(1,25)]
    df = pd.melt(df, id_vars='plan_date', value_vars=hourcols, 
                 var_name='hour', value_name='load')
    df = df.sort_values(['plan_date','hour'])

    ### Remake the index
    try:
        df.index = pd.date_range(
            '{}-01-01'.format(year),'{}-01-01'.format(year+1),
            freq='H', closed='left', tz=tz)
        return df['load']
    except ValueError as err:
        print(err)
        warn('{}\nlen(df) = {}'.format(err, len(df)))
        if error == 'raise':
            raise ValueError(err)
        elif error == 'return':
            return df
        elif error in ['pass', 'ignore']:
            print('{} {}'.format(ba, year))
            pass



def get_netload(iso, year, net='load', resolution=60, units='MW'):
    """
    Notes
    -----
    Solar/PV counts both PV and CSP (models CSP as 1-axis-tracking PV)
    """
    unitconverter = {'W': 1E-6,'kW':1E-3,'MW':1,'GW':1000,'TW':1000000}
    inpath = {
        'load': revmpath+'io/iso-load_ferc/load-MWac-FERC2018-{}-{}.csv'.format(
            iso if iso != 'MISO' else 'MISO_SMEPA_Entergy_CLECO', 
            year),
        'pv': (revmpath+'io/iso-load_ferc-netsolar_eia860/60min/'
               +'netload-MWac-FERC2018_EIA860-{}-{}.csv'.format(
                iso if iso != 'MISO' else 'MISO_SMEPA_Entergy_CLECO',
                year)),
        'wind': (revmpath+'io/iso-load_ferc-netwind_iso/'
                 +'netload-MWac-FERC2018_ISO-{}-{}.csv'.format(
                    iso if iso != 'MISO' else 'MISO_SMEPA_Entergy_CLECO',
                    year)),
        'vre': (revmpath+'io/iso-load_ferc-netsolar_eia860-netwind_iso/'
                +'netload-MWac-FERC2018_EIA860_ISO-{}-{}.csv'.format(
                    iso if iso != 'MISO' else 'MISO_SMEPA_Entergy_CLECO',
                    year)),
    }
    inpath['demand'] = inpath['load']
    inpath[None] = inpath['load']
    inpath['solar'] = inpath['pv']
    df = pd.read_csv(
        inpath[net], header=None, parse_dates=True, names=[iso], squeeze=True,
    ).tz_convert(pvvm.toolbox.tz_iso[iso]) / unitconverter[units]
    
    return df

def getdflmp(iso, market, year, product='lmp'):
    """
    """
    dirmodule = os.path.dirname(os.path.abspath(__file__)) + '/'
    inpath = dirmodule + '../data/lmp/'

    infile = '{}{}-{}-{}-{}.gz'.format(
        inpath, iso.lower(), product, market, year)
    df = pd.read_csv(infile, index_col=0, parse_dates=True)
    # df = df.tz_localize('UTC').tz_convert(pvvm.toolbox.tz_iso[iso])
    df = df.tz_convert(pvvm.toolbox.tz_iso[iso])
    return df

##################################
### ISO historical wind generation

def getwindfile_ercot(year, filepath=None):
    '''
    Loads wind file directly as downloaded from ERCOT. 
    With year in range(2007,2017) as input, return dataframe with 
    columnsout = ['Load', 'WindCap', 'WindPower', 'WindCF', 'WindPen'] 
    of length equal to the number of hours in year.

    First download the hourly wind files from ERCOT from 
    'http://mis.ercot.com/misapp/GetReports.do?reportTypeId=13424&' +
    'reportTitle=Hourly%20Aggregated%20Wind%20Output&showHTMLView=&mimicKey'
    and save to filepath.

    # Try doing this as a class under ERCOT?

    Parameters
    ----------
    year: integer in range(2007,2017)
    filepath: 'default' = 'in/ERCOT/wind/'

    Returns
    -------
    dataframe:
    * index = pd.date_range(start='1/1/'+str(year), 
                            end='12/31/'+str(year)+' 23:00', freq='H')
    * columns = ['Load', 'WindCap', 'WindPower', 'WindCF', 'WindPen']
    '''
    if filepath is None:
        filepath = datapath+'ISO/ERCOT/wind/'

    files = {
        2007: 'rpt.00013424.0000000000000000.20141016.182537113.ERCOT_2007_Hourly_Wind_Output.xls',
        2008: 'rpt.00013424.0000000000000000.20141016.182537562.ERCOT_2008_Hourly_Wind_Output.xls',
        2009: 'rpt.00013424.0000000000000000.20141016.182537070.ERCOT_2009_Hourly_Wind_Output.xls',
        2010: 'rpt.00013424.0000000000000000.20141016.182537380.ERCOT_2010_Hourly_wind_Output.xls',
        2011: 'rpt.00013424.0000000000000000.20141016.182537685.ERCOT_2011_Hourly_Wind_Output.xls',
        2012: 'rpt.00013424.0000000000000000.20141016.182537195.ERCOT_2012_Hourly_Wind_Output.xlsx',
        2013: 'rpt.00013424.0000000000000000.20141016.182537359.ERCOT_2013_Hourly_Wind_Output.xls',
        2014: 'rpt.00013424.0000000000000000.20150212.131843687.ERCOT_2014_Hourly_Wind_Output.xlsx',
        2015: 'rpt.00013424.0000000000000000.ERCOT_2015_Hourly_Wind_Output.xlsx',
        2016: 'rpt.00013424.0000000000000000.20170112.104938392.ERCOT_2016_Hourly_Wind_Output.xlsx',
        2017: 'rpt.00013424.0000000000000000.20180131.170245804.ERCOT_2017_Hourly_Wind_Output.xlsx'
    }

    sheets = {
        2007: '2007',
        2008: '2008',
        2009: '2009',
        2010: '2010',
        2011: 'Sheet1',
        2012: 'numbers only',
        2013: '2013 numbers only',
        2014: 'numbers',
        2015: 'numbers',
        2016: 'numbers',
        2017: 'numbers'
    }

    columnsio = {
        (2007, 'Load'): 'ERCOT Load', 
        (2007, 'WindCap'): 'Wind Capacity Installed', 
        (2007, 'WindPower'): 'Hourly Wind Output', 
        (2007, 'WindCF'): '% Installed Wind Capacity', 
        (2007, 'WindPen'): 'Wind % of ERCOT Load', 

        (2008, 'Load'): 'ERCOT Load', 
        (2008, 'WindCap'): 'Installed Wind Capacity', 
        (2008, 'WindPower'): 'Hourly Wind Output', 
        (2008, 'WindCF'): '% Installed Wind Capacity', 
        (2008, 'WindPen'): 'Wind % of ERCOT Load', 

        (2009, 'Load'): 'ERCOT Load', 
        (2009, 'WindCap'): 'Installed Wind Capacity', 
        (2009, 'WindPower'): 'Hourly Wind Output', 
        (2009, 'WindCF'): '% Installed Wind Capacity', 
        (2009, 'WindPen'): 'Wind % of ERCOT Load', 

        (2010, 'Load'): 'ERCOT Load', 
        (2010, 'WindCap'): 'Installed Wind Capacity', 
        (2010, 'WindPower'): 'Hourly Wind Output', 
        (2010, 'WindCF'): '% Installed Wind Capacity', 
        (2010, 'WindPen'): 'Wind % of ERCOT Load', 

        (2011, 'Load'): 'ERCOT Load', 
        (2011, 'WindCap'): 'Installed Wind Capacity', 
        (2011, 'WindPower'): 'Total Wind Output at Hour', 
        (2011, 'WindCF'): '% Installed Wind Capacity', 
        (2011, 'WindPen'): 'Wind % of ERCOT Load', 

        (2012, 'Load'): 'ERCOT LOAD', 
        (2012, 'WindCap'): 'Total Wind Installed, MW', 
        (2012, 'WindPower'): 'Total Wind Output, MW', 
        (2012, 'WindCF'): 'Wind Output, % of Installed', 
        (2012, 'WindPen'): 'Wind Output, % of Load', 

        (2013, 'Load'): ' ERCOT LOAD', 
        (2013, 'WindCap'): 'Total Wind Installed, MW', 
        (2013, 'WindPower'): 'Total Wind Output, MW', 
        (2013, 'WindCF'): 'Wind Output, % of Installed', 
        (2013, 'WindPen'): 'Wind Output, % of Load', 

        (2014, 'Load'): 'ERCOT Load, MW', 
        (2014, 'WindCap'): 'Total Wind Installed, MW', 
        (2014, 'WindPower'): 'Total Wind Output, MW', 
        (2014, 'WindCF'): 'Wind Output, % of Installed', 
        (2014, 'WindPen'): 'Wind Output, % of Load', 

        (2015, 'Load'): 'ERCOT Load, MW', 
        (2015, 'WindCap'): 'Total Wind Installed, MW', 
        (2015, 'WindPower'): 'Total Wind Output, MW', 
        (2015, 'WindCF'): 'Wind Output, % of Installed', 
        (2015, 'WindPen'): 'Wind Output, % of Load', 

        (2016, 'Load'): 'ERCOT Load, MW', 
        (2016, 'WindCap'): 'Total Wind Installed, MW', 
        (2016, 'WindPower'): 'Total Wind Output, MW', 
        (2016, 'WindCF'): 'Wind Output, % of Installed', 
        (2016, 'WindPen'): 'Wind Output, % of Load',

        (2017, 'Load'): 'ERCOT Load, MW', 
        (2017, 'WindCap'): 'Total Wind Installed, MW', 
        (2017, 'WindPower'): 'Total Wind Output, MW', 
        (2017, 'WindCF'): 'Wind Output, % of Installed', 
        (2017, 'WindPen'): 'Wind Output, % of Load',        
    }

    columnsout = ['Load', 'WindCap', 'WindPower', 'WindCF', 'WindPen']

    rowstodrop = {
        2008: 0,
        2014: 8760,
        2015: 8760,
        2016: 8784
    }

    dfin = pd.read_excel(filepath+files[year], sheet_name=sheets[year], header=0)

    if year in rowstodrop.keys():
        dfin = dfin.drop(dfin.index[[rowstodrop[year]]])

    index = pd.date_range(start='1/1/'+str(year), end='12/31/'+str(year)+' 23:00', freq='H', 
                          tz='Etc/GMT+6')
    dfout = pd.DataFrame(columns=columnsout, index=index)

    for column in columnsout:
        dfout.loc[:,column] = dfin.loc[:,columnsio[(year, column)]].values

    return dfout

def getwindfile_miso(year, filepath=None):
    '''
    Loads wind file directly as downloaded from MISO. 

    First download the hourly wind files from MISO:
    * (2008-2014): ('https://www.misoenergy.org/markets-and-operations/'
    'real-time--market-data/market-reports/market-report-archives/'
    '#nt=%2FMarketReportType%3ASummary%2FMarketReportName%3AArchived%20'
    'Historical%20Hourly%20Wind%20Data%20%20(zip)&t=10&p=0'
    '&s=MarketReportPublished&sd=desc')
    * (2015-2018): ('https://www.misoenergy.org/markets-and-operations/'
    'real-time--market-data/market-reports/#nt=%2FMarketReportType%3ASummary'
    '%2FMarketReportName%3AHistorical%20Hourly%20Wind%20Data%20(csv)'
    '&t=10&p=0&s=MarketReportPublished&sd=desc')

    https://docs.misoenergy.org/marketreports/200812_hwd_HIST.zip
    https://docs.misoenergy.org/marketreports/200912_hwd_HIST.zip
    https://docs.misoenergy.org/marketreports/201012_hwd_HIST.zip
    https://docs.misoenergy.org/marketreports/201112_hwd_HIST.zip
    https://docs.misoenergy.org/marketreports/201212_hwd_HIST.zip
    https://docs.misoenergy.org/marketreports/201312_hwd_HIST.zip
    https://docs.misoenergy.org/marketreports/201412_hwd_HIST.zip
    https://docs.misoenergy.org/marketreports/20151231_hwd_hist.csv
    https://docs.misoenergy.org/marketreports/20161231_hwd_hist.csv
    https://docs.misoenergy.org/marketreports/20171231_hwd_HIST.csv
    https://docs.misoenergy.org/marketreports/20181231_hwd_HIST.csv

    Parameters
    ----------
    year: integer in range(2008,2019)
    filepath: 'default' = 'in/MISO/wind/'

    Returns
    -------
    dataframe:
    * columns = ['MWh']
    '''
    if filepath is None:
        filepath = datapath+'ISO/MISO/wind/'

    files = {
        2008: '2008_hwd_hist.csv',
        2009: '20091231_hwd_HIST.csv',
        2010: '20101231_hwd_HIST.csv',
        2011: '20111231_hwd_HIST.csv',
        2012: '20121231_hwd_HIST.csv',
        2013: '20131231_hwd_HIST.csv',
        2014: '20141231_hwd_HIST.csv',
        2015: '20151231_hwd_hist.csv',
        2016: '20161231_hwd_hist.csv',
        2017: '20171231_hwd_hist.csv',
        2018: '20181231_hwd_HIST.csv',
    }
    
    skipfooter = 1
    
    skiprows = {
        2008: 6,
        2009: [0,1,2,3,4,5,6,8],
        2010: [0,1,2,3,4,5,6,8],
        2011: [0,1,2,3,4,5,6,8],
        2012: 6,
        2013: 6,
        2014: 6,
        2015: 6,
        2016: 6,
        2017: 6,
        2018: 6,
    }

    dfin = pd.read_csv(
        filepath+files[year], 
        skiprows=skiprows[year], 
        skipfooter=skipfooter, engine='python',
        header=0,
    )
    
    dfin.index = pd.date_range(
        '{}-01-01'.format(year), '{}-01-01'.format(year+1),
        freq='H', closed='left', tz=pvvm.toolbox.tz_iso['MISO'],
    )

    return dfin['MWh']

def getwindfile_pjm(year, filepath=None):
    '''
    Loads wind file directly as downloaded from PJM. 
    Source: https://dataminer2.pjm.com/feed/wind_gen/definition

    Parameters
    ----------
    year: integer in range(2011,2018)
    filepath: 'default' = 'in/PJM/wind/'

    Returns
    -------
    dataframe:
    * columns = ['MWh']
    '''
    if filepath is None:
        filepath = datapath+'ISO/PJM/wind/'

    files = {i: '{}-hourly-wind.xls'.format(i)
             for i in range(2011,2019)}
    
    zones = {
        year: ['RTO','RFC','MIDATL','WEST'] if year < 2016
        else ['RTO','RFC','MIDATL','WEST','SOUTH']
        for year in range(2011,2018)}
    index = pd.date_range(
        '{}-01-01'.format(year), '{}-01-01'.format(year+1),
        freq='H', closed='left', tz=pvvm.toolbox.tz_iso['PJM'],
    )
    
    dictin = {}
    for zone in zones[year]:
        df = pd.read_excel(
            filepath+files[year],
            skiprows=1,
            sheet_name=zone,
            usecols='B,D:AA',
        ).melt(
            id_vars='DATE', var_name='hour', value_name=zone
        ).sort_values(['DATE','hour'])
        df.index = index
        dictin[zone] = df[zone]
    dfin = pd.concat(dictin, axis=1)

    return dfin

def getwindfile_isone(year, filepath=None):
    '''
    Loads wind file directly as downloaded from ISONE. 
    Source: 
    https://www.iso-ne.com/isoexpress/web/reports/operations/-/tree/daily-gen-fuel-type

    https://www.iso-ne.com/static-assets/documents/2014/11/hourly_wind_gen_2011_2014.xlsx
    https://www.iso-ne.com/static-assets/documents/2015/04/hourly_wind_gen_2015.xlsx
    https://www.iso-ne.com/static-assets/documents/2016/04/hourly_wind_gen_2016.xlsx
    https://www.iso-ne.com/static-assets/documents/2017/04/hourly_wind_gen_2017.xlsx
    https://www.iso-ne.com/static-assets/documents/2018/04/hourly_wind_gen_2018.xlsx

    Parameters
    ----------
    year: integer in range(2011,2018)
    filepath: 'default' = 'in/ISONE/wind/'

    Returns
    -------
    dataframe:
    * columns = ['MWh']
    '''
    if filepath is None:
        filepath = datapath+'ISO/ISONE/wind/'

    files = {i: 'hourly_wind_gen_2011_2014.xlsx'
             if i in [2011,2012,2013,2014]
             else 'hourly_wind_gen_{}.xlsx'.format(i)
             for i in range(2011,2019)}
    
    skipfooter = {
        2011: 0,
        2012: 0,
        2013: 0,
        2014: 0,
        2015: 0,
        2016: 0,
        2017: 1,
        2018: 0,
    }
    
    skiprows = {
        2011: None,
        2012: None,
        2013: None,
        2014: None,
        2015: None,
        2016: None,
        2017: [0],
        2018: [0]
    }
    
    df = pd.read_excel(
        filepath+files[year],
        skiprows=skiprows[year],
        skipfooter=skipfooter[year],
        sheet_name='HourlyData',
    ).rename(columns={'LOCAL_DAY':'local_day',
                      'LOCAL_HOUR_END':'local_hour_end',
                      'METERED_MW':'tot_wind_mwh',})
    df = df.loc[df.year==year].copy()
    
    ###### Fix time series axis
    ### Get DST dates
    springforward = pd.Timestamp(pvvm.toolbox.dst_springforward[year])
    fallback = pd.Timestamp(pvvm.toolbox.dst_fallback[year])

    ### Functions for deDSTification
    def getdate(row, hourshift=0):
        out = pd.Timestamp(
                year=row.year,
                month=row.local_day.month,
                day=row.local_day.day,
                hour=int(row.local_hour_end)+hourshift,
            )
        return out

    def undstify(row):
        ### Pre-DST
        if row.local_day < springforward:
            out = getdate(row, hourshift=-1)
        ### Post-DST
        elif row.local_day > fallback:
            out = getdate(row, hourshift=-1)
        ### DST
        elif ((row.local_day > springforward) 
              and (row.local_day < fallback)):
            out = (getdate(row, hourshift=-1) 
                   - pd.Timedelta('1H'))
        ### Springforward
        elif row.local_day == springforward:
            if row.local_hour_end < '02':
                out = getdate(row, hourshift=-1)
            else:
                out = (getdate(row, hourshift=-1) 
                   - pd.Timedelta('1H'))
        ### Fallback
        elif row.local_day == fallback:
            if row.local_hour_end < '02X':
                out = (getdate(row, hourshift=-1) 
                       - pd.Timedelta('1H'))
            elif row.local_hour_end == '02X':
                out = pd.Timestamp(
                    year=row.year,
                    month=row.local_day.month,
                    day=row.local_day.day,
                    hour=2-1, # to shift to hour-beginning
                )
            else:
                out = getdate(row, hourshift=-1)
        else:
            print(row)
            raise Exception('Messed up row')
        return out
    
    ### Apply deDST functions
    df.index = df.apply(undstify, axis=1)
    df = df.tz_localize(pvvm.toolbox.tz_iso['ISONE'])
    
    ### Create full timeseries axis
    index = pd.date_range(
        '{}-01-01'.format(year), '{}-01-01'.format(year+1),
        freq='H', closed='left', tz=pvvm.toolbox.tz_iso['ISONE'],
    )
    index = pd.DataFrame(index=index)
    dfout = df.merge(index, left_index=True, right_index=True, how='right')
    
    ### Interpolate over missing values
    dfout.tot_wind_mwh = dfout.tot_wind_mwh.interpolate()
    
    return dfout

def getwindfile_caiso(year, filepath=None):
    """
    Loads CAISO generation mix and saves wind timeseries.
    Fills missing values with monthly mean.
    """
    ### Get raw data
    if filepath is None:
        filepath = revmpath+'CAISO/io/caiso-generation-mix-raw.csv'
    if not os.path.exists(filepath):
        getcaisogenmix().to_csv(filepath)
    dfin = pd.read_csv(
        filepath,
        index_col='DateTimeNoDST',
        parse_dates=True
    ).tz_localize(pvvm.toolbox.tz_iso['CAISO'])

    ### Drop nulls and zero rows
    dfin = dfin.dropna(thresh=10)
    dfin = dfin.loc[
        ~((dfin.Hydro == 0.)
        & (dfin.Imports == 0.)
        & (dfin.Renewables == 0.)
        & (dfin.Nuclear == 0.)
        & (dfin.Thermal == 0.))
    ].copy()
    dfin = dfin.loc[dfin.Geothermal != 0.].copy()
    dfin = dfin.loc[dfin.Imports != 0.].copy()
    ### Drop duplicates
    dfin = dfin.loc[~dfin.index.duplicated(keep='last')].copy()

    ### Create full datetime axis
    index = pd.date_range(
        '{}-01-01'.format(year), '{}-01-01'.format(year+1),
        freq='H', closed='left', tz=pvvm.toolbox.tz_iso['CAISO'],
    )
    index = pd.DataFrame(index=index)
    ### Select individual year
    dfyear = dfin.merge(
        index, left_index=True, right_index=True, how='right')

    ###### Fill nan's with monthly mean
    ### NOTE: This is only partially realistic for wind;
    ### not at all defensible for solar
    ### Get monthly means
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    months = dict(zip(range(1,13), months))
    monthlywindmeans = {
        i: dfyear.loc['{} {}'.format(months[i], year), 'Wind'].mean()
        for i in months
    }
    ### Fill voids with monthly mean
    dfwind = dfyear.apply(
        lambda row: row.Wind if ~np.isnan(row['Wind']) else monthlywindmeans[row.name.month],
        axis=1
    )

    return dfwind

############
### GLUE ###

def glue_iso_columns(iso):
    """
    returns (nsrdbindex, lmpindex, pnodeid, latlonindex, pnodename)
    """
    if iso.lower() == 'pjm':
        nsrdbindex = 'latlonindex'
        lmpindex = 'node'
        pnodeid = 'node'
        latlonindex = 'latlonindex'
        extra_key = 'latlonindex'
        pnodename = 'nodename'
    elif iso.lower() == 'caiso':
        nsrdbindex = 'node'
        lmpindex = 'node'
        pnodeid = 'node'
        latlonindex = 'node'
        pnodename = 'node'
    elif iso.lower() == 'miso':
        nsrdbindex = 'node'
        lmpindex = 'node'
        pnodeid = 'node'
        latlonindex = 'node'
        extra_key = 'node'
        pnodename = 'node'
    elif iso.lower() == 'ercot':
        nsrdbindex = 'latlonindex'
        lmpindex = 'node'
        pnodeid = 'node'
        latlonindex = 'latlonindex'
        extra_key = 'latlonindex'
        pnodename = 'node'
    elif iso.lower() == 'isone':
        nsrdbindex = 'node'
        lmpindex = 'node'
        pnodeid = 'node'
        latlonindex = 'node'
        extra_key = 'node'
        pnodename = 'node'
    elif iso.lower() == 'nyiso':
        nsrdbindex = 'latlonindex'
        lmpindex = 'node'
        pnodeid = 'node'
        latlonindex = 'latlonindex'
        extra_key = 'latlonindex'
        pnodename = 'node'

    return (nsrdbindex, lmpindex, pnodeid, latlonindex, pnodename)

################
### IO FILES ###

def get_iso_nodes(iso, market='da', yearlmp=None, 
    merge=False, fulltimeonly=False):
    """
    Inputs
    ------
    * fullhourslist: 'default' or dict(zip(isos, absolutefilepaths))
    * nodelist: 'default' or dict(zip(isos, absolutefilepaths))

    Returns
    -------
    * tuple of (fulltime, nodemap)

    Usage
    -----
    dfin = nodemap.merge(fulltime, on=pnodeid)
    """
    ### Align labels and create filenames
    pnodeid = glue_iso_columns(iso)[2]

    fulltimefile = revmpath + '{}/io/fulltimenodes/{}-{}lmp-fulltime-{}.csv'.format(
        iso.upper(), iso.lower(), market, yearlmp)

    nodemapfile = revmpath + '{}/io/{}-node-latlon.csv'.format(
        iso.upper(), iso.lower())

    ### Load and return dataframes
    if yearlmp is not None:
        fulltime = pd.read_csv(
            fulltimefile, header=None, names=[pnodeid])
        if fulltimeonly == True:
            return fulltime

    nodemap = pd.read_csv(nodemapfile)

    ### Return results
    if yearlmp is None:
        return nodemap
    elif (yearlmp is not None) and (merge == False):
        return fulltime, nodemap
    elif merge == True:
        dfout = nodemap.merge(fulltime, on=pnodeid)
        return dfout

##########################
### LOAD MODEL RESULTS ###

def getresults(market='da', orientation='default', carbon=True, 
    optversion=3, dropcanada=True, inflation=True, dollaryear=2017,
    inpath='out', defaultversion=7,):
    """
    Inputs
    ------
    * orientation: in ['default', 'def'] or in ['optimized', 'opt']

    """
    ###### Load files
    if orientation in ['default', 'def']:
        def fileparams(infile, returndict=True):
            isodict = {'C': 'CAISO', 'E': 'ERCOT', 'M': 'MISO', 
                       'P': 'PJM', 'N': 'NYISO', 'I': 'ISONE'}
            file = os.path.basename(infile)
            filelist = file.split('-')
            ### Parse filename components
            ## PVvalueV6-CEMPNI-da-2015lmp-2015sun-track-107ILR-0m-0d-0cutoff-0.csv
            program = filelist[0]
            isolist = [isodict[letter] for letter in filelist[1]]
            market = filelist[2]
            yearlmp = int(filelist[3][:-3])
            yearsun= filelist[4][:-3]
            if yearsun != 'tmy':
                yearsun = int(yearsun)
            systemtype = filelist[5]
            ilr = float(filelist[6][:-3])
            pricecutoff = filelist[9][:filelist[9].find('cutoff')]
            modifier = '-'.join(filelist[10:])
            modifier = modifier[:modifier.find('.')]
            if returndict:
                outdict = dict(zip(
                    ['program', 'isolist', 'market', 'yearlmp', 'yearsun', 
                     'systemtype', 'ilr', 'pricecutoff', 'modifier'],
                    [program, isolist, market, yearlmp, yearsun, 
                     systemtype, ilr, pricecutoff, modifier]
                ))
                return outdict
            return (program, isolist, market, yearlmp, yearsun, 
                    systemtype, ilr, pricecutoff, modifier)

        ### Get the file list
        infiles = glob('{}USA/{}/PVvalueV7-*-{}-*107ILR-0m-0d-0cutoff-0.csv'.format(
            revmpath, inpath, market))

    elif orientation in ['optimized', 'opt']:
        def fileparams(infile, optversion=optversion, returndict=True):
            isodict = {'C': 'CAISO', 'E': 'ERCOT', 'M': 'MISO', 
                       'P': 'PJM', 'N': 'NYISO', 'I': 'ISONE'}
            file = os.path.basename(infile)
            filelist = file.split('-')
            ### Parse filename components
            ### PVvalueOptV3-C-da-2015lmp-tmysun-track-107ILR-0cutoff-1000_2000-0.csv
            program = filelist[0]
        #     iso = isodict[filelist[1]]
            isolist = [isodict[letter] for letter in filelist[1]]
            market = filelist[2]
            yearlmp = int(filelist[3][:-3])
            yearsun = filelist[4][:-3]
            if yearsun != 'tmy':
                yearsun = int(yearsun)
            systemtype = filelist[5]
            ilr = float(filelist[6][:-3])

            pricecutoff = filelist[7][:filelist[7].find('cutoff')]
            modifier = filelist[9]
            modifier = modifier[:modifier.find('.')]
            if returndict:
                outdict = dict(zip(
                    ['program', 'isolist', 'market', 'yearlmp', 'yearsun', 
                    'systemtype', 'ilr', 'pricecutoff', 'modifier'],
                    [program, isolist, market, yearlmp, yearsun, 
                    systemtype, ilr, pricecutoff, modifier]
                ))
                return outdict
            return (program, isolist, market, yearlmp, yearsun, 
                    systemtype, ilr, pricecutoff, modifier)

        ### Get the file list
        infiles = glob('{}USA/{}/PVvalueOptV3-*-{}-*.csv'.format(
            revmpath, inpath, market))

    ### Load the files
    dictin = {}
    for i in range(len(infiles)):
        df = pd.read_csv(infiles[i])
        params = fileparams(infiles[i])
        df['market'] = params['market']
        df['systemtype'] = params['systemtype']
        df['yearlmp'] = params['yearlmp']
        df['yearsun'] = params['yearsun']
        try:
            df['pricecutoff'] = params['pricecutoff']
        except KeyError:
            df['pricecutoff'] = None
        df['modifier'] = params['modifier']
        dictin[i] = df
    dfin = (pd.concat(dictin, axis=0, copy=False)
            .drop_duplicates(subset=[
                'ISO:Node', 'market', 'systemtype', 
                'yearlmp', 'yearsun', 'pricecutoff', 'modifier'])
            .reset_index(drop=True)
           )

    ###### Add additional value parameters for optimized orientation
    if orientation in ['optimized', 'opt']:
        ### Extra value parameters
        def yearhourify(x):
            if x == 'tmy':
                return 8760
            else:
                return yearhours(x)
            
        dfin['OptCF_Value']  = (dfin['OptCF_Rev']   / dfin['OptCF_CF']
                                / dfin.yearlmp.map(lambda x: yearhourify(x)) * 1000)
        dfin['OptCF_VF']     = dfin['OptCF_Value']  / dfin['Default_Price']

        dfin['OptRev_Value'] = (dfin['OptRev_Rev']  / dfin['OptRev_CF']
                                / dfin.yearlmp.map(lambda x: yearhourify(x)) * 1000)
        dfin['OptRev_VF']    = dfin['OptRev_Value'] / dfin['Default_Price']

        dfin['OptRev_Rev/Default_Revenue'] = dfin['OptRev_Rev']   / dfin['Default_Revenue']
        dfin['OptCF_CF/Default_CF']        = dfin['OptCF_CF']     / dfin['Default_CF']
        dfin['OptRev_Value/Default_Value'] = dfin['OptRev_Value'] / dfin['Default_Value']
        dfin['OptRev_VF/Default_VF']       = dfin['OptRev_VF']    / dfin['Default_VF']

        dfin['OptRev_Rev/OptCF_Rev']     = dfin['OptRev_Rev']   / dfin['OptCF_Rev']
        dfin['OptRev_Value/OptCF_Value'] = dfin['OptRev_Value'] / dfin['OptCF_Value']
        dfin['OptRev_VF/OptCF_VF']       = dfin['OptRev_VF']    / dfin['OptCF_VF']

        dfin['OptCF_Tilt-Latitude'] = dfin['OptCF_Tilt'] - dfin['Latitude']
        dfin['OptRev_Tilt-Latitude'] = dfin['OptRev_Tilt'] - dfin['Latitude']


    ###### Incorporate carbon displacement
    if (carbon == True) and (orientation == 'default'):
        ### Assuming TMY profile for all years
        carbon_fixed = pd.read_csv(
            os.path.join(
                revmpath, 
                'USA/io/PVCO2displace_Callaway2018-CEMPNI-tmysun-fixed-latitudetilt-107ILR-0.csv')
        ).drop_duplicates(subset=['ISO:Node'])
        carbon_track = pd.read_csv(
            os.path.join(
                revmpath, 
                'USA/io/PVCO2displace_Callaway2018-CEMPNI-tmysun-track-0tilt-107ILR-0.csv')
        ).drop_duplicates(subset=['ISO:Node'])

        ### Make dictionary converters
        ## fixed
        foo = tuple(zip(carbon_fixed['ISO:Node'].values, 
                        ['fixed']*len(carbon_fixed)))
        dictcarbonfixed = dict(zip(foo, carbon_fixed['CO2'].values))

        ## track
        foo = tuple(zip(carbon_track['ISO:Node'].values, 
                        ['track']*len(carbon_track)))
        dictcarbontrack = dict(zip(foo, carbon_track['CO2'].values))

        ## combined
        dictcarbon = {**dictcarbonfixed, **dictcarbontrack}

        dfin['CO2Tons'] = dfin.apply(
            lambda row: dictcarbon[row['ISO:Node'], row['systemtype']], axis=1)


    ### Drop the two MISO nodes in Canada
    if dropcanada:
        dfin.drop(
            dfin[dfin['ISO:Node'].map(lambda x: x in ['MISO:MHEB', 'MISO:SPC'])].index,
            inplace=True)


    ###### Differentiate the CAISO nodes
    dfcaisonodes = get_iso_nodes('CAISO', 'da')[1]
    dfcaisonodes['ISO:Node'] = dfcaisonodes.node.map(
        lambda x: 'CAISO:{}'.format(x))

    dfin = dfin.merge(
        dfcaisonodes[['ISO:Node','area']], on='ISO:Node', how='outer'
    ).dropna(subset=['Latitude']).copy()

    ### Add column with WECCified ISO
    def WECCify(row):
        if (row['ISO'] == 'CAISO') and (row['area'] == 'CA'):
            return 'CAISO'
        elif (row['ISO'] == 'CAISO') and (row['area'] != 'CA'):
            return 'WECC -CAISO'
        else:
            return row['ISO']

    dfin['ISOwecc'] = dfin.apply(WECCify, axis=1)

    ### Fix a few other things
    dfin.yearlmp = dfin.yearlmp.map(int)

    ###### Incorporate inflation
    if inflation:
        inflatifier = {year: inflate(yearin=year, yearout=dollaryear) 
                       for year in dfin['yearlmp'].unique()}
        inflatifier = dfin['yearlmp'].map(lambda x: inflatifier[x])

        if orientation in ['default', 'def']:
            colstoinflate = ['PriceAverage', 'Revenue', 'Revenue_dispatched', 
                             'ValueAverage', 'ValueAverage_dispatched']
        else:
            colstoinflate = ['Default_Price', 'Default_Value', 'Default_Revenue',
                             'OptCF_Rev', 'OptRev_Rev']

        for col in colstoinflate:
            dfin[col] = inflatifier * dfin[col]
        
    return dfin


######################
### EPA CEMS AMPD data

# def getampd(state, year, inpath=None):
#     if inpath is None:
#         inpath = datapath+'EPA/AMPD/emissions/hourly/'
#     dictin = {}
#     for month in range(1,13):
#         dictin[month] = pd.read_csv(
#             inpath+'{}{}{:02}.zip'.format(year, state.lower(), month)
#         )
#     dfin = pd.concat(dictin, axis=0, ignore_index=True)
#     return dfin

def getampd(state, year, inpath=None, timestamp=False):
    columnrenamer = {
        'GLOAD':           'GLOAD (MW)',
        'CO2_MASS':        'CO2_MASS (tons)',
        'CO2_MASS (TONS)': 'CO2_MASS (tons)',
        'NOX_MASS':        'NOX_MASS (lbs)',
        'NOX_MASS (LBS)':  'NOX_MASS (lbs)',
        'SO2_MASS':        'SO2_MASS (lbs)',
        'SO2_MASS (LBS)':  'SO2_MASS (lbs)',
        'HEAT_INPUT':      'HEAT_INPUT (mmBtu)',
        'CO2_RATE':        'CO2_RATE (tons/mmBtu)',
        'SO2_RATE':        'SO2_RATE (lbs/mmBtu)',
        'NOX_RATE':        'NOX_RATE (lbs/mmBtu)',
    }
    if inpath is None:
        inpath = datapath+'EPA/AMPD/emissions/hourly/'
    dictin = {}
    for month in range(1,13):
        dictin[month] = pd.read_csv(
            inpath+'{}{}{:02}.zip'.format(year, state.lower(), month),
            low_memory=False
        ).rename(columns=columnrenamer)
    dfin = pd.concat(dictin, axis=0, ignore_index=True, sort=False)
    
    if timestamp == True:
        dfin['timestamp'] = (
            dfin['OP_DATE'] + ' ' + dfin['OP_HOUR'].astype(str) + ':00'
        ).map(lambda x: pd.Timestamp(x))
    
    dfin['orispl__unitid'] = (
        dfin['ORISPL_CODE'].astype(str) + '__' + dfin['UNITID'].astype(str))
    
    return dfin

def get_ampd_pm_timeseries(
    state, pm='GT', dfgen=None, datum='gload'):
    """
    datum: str in ['gload', 'co2_mass']
    """
    column = {
        'gload': 'GLOAD (MW)', 'co2_mass': 'CO2_MASS (tons)',
        'so2_mass': 'SO2_MASS (lbs)', 'nox_mass': 'NOX_MASS (lbs)',
    }
    if dfgen is None:
        dfgen = geteia('generator')
        
    fullseries = pd.date_range(
        '1998-01-01 00:00', '2019-01-01 00:00', 
        freq='H', closed='left')
    fullindex = pd.DataFrame(index=fullseries)
    
    ### Load AMPD data
    dfepa = {}
    for year in trange(1998,2019, leave=True, desc=state):
        try:
            dfepa[year] = getampd(state, year, timestamp=True)
        except FileNotFoundError:
            print('Missing: {} {}'.format(state, year))
    dfepa = (pd.concat(dfepa, axis=0, sort=False).reset_index()
             .rename(columns={'level_0':'year'}))
    
    ### Get list of all plant codes with listed prime mover
    codes_pm = dfgen.loc[dfgen['Prime Mover'] == pm, 
                         'Plant Code'].unique().tolist()
    
    ### Get plant codes and unit codes for specified prime mover
    codes_epa = dfepa.ORISPL_CODE.unique().tolist()

    dfpm = dfepa.loc[dfepa.ORISPL_CODE.isin(codes_pm)].copy()
    codes_state_pm = [code for code in codes_epa if code in codes_pm]

    units_pm = {}
    for code in codes_state_pm:
        units_pm[code] = dfpm.loc[dfpm.ORISPL_CODE==code,'UNITID'].unique().tolist()
    
    peak_outputs_byunit = dfpm.groupby('UNIT_ID')[column[datum]].max().to_dict()
    
    dfpm_generation = dfpm.pivot(columns='orispl__unitid', index='timestamp', values=column[datum])
    dfout = fullindex.merge(dfpm_generation, how='left', left_index=True, right_index=True)
    
    return dfout

