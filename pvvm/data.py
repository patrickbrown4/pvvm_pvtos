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
import sys, os, site, zipfile, math, time, json, io
import googlemaps, urllib, shapely, shutil, requests
import xml.etree.ElementTree as ET
from glob import glob
from urllib.error import HTTPError
from urllib.request import URLError
from http.client import IncompleteRead
from zipfile import BadZipFile
from tqdm import tqdm, trange
from warnings import warn

###########################
### IMPORT PROJECT PATH ###
import pvvm.settings
revmpath = pvvm.settings.revmpath
datapath = pvvm.settings.datapath
apikeys = pvvm.settings.apikeys
nsrdbparams = pvvm.settings.nsrdbparams

#####################
### Imports from pvvm
import pvvm.toolbox
import pvvm.io


#######################
### DICTS AND LISTS ###
#######################

isos = ['CAISO', 'ERCOT', 'MISO', 'PJM', 'NYISO', 'ISONE']

resolutionlmps = {
    ('CAISO', 'da'): 60, ('CAISO', 'rt'): 5,
    ('ERCOT', 'da'): 60, ('ERCOT', 'rt'): 5,
    ('MISO',  'da'): 60, ('MISO',  'rt'): 60,
    ('PJM',   'da'): 60, ('PJM',   'rt'): 60,
    ('NYISO', 'da'): 60, ('NYISO', 'rt'): 5,
    ('ISONE', 'da'): 60, ('ISONE', 'rt'): 60,
}

################
### DOWNLOAD ###
################

###############
### General use

def constructpayload(**kwargs):
    out = []
    for kwarg in kwargs:
        out.append('{}={}'.format(kwarg, kwargs[kwarg]))
    stringout = '&'.join(out)
    return stringout

def constructquery(urlstart, **kwargs):
    out = '{}{}'.format(urlstart, constructpayload(**kwargs))
    return out

def stampify(date, interval=pd.Timedelta('1H')):
    datetime = pd.Timestamp(date)
    if interval == pd.Timedelta('1H'):
        dateout = '{}{:02}{:02}T{:02}'.format(
            datetime.year, datetime.month, 
            datetime.day, datetime.hour)
    elif interval == pd.Timedelta('1D'):
        dateout = '{}{:02}{:02}'.format(
            datetime.year, datetime.month, 
            datetime.day)
    return dateout

def download_file_series(urlstart, urlend, fileseries, filepath, 
    overwrite=False, sleeptime=60, numattempts=200, seriesname=True):
    """
    Example
    -------
    You want to download a list of files at urls = [
        'http://www.test.com/foo001.csv', 'http://www.test.com/foo002.csv'].
    Then:
        urlstart = 'http://www.test.com/foo'
        urlend = '.csv'
        fileseries = ['001', '002']
    If you want the files to be named 'foo001.csv', use seriesname=False
    If you want the files to be named '001.csv', use seriesname=True
    """
    filepath = pvvm.toolbox.pathify(filepath, make=True)

    ### Make lists of urls, files to download, and filenames
    urls = [(urlstart + file + urlend) for file in fileseries]
    todownload = [os.path.basename(url) for url in urls]

    if seriesname == True:
        filenames = [os.path.basename(file) + urlend for file in fileseries]
    else:
        filenames = todownload

    ### Get the list of downloaded files
    downloaded = [os.path.basename(file) for file in glob(filepath + '*')]

    ### Remake the list if overwrite == False
    if overwrite == False:
        filestodownload = []
        urlstodownload = []
        fileseriesnames = []
        for i in range(len(filenames)):
            if filenames[i] not in downloaded:
                filestodownload.append(todownload[i])
                urlstodownload.append(urls[i])
                fileseriesnames.append(filenames[i])
    elif overwrite == True:
        filestodownload = todownload
        urlstodownload = urls
        fileseriesnames = filenames

    ### Download the files
    for i in trange(len(urlstodownload)):
        ### Attempt the download
        attempts = 0
        while attempts < numattempts:
            try:
                urllib.request.urlretrieve(
                    urlstodownload[i], filepath + fileseriesnames[i])
                break
            except (HTTPError, IncompleteRead, EOFError) as err:
                print(urlstodownload[i])
                print(filestodownload[i])
                print('Rebuffed on attempt # {} at {} by "{}".'
                      'Will retry in {} seconds.'.format(
                        attempts, pvvm.toolbox.nowtime(), err, sleeptime))
                attempts += 1
                time.sleep(sleeptime)


###########################
### Geographic manipulation

def rowlatlon2x(row):
    latrad = row['latitude'] * math.pi / 180
    lonrad = row['longitude'] * math.pi / 180
    x = math.cos(latrad) * math.cos(lonrad)
    return x

def rowlatlon2y(row):
    latrad = row['latitude'] * math.pi / 180
    lonrad = row['longitude'] * math.pi / 180
    y = math.cos(latrad) * math.sin(lonrad)
    return y

def rowlatlon2z(row):
    latrad = row['latitude'] * math.pi / 180
    z = math.sin(latrad)
    return z


############
### ISO LMPs

"""
Note: These scripts worked as of early 2018, but MISO, PJM, and NYISO have since
changed their websites, and CAISO has removed data prior to 20150303. Scripts
are included here for documentary purposes and as a resource for future 
data collection, but are unlikely to work given ISO website changes.
"""

def download_caiso_lmp_allnodes(market, start, filepathout, 
    product='LMP', numattempts=200, waittime=10):

    urlstart = 'http://oasis.caiso.com/oasisapi/GroupZip?'
    columnsout = [
        'INTERVALSTARTTIME_GMT', 'NODE', 'MW',
        'OPR_DT', 'OPR_HR', 'OPR_INTERVAL']

    if market in ['RTM', 'HASP', 'RTPD']:
        interval = pd.Timedelta('1H')
    elif market in ['DAM', 'RUC']:
        interval = pd.Timedelta('1D')

    starttimestamp = pd.Timestamp(start)
    endtimestamp = starttimestamp + interval

    startdatetime = '{}{:02}{:02}T{:02}:00-0000'.format(
        starttimestamp.year, starttimestamp.month, 
        starttimestamp.day, starttimestamp.hour)
    enddatetime = '{}{:02}{:02}T{:02}:00-0000'.format(
        endtimestamp.year, endtimestamp.month, 
        endtimestamp.day, endtimestamp.hour)

    if interval == pd.Timedelta('1D'):
        fileout = '{}{:02}{:02}.gz'.format(
            starttimestamp.year, starttimestamp.month, 
            starttimestamp.day)
    elif interval == pd.Timedelta('1H'):
        fileout = '{}{:02}{:02}T{:02}.gz'.format(
            starttimestamp.year, starttimestamp.month, 
            starttimestamp.day, starttimestamp.hour)

    url = constructquery(
        urlstart, 
        groupid='{}_LMP_GRP'.format(market),
        startdatetime=startdatetime,
        enddatetime=enddatetime,
        version=1,
        resultformat=6)

    attempts = 0
    while attempts < numattempts:
        try:
            # if product.lower() in ['mcc', 'mce', 'mcl']:
            # if (market.upper() in ['DAM', 'RUC']) and (starttimestamp.year >= 2016):
            # if market.upper() in ['DAM', 'RUC']:
            if ((product.lower() in ['mcc', 'mce', 'mcl'])
                or ((market == 'DAM') and product.lower() == 'lmp')):
                zip_file = zipfile.ZipFile(io.BytesIO(
                    urllib.request.urlopen(url).read()))
                for csv_file in zip_file.infolist():
                    if csv_file.filename.endswith(
                        '{}_v1.csv'.format(product.upper())):
                        df = pd.read_csv(zip_file.open(csv_file.filename))
            else:
                df = pd.read_csv(url, compression='zip')

            dfout = df[df['LMP_TYPE'] == product.upper()][columnsout]

            dfout.to_csv(
                '{}{}'.format(filepathout, fileout),
                columns=columnsout,
                index=False,
                compression='gzip')
            
            return dfout
        except (
            URLError, IncompleteRead, pd.errors.ParserError, 
            BadZipFile, KeyError, HTTPError, UnboundLocalError) as error:
            print(
                'Error for {} on attempt {}/{}: {}'.format(
                    start, attempts, numattempts, error),
                # end='\r',
                )
            attempts += 1
            time.sleep(waittime)
    
            if attempts >= numattempts:
                raise URLError('{}{}'.format(filepathout, fileout))

def download_lmps(year, iso, market, overwrite=False, sleeptime=60,
    product='LMP', submarket=None, numattempts=200, subset=None,
    waittime=10, filepath=None):
    """
    Inputs
    ------
    subset: None or slice()
    
    Notes
    -----
    * ERCOT LMPs more than 30 days old must be requested from ERCOT. 
    Requests can be filed at http://www.ercot.com/about/contact/inforequest.
    Files should be placed in the folder 
    revmpath + 'ERCOT/in/lmp/{}/{}/'.format(market, year)
    where year is the year of the timestamp within the files.
    Note that the date in the filename for day-ahead LMPs is the date before
    the timestamps within the file: for example, file 
    ('cdr.00012328.0000000000000000.20151231.125905514.DAMHRLMPNP4183_csv')
    contains timestamps for 20160101, and should be placed in the 2016 folder.
    """
    ### Normalize inputs
    iso = iso.upper()
    market = market.lower()
    year = int(year)
    assert market in ['da', 'rt']
    assert iso in ['CAISO', 'MISO', 'PJM', 'NYISO', 'ISONE']

    ### Set file structure
    if filepath is None:
        filepath = revmpath+'{}/in/lmp/{}/'.format(iso, market)
    if not os.path.exists(filepath): os.makedirs(filepath)

    ### Adjust inputs for different isos
    urlstart = {
        'ISONE': {
            'da': 'https://www.iso-ne.com/static-transform/csv/histRpts/da-lmp/WW_DALMP_ISO_',
            'rt': 'https://www.iso-ne.com/static-transform/csv/histRpts/rt-lmp/lmp_rt_final_'},
        'MISO': {
            # 'da': 'https://old.misoenergy.org/Library/Repository/Market%20Reports/',
            # 'rt': 'https://old.misoenergy.org/Library/Repository/Market%20Reports/',
            'da': 'https://docs.misoenergy.org/marketreports/',
            'rt': 'https://docs.misoenergy.org/marketreports/',
        },
        'PJM': {
            'da': 'http://www.pjm.com/pub/account/lmpda/',
            'rt': 'http://www.pjm.com/pub/account/lmp/'},
        'NYISO': {
            'da': 'http://mis.nyiso.com/public/csv/damlbmp/',
            'rt': 'http://mis.nyiso.com/public/csv/realtime/'},
    }

    urlend = {
        'ISONE': {'da': '.csv', 'rt': '.csv'},
        'MISO': {'da': '_da_lmp.csv', 'rt': '_rt_lmp_final.csv'},
        'PJM': {'da': '-da.zip', 'rt': '.zip'},
        'NYISO': {'da': 'damlbmp_gen_csv.zip', 'rt': 'realtime_gen_csv.zip'},
    }

    files = {
        'ISONE': pvvm.toolbox.makedays(year),
        'MISO': pvvm.toolbox.makedays(year),
        'PJM': pvvm.toolbox.makedays(year),
        'NYISO': ['{}{:02}01'.format(year, month) for month in range(1,13)]
    }

    ### Download files
    if iso == 'ISONE':
        download_file_series(
            urlstart=urlstart[iso][market], urlend=urlend[iso][market],
            fileseries=files[iso], filepath=filepath, 
            overwrite=overwrite, sleeptime=sleeptime, numattempts=numattempts)

    elif iso == 'MISO':
        urls = [(urlstart[iso][market] + file + '_da_expost_lmp.csv')
                if (int(file) >= 20150301) and (market == 'da')
                else (urlstart[iso][market] + file + urlend[iso][market])
                for file in files[iso]]
        download_file_series(
            urlstart='', urlend='', fileseries=urls, filepath=filepath, 
            overwrite=overwrite, sleeptime=sleeptime, numattempts=numattempts)

    elif iso == 'PJM':
        da_updated = {
            '20151201': '-da_updated.zip',
            '20150930': '-da_updated.zip',
            '20140617': '-da_updated.zip',
            '20150616': '-da_updated.zip',
            '20150615': '-da_updated.zip',
            '20150614': '-da_updated.zip',
            '20140613': '-da_updated.zip',
            '20150603': '-da_updated.zip',
            '20150602': '-da_updated.zip',
            '20150601': '-da_updated.zip',
            '20150409': '-da_updated.zip',
            '20140327': '-da_updated.zip',
            '20111012': '-da_update.zip',
            '20111011': '-da_update.zip',
        }
        rt_updated = {
            '20170116': '_updated.zip',
            '20170115': '_updated.zip',
            '20170114': '_updated.zip',
            '20170113': '_updated.zip',
            '20160923': '_updated.zip',
            '20160417': '_updated.zip',
            '20160416': '_updated.zip',
            '20160415': '_updated.zip',
            '20151110': '_updated.zip',
            '20150929': '_updated.zip',
            '20150901': '_updated.zip',
            '20150831': '_updated.zip',
            '20150601': '_updated.zip',
            '20150504': '_updated.zip',
            '20150427': '_updated.zip',
            '20150407': '_updated.zip',
            '20150310': '_updated.zip',
            '20150309': '_updated.zip',
            '20150201': '_updated.zip',
            '20150131': '_updated.zip',
            '20150130': '_updated.zip',
            '20141112': '_updated.zip',
            '20141023': '_updated.zip',
            '20141013': '_updated.zip',
            '20140805': '_updated.zip',
            '20140710': '_updated.zip',
            '20140507': '_updated.zip',
            '20140128': '_updated.zip',
            '20131125': '_updated.zip',
            '20131120': '_updated.zip',
            '20130424': '_updated.zip',
            '20130307': '_updated.zip',
            '20121109': '_updated.zip',
            '20121023': '_updated.zip',
            '20121004': '_updated.zip',
            '20121003': '_updated2.zip',
            '20121001': '_updated.zip',
            '20110914': '_updated.zip',
            '20110829': '_updated.zip',
            '20110617': '_updated.zip',
            '20110306': '_updated.zip',
            '20110305': '_updated.zip',
            '20110304': '_updated.zip',
            '20101005': '_updated.zip',
            '20100526': '_updated.zip',
            '20100201': '_updated.zip',
            '20100129': '_updated.zip',
            '20100125': '_updated.zip',
            '20080904': '_updated.zip',
            '20080413': '_updated.zip',
            '20080305': '_updated.zip',
            '20080215': '_updated.zip',
            '20080214': '_updated.zip',
            '20071002': '_updated.zip',
            '20070822': '_updated.zip',
        }
        if market == 'da':
            # print("Download 'updated' files from http://www.pjm.com/markets-and-operations/"
            #     "energy/day-ahead/lmpda.aspx and replace the files of the corresponding date"
            #     "downloaded here")
            # ### Files switch from .zip to .csv on 20171109 for day-ahead
            # urls = [(urlstart[iso][market] + file + '-da.csv')
            #         if int(file) >= 20171109
            #         else (urlstart[iso][market] + file + '-da.zip')
            #         for file in files[iso]]
            # ^ Out of date; files have been reposted as zips (20180621)
            urls = [(urlstart[iso][market] + file + da_updated[file])
                    if file in da_updated.keys()
                    else (urlstart[iso][market] + file + '-da.zip')
                    for file in files[iso]]
        elif market == 'rt':
            # print("Download 'updated' files from http://www.pjm.com/markets-and-operations/"
            #     "energy/real-time/lmpda.aspx and replace the files of the corresponding date"
            #     "downloaded here")
            # ### Files switch from .zip to .csv on 20171212 for real-time
            # urls = [(urlstart[iso][market] + file + '.csv')
            #         if int(file) >= 20171212
            #         else (urlstart[iso][market] + file + '.zip')
            #         for file in files[iso]]
            # ^ Out of date; files have been reposted as zips (20180621)
            urls = [(urlstart[iso][market] + file + rt_updated[file])
                    if file in rt_updated.keys()
                    else (urlstart[iso][market] + file + '.zip')
                    for file in files[iso]]


        download_file_series(
            urlstart='', urlend='', fileseries=urls, filepath=filepath, 
            overwrite=overwrite, sleeptime=sleeptime, numattempts=numattempts)

    elif iso == 'NYISO':
        ### NYISO files are zipped by month; put them in a separate folder
        zippath = '{}/in/lmp/{}-zip/'.format(iso, market)
        if not os.path.exists(zippath): os.makedirs(zippath)
        download_file_series(
            urlstart=urlstart[iso][market], urlend=urlend[iso][market],
            fileseries=files[iso], filepath=zippath, 
            overwrite=overwrite, sleeptime=sleeptime, numattempts=numattempts)

        ### Unzip files
        zips = [(zippath + file + urlend[iso][market]) for file in files[iso]]
        for i in trange(len(zips)):
            zip_ref = zipfile.ZipFile(zips[i], 'r')
            zip_ref.extractall(filepath)
            zip_ref.close()

    elif iso == 'CAISO':
        if (submarket == None) and (market == 'rt'): submarket = 'RTM'
        elif (submarket == None) and (market == 'da'): submarket = 'DAM'

        if submarket in ['RTM', 'HASP', 'RTPD']:
            interval = pd.Timedelta('1H')
        elif submarket in ['DAM', 'RUC']:
            interval = pd.Timedelta('1D')

        ### Set output filepath
        filepath = '{}/in/{}/{}/'.format(iso, product.lower(), market)
        if (((market == 'da') and (submarket != 'DAM')) 
            or ((market == 'rt') and (submarket != 'RTM'))):
            filepath = '{}/in/{}/{}/{}/'.format(
                iso, product.lower(), market, submarket)
        if not os.path.exists(filepath): os.makedirs(filepath)

        queries = pd.date_range(
            start=pd.Timestamp('{}-01-01T00:00'.format(year)),
            end=(pd.Timestamp('{}-01-01T00:00'.format(year+1)) - interval),
            freq=interval)

        ### Initialize error container and subset if necessary
        errors = []
        if subset == None: subset = slice(None)

        # already_downloaded = glob('{}{}*'.format(filepath, year))
        for query in tqdm(queries[subset]):
            # if '{}{}.gz'.format(filepath, stampify(query)) not in already_downloaded:
            if interval == pd.Timedelta('1D'):
                fileout = stampify(query)[:-3]
            elif interval == pd.Timedelta('1H'):
                fileout = stampify(query)
            if not os.path.exists('{}{}.gz'.format(filepath, fileout)):
            # if overwrite == False:
            #     if os.path.exists('{}{}.gz'.format(filepath, stampify(query))):
            #         break
                try:
                    download_caiso_lmp_allnodes(
                        market=submarket, start=str(query), filepathout=filepath, 
                        product=product, numattempts=numattempts, waittime=waittime)
                except (URLError, IncompleteRead, pd.errors.ParserError, 
                        BadZipFile, HTTPError) as error:
                    errors.append(error)
                    print(error)

        if len(errors) > 0:
            pd.Series(errors).to_csv(
                '{}__Errors__{}.csv'.format(filepath, time.strftime('%Y%m%dT%H%M%S')),
                index=False)


################
### NODALIZE ###

def nodalize(year, market, iso,
    filepathin=None, filepathout=None, nodesfile=None,
    product='LMP', submarket=None, fillmissinghour=True):
    """
    """
    ### Set defaults if necessary
    if iso.upper() == 'CAISO':
        if filepathin == None:
            filepathin = revmpath+'{}/in/{}/{}'.format(
                iso, product.lower(), market)
            if (((market == 'da') and (submarket != 'DAM')) 
                or ((market == 'rt') and (submarket != 'RTM'))):
                filepathin = revmpath+'{}/in/{}/{}/{}/'.format(
                    iso, product.lower(), market, submarket)
        
        if filepathout == None:
            filepathout = revmpath+'{}/io/{}-nodal/{}/'.format(
                iso, product.lower(), market)

            if (market == 'rt') and (submarket == 'RTM'):
                filepathout = revmpath+'{}/io/{}-nodal/{}-month/'.format(
                    iso, product.lower(), market)

            if (((market == 'da') and (submarket != 'DAM')) 
                or ((market == 'rt') and (submarket != 'RTM'))):
                filepathout = revmpath+'{}/io/{}-nodal/{}/{}/'.format(
                    iso, product.lower(), market, submarket)

        if (submarket == None) and (market == 'rt'): submarket = 'RTM'
        elif (submarket == None) and (market == 'da'): submarket = 'DAM'

    elif iso.upper() == 'ERCOT':
        if (filepathin == None) and (market == 'da'):
            filepathin = revmpath+'{}/in/lmp/{}/{}/'.format(iso, market, year)

        elif (filepathout == None) and (market == 'rt'):
            filepathout = revmpath+'{}/io/lmp-nodal/{}-month/'.format(iso, market)
        elif filepathout == None:
            filepathout = revmpath+'{}/io/lmp-nodal/{}/'.format(iso, market)

    else:
        if filepathin == None:
            filepathin = revmpath+'{}/in/lmp/{}/'.format(iso, market)
        if filepathout == None:
            filepathout = revmpath+'{}/io/lmp-nodal/{}/'.format(iso, market)

    ### Make output folders if necessary
    if not os.path.exists(filepathout):
        os.makedirs(filepathout, exist_ok=True)
    if not os.path.exists(revmpath+'{}/io/missingnodes/'.format(iso.upper())):
        os.makedirs(revmpath+'{}/io/missingnodes/'.format(iso.upper()), exist_ok=True)
    if not os.path.exists(revmpath+'{}/io/datatimes/'.format(iso.upper())):
        os.makedirs(revmpath+'{}/io/datatimes/'.format(iso.upper()), exist_ok=True)
    if not os.path.exists(revmpath+'{}/io/fulltimenodes/year/'.format(iso.upper())):
        os.makedirs(revmpath+'{}/io/datatimes/'.format(iso.upper()), exist_ok=True)
    if not os.path.exists(revmpath+'{}/io/fulltimenodes/day/{}/'.format(iso.upper(), market)):
        os.makedirs(revmpath+'{}/io/fulltimenodes/day/{}/'.format(iso.upper(), market), 
            exist_ok=True)
    print(filepathout)


    ### Shared components
    nodesfiles = {
        'CAISO': revmpath+'CAISO/io/caiso-node-latlon.csv',
        'ERCOT': revmpath+'ERCOT/io/ercot-node-latlon.csv',
        #'MISO':  revmpath+'MISO/in/miso-node-map.csv',
        'MISO':  revmpath+'MISO/io/miso-node-latlon.csv',
        # 'PJM':   revmpath+'PJM/io/pjm-pnode-latlon-uniquepoints.csv',
        'PJM':   revmpath+'PJM/io/pjm-node-latlon.csv',
        'NYISO': revmpath+'NYISO/io/nyiso-node-latlon.csv',
        'ISONE': revmpath+'ISONE/io/isone-node-latlon.csv' 
    }
    if nodesfile is None:
        nodesfile = nodesfiles[iso]
    resolution = {
        'CAISO': {'da': 60, 'rt':  5}, 'ERCOT': {'da': 60, 'rt':  5}, 
        'MISO':  {'da': 60, 'rt': 60}, 'PJM':   {'da': 60, 'rt': 60}, 
        'NYISO': {'da': 60, 'rt':  5}, 'ISONE': {'da': 60, 'rt': 60}, 
    }

    ### Get file list and iso/market info
    # files = glob('{}{}*'.format(filepathin, year))
    files = sorted(glob('{}{}*'.format(filepathin, year)))
    print('head(files):')
    for file in files[:3]:
        print(file)
    print('tail(files):')
    for file in files[-3:]:
        print(file)
    timezone = pvvm.toolbox.tz_iso[iso]
    res = resolution[iso][market]
    
    ### Make the inputs easier to work with
    iso = iso.upper()
    hours = pvvm.toolbox.yearhours(year)
    dates = pvvm.toolbox.makedays(year)

    ### DO: figure out how to generalize this
    # if len(files) != len(dates):
    #     print('len(files) = {}'.format(len(files)))
    #     print('len(dates) = {}'.format(len(dates)))
    #     raise Exception("files and dates don't match")


    if iso == 'ISONE':
        ### Load file containing nodes with geographic information
        nodesin = pd.read_csv(nodesfile, usecols=[0], squeeze=True, 
            names=['Node'], skiprows=1)

        ### Load daily files
        colnames = ['intime', 'node', 'lmp']
        dfdict = {}

        for i in trange(len(files)):
            dfday = pd.read_csv(
                files[i], skiprows=6, usecols=[2,4,6], names=colnames,
                dtype={'intime':str, 'node':'category', 'lmp':float})
            dfday.drop(dfday.index[-1], inplace=True)
            dfday.loc[:,'intime'] = dates[i] + 'H' + dfday.loc[:,'intime']
            dfdict[dates[i]] = dfday

        ### Concat into one dataframe with localized datetime index
        dfall = pd.concat(dfdict)

        ### Make new index
        oldtime = list(dfall.intime.unique())

        newtime = list(pd.date_range(dates[0], freq='H', periods=pvvm.toolbox.yearhours(year)))
        for i in range(len(newtime)):
            newtime[i] = str(newtime[i])

        indexconvert = dict(zip(oldtime, newtime))

        dfall.loc[:,'intime'] = dfall.loc[:,'intime'].apply(
            lambda x: indexconvert[x])
        dfall.loc[:,'intime'] = pd.to_datetime(dfall['intime'])

        fullindex = pd.date_range(dates[0], freq='H', periods=pvvm.toolbox.yearhours(year))
        fullindex = fullindex.tz_localize(timezone)
        fullindex = pd.DataFrame(index=fullindex)

        ### Determine missing nodes and data coverage, and save as one-node files
        missingnodes = []
        datalength = []

        for j in trange(len(nodesin)):  
            try:
                df = dfall[dfall['node'] == nodesin[j]][['intime','lmp']].copy()
                df.index = df['intime'].values
                del df['intime']
                df.index = df.index.tz_localize(timezone)
                
                df = df.merge(fullindex, how='right', left_index=True, right_index=True)
                
                numhours = hours - len(df[df['lmp'].isnull()])
                datalength.append([nodesin[j], numhours])
                df.to_csv('{}{}-{}.gz'.format(filepathout, nodesin[j], year), 
                    compression='gzip', header=False)
            except KeyError:
                missingnodes.append(nodesin[j])
                continue

    elif iso == 'MISO':
        ### Load file containing nodes with geographic information
        nodesin = pd.read_csv(nodesfile, usecols=[0], squeeze=True, names=['Node'])

        ### Pick columns from input file
        usecols = [0, 2,
            3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14, 
            15,16,17,18,19,20,21,22,23,24,25,26]

        ### Load daily files
        dfdict = {}
        for i in trange(len(files)):
            colnames = ['Node', 'Value']
            for j in range(24):
                colnames.append(dates[i] + 'H{:02d}'.format(j))
            dfin = pd.read_csv(
                files[i], skiprows=5, header=None, 
                usecols=usecols,
                dtype={0: 'category'}, names=colnames)

            dfday = dfin.loc[dfin['Value'] == 'LMP'].T.copy()
            dfday.columns = dfday.iloc[0,:]
            dfday = dfday.drop(dfday.index[[0,1]])
            
            dfdict[dates[i]] = dfday

        ### Concat into one dataframe with localized datetime index
        dfall = pd.concat(dfdict)
        dfall.index = dfall.index.droplevel(0)
        dfall.index = pd.date_range(dates[0], periods=hours, freq='H')
        dfall.index = dfall.index.tz_localize(timezone)

        ### Determine missing nodes and data coverage, and save as one-node files
        missingnodes = []
        datalength = []
        for j in trange(len(nodesin)):
            try:
                df = pd.DataFrame(dfall.loc[:,nodesin[j]])
                numhours = hours - len(df[df[nodesin[j]].isnull()])
                datalength.append([nodesin[j], numhours])
                df.to_csv('{}{}-{}.gz'.format(filepathout, nodesin[j], year), 
                    compression='gzip', header=False)
            except KeyError:
                missingnodes.append(nodesin[j])
                continue

    elif iso == 'PJM':
        ### Set skiprows (different headers for 'da' and 'rt' markets)
        skiprows = {'da': 8, 'rt': 18}

        ### Load file containing nodes with geographic information
        nodesin = pd.read_csv(nodesfile, usecols=[0], squeeze=True)

        ### Pick columns from input file
        usecols = [1, 
            7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 
            43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76]

        usecols_dst_springforward = [1, 
            7, 10, 16, 19, 22, 25, 28, 31, 34, 37, 40, 
            43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76]

        usecols_dst_fallback = [1, 
            7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 
            43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]

        ### Load daily files
        dfdict = {}
        for i in trange(len(files)):
            colnames = ['PnodeID']
            if dates[i] not in [pvvm.toolbox.dst_springforward[year], pvvm.toolbox.dst_fallback[year]]:
                for j in range(24):
                    colnames.append(dates[i] + 'H{:02d}'.format(j))
                dfin = pd.read_csv(
                    files[i], skiprows=skiprows[market], header=None, 
                    usecols=usecols,
                    dtype={1: 'category'}, names=colnames)

            elif dates[i] == pvvm.toolbox.dst_springforward[year]:
                for j in range(23):
                    colnames.append(dates[i] + 'H{:02d}'.format(j))
                dfin = pd.read_csv(
                    files[i], skiprows=skiprows[market], header=None, 
                    usecols=usecols_dst_springforward,
                    dtype={1: 'category'}, names=colnames)

            elif dates[i] == pvvm.toolbox.dst_fallback[year]:
                for j in range(25):
                    colnames.append(dates[i] + 'H{:02d}'.format(j))
                dfin = pd.read_csv(
                    files[i], skiprows=skiprows[market], header=None, 
                    usecols=usecols_dst_fallback,
                    dtype={1: 'category'}, names=colnames)
                
            dfday = dfin.T.copy()
            dfday.columns = dfday.iloc[0,:]
            dfday = dfday.drop(dfday.index[[0]])
            del dfday[np.nan]

            dfdict[dates[i]] = dfday

        ### Concat into one dataframe with localized datetime index
        dfall = pd.concat(dfdict)
        dfall.index = dfall.index.droplevel(0)
        dfall.index = pd.date_range(dates[0], periods=hours, freq='H')
        dfall.index = dfall.index.tz_localize(timezone)

        ### Determine missing nodes and data coverage, and save as one-node files
        missingnodes = []
        datalength = []
        for j in trange(len(nodesin)):
            try:
                df = pd.DataFrame(dfall.loc[:,nodesin[j].astype(str)])
                numhours = hours - len(df[df[nodesin[j].astype(str)].isnull()])
                datalength.append([nodesin[j], numhours])
                df.to_csv(filepathout + '{}-{}.gz'.format(nodesin[j], year), 
                    compression='gzip', header=False)
            except KeyError:
                missingnodes.append(nodesin[j])
                continue

    elif iso == 'NYISO':
        ### Load file containing nodes with geographic information
        nodesin = pd.read_csv(nodesfile, usecols=[0], squeeze=True, names=['node'], skiprows=1)

        if market == 'da':
            
            dates = pvvm.toolbox.makedays(year)
            if len(files) != len(dates):
                print('len(files) = {}'.format(len(files)))
                print('len(dates) = {}'.format(len(dates)))
                raise Exception("files and dates don't match") 

            ### Make daylight savings mangler
            def dstfallback(dataframe):
                fallback = pvvm.toolbox.dst_fallback[year]
                backfall = '{}/{}/{}'.format(fallback[4:6], fallback[6:], fallback[:4])

                fallbackhalf = int(len(dataframe[dataframe['intime'] == backfall + ' 01:00'])/2)

                if str(dataframe[dataframe['intime'] == backfall + ' 01:00'].iloc[0,1]) != \
                    str(dataframe[dataframe['intime'] == backfall + ' 01:00'].iloc[fallbackhalf,1]):
                        raise Exception("DST fallback ptid's don't match.")

                mask = dataframe['intime'] == backfall + ' 01:00'
                mask.iloc[fallbackhalf:2*fallbackhalf] = False

                dataframe.loc[mask, 'intime'] = backfall + ' 01:00 DST'
                print("DST fallback conversion worked!")
                return dataframe    

            ### Make datetime converter
            def makeindexconvert(files, dates):
                """
                """
                dicttimes = {}

                for i in trange(len(files)):
                    df = pd.read_csv(files[i], 
                        usecols = [0,2,3], skiprows=1,
                        names=['intime', 'node', 'lmp'], 
                        dtype={'ptid': 'category', 'lmp': float})
                    
                    if dates[i] == pvvm.toolbox.dst_fallback[year]:
                        # print(df.head())
                        df = dstfallback(df)

                    dicttimes[dates[i]] = df
                
                dftimes = pd.concat(dicttimes, copy=False)
                
                oldtime = list(dftimes.intime.unique())
                print('len(oldtime) = {}'.format(len(oldtime)))

                newtime = list(pd.date_range(dates[0], freq='H', periods=pvvm.toolbox.yearhours(year)))
                print('len(newtime) = {}'.format(len(newtime)))
                
                for i in range(len(newtime)):
                    newtime[i] = str(newtime[i])
                
                indexconvert = dict(zip(oldtime, newtime))
                
                return indexconvert

            indexconvert = makeindexconvert(files, dates)

            ### Load daily files

            dfdict = {}

            for i in trange(len(files)):
                dfday = pd.read_csv(files[i], 
                    usecols = [0,2,3], skiprows=1,
                    names=['intime', 'node', 'lmp'], 
                    dtype={'ptid': 'category', 'lmp': float})
                
                if dates[i] == pvvm.toolbox.dst_fallback[year]:
                    dfday = dstfallback(dfday)
                
                dfday.loc[:,'intime'] = dfday.loc[:,'intime'].apply(lambda x: indexconvert[x])
                dfday.loc[:,'intime'] = pd.to_datetime(dfday['intime'])
                
                dfdict[dates[i]] = dfday

            ### Concat into one dataframe with localized datetime index
            ### copy=False is experimental
            dfall = pd.concat(dfdict, copy=False)

            ### Change node type to 'category'. SUPER important. >10x speedup.
            dfall['node'] = dfall['node'].astype('category')

            ### Make new index
            fullindex = pd.date_range(dates[0], freq='H', periods=pvvm.toolbox.yearhours(year))
            fullindex = fullindex.tz_localize(timezone)
            fullindex = pd.DataFrame(index=fullindex)

            ### Determine missing nodes and data coverage, and save as one-node files
            missingnodes = []
            datalength = []
            fulldaynodes = {}
            for j in trange(len(nodesin)):
            # for j in trange(20):
                node = str(nodesin[j])
                try:
                    df = dfall[dfall['node'] == nodesin[j]][['intime','lmp']].copy()
                    df.index = df['intime'].values
                    del df['intime']
                    df.index = df.index.tz_localize(timezone)
                    
                    df = df.merge(fullindex, how='right', left_index=True, right_index=True)
                    
                    ## Record datapoints
                    numhours = hours - len(df[df['lmp'].isnull()])
                    datalength.append([nodesin[j], numhours])

                    ## Determine full-data days
                    dfcount = df.groupby([df.index.month, df.index.day]).count()
                    for date in dates:
                        month = int(date[4:6])
                        day = int(date[6:])
                        count = dfcount.loc[month].loc[day][0]
                        if count == 24:
                            nodes = fulldaynodes.get(date, [])
                            nodes.append(node)
                            fulldaynodes[date] = nodes

                    ## Write nodalized file
                    df.to_csv('{}{}-{}.gz'.format(filepathout, nodesin[j], year), 
                        compression='gzip', header=False)

                except KeyError:
                    missingnodes.append(nodesin[j])
                    continue


        elif market == 'rt':
            datesprev = pvvm.toolbox.makedays(year - 1)
            datesthis = pvvm.toolbox.makedays(year)
            dates = [datesprev[-1]] + datesthis

            filesprev = sorted(glob('{}{}*'.format(filepathin, (year - 1))))
            filesthis = sorted(glob('{}{}*'.format(filepathin, year)))
            files = [filesprev[-1]] + filesthis

            if len(files) != len(dates): 
                print('len(files) = {}'.format(len(files)))
                print('len(dates) = {}'.format(len(dates)))
                for date in dates:
                    if date not in [file[88:96] for file in files]:
                        print(date)
                raise Exception("files and dates don't match")

            ### Make nice index
            niceindex_hourstart = pd.date_range(
                start='{}-01-01 00:00'.format(year),
                periods = hours * 12,
                freq = '5T',
                tz=pvvm.toolbox.tz_iso[iso])
            niceindex = pd.DataFrame(index=niceindex_hourstart)

            ### Load daily files
            dfdict = {}
            for i in trange(len(files)):
                df = pd.read_csv(
                    files[i], 
                    usecols=[0,2,3],
                    skiprows=1,
                    names=['intime', 'node', 'lmp'],
                    dtype={'intime': 'category', 
                        'node': 'category', 
                        'lmp': float},
                    parse_dates=['intime'],
                    infer_datetime_format=True)
                dfdict[dates[i]] = df

            ### Concat into one dataframe with localized datetime index
            dfall = pd.concat(dfdict, copy=False)

            ### Change node type to 'category'. SUPER important. >10x speedup.
            dfall['node'] = dfall['node'].astype('category')

            ### Check number of nodes. Good for error checking.
            numnodes = len(dfall['node'].unique())
            print("len(dfall['node']): {}".format(numnodes))

            ### Reset index
            dfall.index = dfall['intime'].values
            dfall.index = dfall.index.tz_localize(pvvm.toolbox.tz_iso[iso])

            ### Fix DST
            dststart = dfall.index.get_loc(pvvm.toolbox.dst_springforward[year] + ' 01:55')
            print('len(dststart) = {}'.format(len(dststart)))
            print('num nodes = {}'.format(numnodes))
            if len(dststart) > numnodes: 
                raise Exception('len(dststart) > numnodes')
            dststart = dststart[-1] + 1

            if year == 2012:
                dstend = dfall.index.get_loc(pvvm.toolbox.dst_fallback[year] + ' 01:59:34')
            else:
                dstend = dfall.index.get_loc(pvvm.toolbox.dst_fallback[year] + ' 01:55')
            print('len(dstend) = {}'.format(len(dstend)))
            if year == 2012:
                if len(dstend) > numnodes: 
                    raise Exception('len(dststart) > numnodes')
                dstend = dstend[-1]
            else:
                if len(dstend) % 2 != 0: 
                    raise Exception('len(dstend) % 2 != 0')
                if len(dstend) / 2 > numnodes: 
                    raise Exception('len(dstend) / 2 > numnodes')
                if ((dstend[int(len(dstend)/2) + 0] - dstend[int(len(dstend)/2) - 1] - 1) / 11
                    != (len(dstend) / 2)):
                    print((dstend[int(len(dstend)/2) + 0] - dstend[int(len(dstend)/2) - 1] - 1) / 11)
                    print(len(dstend) / 2)
                    raise Exception('node added or lost during DST fallback')
                dstend = dstend[int(len(dstend)/2) - 1]

            dfall.iloc[dststart:(dstend + 1),0] = (
                dfall.iloc[dststart:(dstend + 1),0]
                + pd.Timedelta(-1, unit='h'))

            ### Reset index
            dfall.index = dfall['intime'].values
            dfall.index = dfall.index.tz_localize(pvvm.toolbox.tz_iso[iso])

            ### Determine missing nodes and data coverage, and save as one-node files
            missingnodes = []
            datalength = []
            fulldaynodes = {}
            for j in trange(len(nodesin)):
                node = str(nodesin[j])
                try:
                    dfin = dfall[dfall['node'] == node].copy()

                    ## Add missing timestamps
                    df = dfin.merge(
                        niceindex,
                        how='outer',
                        left_index=True, right_index=True)

                    ## Fill gaps, using off-5T values
                    df = df['lmp'].interpolate(method='time', limit=11)

                    ## Remove off-5T values
                    dfout = pd.DataFrame(df).merge(
                        niceindex, 
                        how='right',
                        left_index=True, right_index=True)

                    ## Fill missing hour if desired
                    if fillmissinghour:
                        dfout = dfout.interpolate('linear', limit=12)

                    ## Record datapoints
                    numpoints = dfout.notnull().sum().values[0]
                    datalength.append([nodesin[j], numpoints])

                    ## Determine full-data days
                    dfcount = dfout.groupby([dfout.index.month, dfout.index.day]).count()
                    for date in dates[1:]:
                        month = int(date[4:6])
                        day = int(date[6:])
                        count = dfcount.loc[month].loc[day][0]
                        if count == 288:
                            nodes = fulldaynodes.get(date, [])
                            nodes.append(node)
                            fulldaynodes[date] = nodes

                    ## Write nodalized file
                    dfout.to_csv(
                        '{}{}-{}.gz'.format(
                            filepathout, nodesin[j], year),
                        compression='gzip', header=False)

                except KeyError:
                    missingnodes.append(node)
                    continue

    elif iso == 'CAISO':
        if market == 'da':
            ### Input housekeeping
            filesin = sorted(glob('{}{}*'.format(filepathin, year)))
            datesin = pvvm.toolbox.makedays(year)
            if len(filesin) != len(datesin):
                print('filepathin = {}'.format(filepathin))
                print('len(filesin) = {}'.format(len(filesin)))
                print('len(datesin) = {}'.format(len(datesin)))
                raise Exception("files and dates don't match")
                
            ### Load file containing nodes with geographic information
            nodesin = pd.read_csv(nodesfile, usecols=[0], squeeze=True, 
                names=['Node'], skiprows=1)

            ### Make nice hourly index
            hourlyindex = pd.date_range(
                start='{}-01-01 00:00'.format(year),
                end='{}-12-31 23:00'.format(year),
                freq = '1H',
                tz=pvvm.toolbox.tz_iso[iso])
            hourlyindex = pd.DataFrame(index=hourlyindex)

            ### Make nice daily index
            dailyindex = pd.date_range(
                start='{}-01-01'.format(year),
                end='{}-12-31'.format(year),
                freq='1D')

            ### Load daily files
            dfdict = {}
            for i in trange(len(filesin)):
                if ((product == 'lmp') and (market == 'da')):
                    df = pd.read_csv(
                        filesin[i],
                        usecols=[1,2,3],
                        skiprows=1,
                        names=['node', 'intime', product],
                        dtype={'intime':'category', 'node':'category', product:float},
                        # index_col='intime',
                        parse_dates=['intime'],
                        infer_datetime_format=True
                    )
                    # df.intime = df.intime.map(
                    #     lambda x: pd.to_datetime('{}{}{} {}:00'.format(x[:4], x[5:7], x[9:11], x[12:14])))
                else:
                    df = pd.read_csv(
                        filesin[i], 
                        usecols=[0,1,2],
                        skiprows=1,
                        names=['intime', 'node', product],
                        dtype={'intime':'category', 'node':'category', product:float},
                        parse_dates=['intime'],
                        infer_datetime_format=True
                    )
                dfdict[datesin[i]] = df

            ### Concat into one dataframe
            dfall = pd.concat(dfdict, copy=False)
            # dfall.reset_index(level=0, drop=True, inplace=True)

            ### Categorize nodes (accelerates lookup)
            dfall['node'] = dfall['node'].astype('category')

            ### Check number of nodes. Good for error checking.
            numnodes = len(dfall['node'].unique())
            print("numnodes = {}".format(numnodes))

            ### Reset index and set to local timezone
            dfall.index = dfall['intime'].values
            dfall.index = (
                dfall.index
                .tz_localize('UTC')
                .tz_convert(pvvm.toolbox.tz_iso[iso]))

            ### Determine missing nodes and data coverage, and save as one-node files
            missingnodes = []
            datalength = []
            fulldaynodes = {}
            for j in trange(len(nodesin)):
                node = str(nodesin[j])
                try:
                    dfin = dfall[dfall['node'] == node].copy()

                    ## Add missing timestamps
                    df = dfin.merge(
                        hourlyindex,
                        how='right', 
                        left_index=True, right_index=True)

                    df = pd.DataFrame(df[product])

                    ## Record datapoints
                    numpoints = df.notnull().sum().values[0]
                    datalength.append([nodesin[j], numpoints])

                    ## Determine full-data days
                    dfcount = df.groupby([df.index.month, df.index.day]).count()
                    for date in dailyindex:
                        month = date.month
                        day = date.day
                        count = dfcount.loc[month].loc[day][0]
                        if count == 24:
                            nodes = fulldaynodes.get(date.strftime('%Y%m%d'), [])
                            nodes.append(node)
                            fulldaynodes[date.strftime('%Y%m%d')] = nodes

                    ## Write nodalized file
                    ## ONLY if it contains data
                    if df.notnull().sum()[0] > 0:
                        df.to_csv(
                            '{}{}-{}.gz'.format(
                                filepathout, node, year),
                            compression='gzip', header=False)
                    else:
                        missingnodes.append(node)
                except KeyError:
                    missingnodes.append(node)

        elif market == 'rt':
            ### Make convenience variables
            months = list(range(1,13))

            ### Load file containing nodes with geographic information
            nodesin = list(pd.read_csv(
                nodesfile, 
                usecols=[0],
                squeeze=True
            ))

            ### Loop over months
            for month in months:
                datetimesin = pd.date_range(
                    start='{}{:02}01T{:02}:00'.format(year, month, abs(pvvm.toolbox.timezone_iso[iso])),
                    periods = pvvm.toolbox.monthhours(year, month),
                    freq = 'H')
                files = ['{}{}.gz'.format(filepathin, d.strftime('%Y%m%dT%H')) for d in datetimesin]

                ### Make nice MONTHLY index
                niceindex = pd.date_range(
                    start='{}-{:02}-01 00:00'.format(year, month),
                    periods = (pvvm.toolbox.monthhours(year, month) * 60 / res),
                    freq = '5T',
                    tz=pvvm.toolbox.tz_iso[iso])
                niceindex = pd.DataFrame(index=niceindex)

                ### Make date index (for labeling daily output files)
                dates = pd.date_range(
                    start = '{}-{:02}-01'.format(year, month),
                    periods = (pvvm.toolbox.monthhours(year, month) / 24),
                    freq = '1D')

                dates = [date.strftime('%Y%m%d') for date in dates]

                ### Load daily files
                dfdict = {}
                for i in trange(len(files)):
                    df = pd.read_csv(
                        files[i], 
                        usecols=[0,1,2],
                        skiprows=1,
                        names=['intime', 'node', 'lmp'],
                        dtype={
                            'intime': 'category', 
                            'node': 'category', 
                            'lmp': float},
                        parse_dates=['intime'],
                        infer_datetime_format=True)
                    dfdict[datetimesin[i]] = df

                ### Concat into one dataframe
                dfall = pd.concat(dfdict, copy=False)

                ### Categorize nodes (accelerates lookup)
                dfall['node'] = dfall['node'].astype('category')

                ### Check number of nodes. Good for error checking.
                numnodes = len(dfall['node'].unique())
                print("numnodes({:02}) = {}".format(month, numnodes))

                ### Reset index and set to local timezone
                dfall.index = dfall['intime'].values
                dfall.index = (
                    dfall.index
                    .tz_localize('UTC')
                    .tz_convert(pvvm.toolbox.tz_iso[iso]))

                ### Determine missing nodes and data coverage, and save as one-node files
                missingnodes = []
                datalength = []
                fulldaynodes = {}
                for j in trange(len(nodesin)):
                    node = str(nodesin[j])
                    try:
                        dfin = dfall[dfall['node'] == node].copy()

                        ## Add missing timestamps
                        df = dfin.merge(
                            niceindex,
                            how='right', 
                            left_index=True, right_index=True)
                        
                        df = pd.DataFrame(df['lmp'])

                        ## Record datapoints
                        numpoints = df.notnull().sum().values[0]
                        datalength.append([nodesin[j], numpoints])

                        ## Determine full-data days
                        dfcount = df.groupby([df.index.month, df.index.day]).count()
                        for date in dates:
                            day = int(date[6:])
                            count = dfcount.loc[month].loc[day][0]
                            if count == 288:
                                nodes = fulldaynodes.get(date, [])
                                nodes.append(node)
                                fulldaynodes[date] = nodes

                        ## Write nodalized file
                        ## ONLY if it contains data
                        if df.notnull().sum()[0] > 0:
                            df.to_csv(
                                '{}{}-{}{:02}.gz'.format(
                                    filepathout, nodesin[j], year, month),
                                compression='gzip', header=False)
                        else:
                            missingnodes.append(node)
                    except KeyError:
                        missingnodes.append(node)

            #############################
            ### Unmonthify the nodal lmps
            
            ### Set new filepaths
            filepathin = revmpath+'{}/io/lmp-nodal/{}-month/'.format(iso, market)
            filepathout = revmpath+'{}/io/lmp-nodal/{}/'.format(iso, market)
            if not os.path.exists(filepathout): os.makedirs(filepathout)

            ### Make list of all files for year
            filesin = sorted(glob('{}*-{}??.gz'.format(filepathin, year)))

            ### Make list of all nodes (with duplicates)
            nodes = []
            for i in range(len(filesin)):
                nodes.append(filesin[i][:filesin[i].find('-{}'.format(year))])

            ### Make list of unique nodes
            uniquenodes = np.unique(np.array(nodes))

            ### Make dict of monthly files for each node
            dictfiles = {}
            for node in uniquenodes:
                out = []
                for file in filesin:
                    if file.find(node) != -1:
                        out.append(file)
                dictfiles[node] = out

            ### Load and concat monthly files for each node, then write as yearly csv
            for node in tqdm(uniquenodes):
                dfdict = {}
                for file in dictfiles[node]:
                    dfin = pd.read_csv(
                        file,
                        header=None,
                        names=['datetime', 'lmp'],
                    )
                    dfdict[file[-5:-3]] = dfin
                dfyear = pd.concat(dfdict, ignore_index=True)
                
                dfyear.to_csv(
                    '{}{}-{}.gz'.format(filepathout, node[20:], year),
                    index=False,
                    header=False,
                    compression='gzip'
                )

    elif iso == 'ERCOT':

        #############
        ### Functions

        def makeindexconvert(files, dates):
            """
            Datetime converter for day-ahead
            """
            dicttimes = {}

            for i in trange(len(files)):
                try:
                    df = pd.read_csv(files[i], usecols=[0,1,4], 
                        dtype={
                            'DeliveryDate': 'category', 
                            'HourEnding': 'category', 
                            'DSTFlag': 'category'})
                    df.loc[:,'DSTFlag'] = (
                        df.loc[:,'DeliveryDate'].astype(str) +
                        'H' + df.loc[:,'HourEnding'].astype(str) + 
                        df.loc[:,'DSTFlag'].astype(str)).astype('category')
                except ValueError as err:
                    df = pd.read_csv(files[i], usecols=[0,1], 
                        dtype={
                            'DeliveryDate': 'category', 
                            'HourEnding': 'category'})
                    df['DSTFlag'] = (
                        df['DeliveryDate'].astype(str)
                        + 'H' + df['HourEnding'].astype(str)
                        + 'N').astype('category')

                dicttimes[dates[i]] = df
            
            ### copy=False is experimental
            dftimes = pd.concat(dicttimes, copy=False)
            
            oldtime = list(dftimes.DSTFlag.unique())
            print('len(oldtime) = {}'.format(len(oldtime)))

            newtime = list(pd.date_range(dates[0], freq='H', periods=pvvm.toolbox.yearhours(year)))
            print('len(newtime) = {}'.format(len(newtime)))

            if len(oldtime) != len(newtime):
                raise Exception("len(oldtime) and len(newtime) don't match")
            
            for i in range(len(newtime)):
                newtime[i] = str(newtime[i])
            
            indexconvert = dict(zip(oldtime, newtime))
            
            return indexconvert

        def datetimefromfile(file, clean=False):
            """
            Only works for ERCOT bus RTLMP files
            Example: ('cdr.00011485.0000000000000000.20101201.005033.'
                      'LMPSELECTBUSNP6787_20101201_005025_csv.zip')
            """
            basename = os.path.basename(file)
            file_datetime = basename[65:80]
            year = file_datetime[:4]
            
            if year == 'retr':
                file_datetime = basename[71:86]
            if year == '87_2':
                file_datetime = basename[68:83]
            if year == '87_r':
                file_datetime = basename[74:89]
            year = file_datetime[:4]
            
            try:
                year = int(year)
            except:
                print(year)
                print(type(year))
                print(basename)
                raise ValueError        
            
            if (year < 2010) or (year > 2017):
                print(year)
                print(type(year))
                print(basename)
                raise ValueError
            
            springforward_date = pvvm.toolbox.dst_springforward[year]
            fallback_date = pvvm.toolbox.dst_fallback[year]
            springforward_time = pd.to_datetime(
                '{} 03:00:00'.format(springforward_date))
            fallback_time = pd.to_datetime(
                '{} 02:00:00'.format(fallback_date))
            
            if basename.find('xhr') == -1:
                dst = False
            else:
                dst = True
            
            if not clean:
                datetime_predst = pd.to_datetime(
                    file_datetime, 
                    format='%Y%m%d_%H%M%S')
            elif clean:
                datetime_predst = pd.to_datetime(
                    file_datetime[:-2], 
                    format='%Y%m%d_%H%M')        
            
            if (
                (datetime_predst >= springforward_time)
                & (datetime_predst < fallback_time)
                & (not dst)
            ):
                datetime = datetime_predst - pd.Timedelta('1H')
            
            else:
                datetime = datetime_predst
            
            return datetime

        def datetimebin(datetime, mod=5):
            """
            Take a datetime and determine what bin it falls into.
            If mod == 5, then 08:39:42 --> 08:35:00.
            """
            assert 60 % mod == 0, "60 must be divisible by mod"
            assert type(datetime) == pd.Timestamp, "datetime must be pd.Timestamp"
            newminute = int(datetime.minute / mod) * mod
            out = pd.Timestamp(
                year=datetime.year, month=datetime.month, day=datetime.day,
                hour=datetime.hour, minute=newminute)
            return out
        

        #############
        ### Procedure

        if market == 'da':
            ### Make the inputs easier to work with
            files = sorted(glob(filepathin + '*csv.zip'))
            dates = pvvm.toolbox.makedays(year)
            if len(files) != len(dates): 
                print('filepathin = {}'.format(filepathin))
                print('len(files) = {}'.format(len(files)))
                print('len(dates) = {}'.format(len(dates)))
                raise Exception("files and dates don't match")

            ### Load file containing nodes with geographic information
            nodesin = pd.read_csv(nodesfile, usecols=[0], squeeze=True, names=['Node'], skiprows=1)
            nodesin.drop_duplicates(inplace=True)
            nodesin = list(nodesin)
            for i in range(len(nodesin)):
                nodesin[i] = nodesin[i].upper()

            ### Convert datetimes
            indexconvert = makeindexconvert(files, dates)

            ### Load daily files
            dfdict = {}
            for i in trange(len(files)):
                try:
                    dfday = pd.read_csv(files[i], 
                        dtype={
                            'DeliveryDate': 'category', 'HourEnding': 'category', 
                            'BusName': 'category', 'LMP': float, 'DSTFlag': 'category'})
                    dfday.loc[:,'DSTFlag'] = (
                        dfday.loc[:,'DeliveryDate'].astype(str) +
                        'H' + dfday.loc[:,'HourEnding'].astype(str) + 
                        dfday.loc[:,'DSTFlag'].astype(str)).astype('category')

                except KeyError as err:
                    dfday = pd.read_csv(files[i], 
                        dtype={
                            'DeliveryDate': 'category', 'HourEnding': 'category', 
                            'BusName': 'category', 'LMP': float})
                    dfday['DSTFlag'] = (
                        dfday['DeliveryDate'].astype(str)
                        + 'H' + dfday['HourEnding'].astype(str)
                        + 'N').astype('category')
                    
                dfday.loc[:,'DSTFlag'] = dfday.loc[:,'DSTFlag'].apply(lambda x: indexconvert[x])
                dfday.loc[:,'DSTFlag'] = pd.to_datetime(dfday['DSTFlag'])

                del dfday['DeliveryDate']
                del dfday['HourEnding']
                dfday = dfday.rename(columns={'BusName': 'node', 'DSTFlag': 'intime', 'LMP': 'lmp'})
                
                dfdict[dates[i]] = dfday

            ### Concat into one dataframe with localized datetime index
            ### copy=False is experimental
            dfall = pd.concat(dfdict, copy=False)

            ### Change node type to 'category'. SUPER important. >10x speedup.
            dfall['node'] = dfall['node'].astype('category')

            # if len(dfall.index.unique()) != pvvm.toolbox.yearhours(year):
            #   raise Exception("len(dfall.index.unique() != pvvm.toolbox.yearhours(year)")

            ### Make new index
            fullindex = pd.date_range(dates[0], freq='H', periods=pvvm.toolbox.yearhours(year))
            fullindex = fullindex.tz_localize(timezone)
            fullindex = pd.DataFrame(index=fullindex)

            ### Determine missing nodes and data coverage, and save as one-node files
            missingnodes = []
            datalength = []
            for j in trange(len(nodesin)):
                try:
                    df = dfall[dfall['node'] == nodesin[j]][['intime','lmp']].copy()
                    df.index = df['intime'].values
                    del df['intime']
                    df.index = df.index.tz_localize(timezone)
                    
                    df = df.merge(fullindex, how='right', left_index=True, right_index=True)
                    
                    numhours = hours - len(df[df['lmp'].isnull()])
                    datalength.append([nodesin[j], numhours])
                    df.to_csv('{}{}-{}.gz'.format(filepathout, nodesin[j], year), 
                        compression='gzip', header=False)
                except KeyError:
                    missingnodes.append(nodesin[j])
                    continue

        elif market == 'rt':
            ### Set defaults
            months = list(range(1,13))

            ################################
            ###### Make filekey if necessary
            ###### It takes a while to make, so save ane load if possible
            filekey = revmpath+'{}/io/{}-{}-filekey-cleaned-existing.csv'.format(
                iso.upper(), iso.lower(), market)
            ### Check if it exists already and is long enough to contain data through end 2017
            if (os.path.exists(filekey)) and (len(pd.read_csv(filekey, usecols=[2])) >= 743827):
                pass
            else:
                ### Make list of ALL ERCOT RT LMP files
                allyears = range(2010,2051)
                files = {}
                for i_year in list(allyears):
                    files[i_year] = sorted(glob('{}{}/*csv.zip'.format(filepathin, i_year)))
                files_all = sum([files[i] for i in allyears], [])

                ### Round all files to start of 5-minute bin
                filestodatetimes = {}
                for i in trange(len(files_all)):
                    filestodatetimes[files_all[i]] = datetimefromfile(files_all[i], clean=False)

                filetimes = [filestodatetimes[key] for key in filestodatetimes]

                dffiles = pd.DataFrame(files_all, columns=['Filename'])
                dffiles['Datetime'] = filetimes
                dffiles['Retry'] = dffiles.apply(
                    lambda row: row['Filename'].find('retr') != -1,
                    axis=1)

                dffiles.drop_duplicates(subset='Datetime', keep='last', inplace=True)

                datetimes_binned = []
                for i in dffiles.index:
                    try:
                        datetimes_binned.append(datetimebin(dffiles.loc[i, 'Datetime']))
                    except:
                        print(i)
                        print(dffiles.loc[i, 'Datetime'])
                        raise TypeError

                dffiles['Datetime-Binned'] = datetimes_binned

                dffiles.drop_duplicates(subset='Datetime-Binned', keep='first', inplace=True)

                dffiles.to_csv(
                    revmpath+'{}/io/{}-{}-filekey-cleaned-existing.csv'.format(
                        iso.upper(), iso.lower(), market),
                    index=False)

            ### Load file containing nodes with geographic information
            nodesin = pd.read_csv(nodesfile, usecols=[0], squeeze=True, names=['Node'], skiprows=1)
            nodesin.drop_duplicates(inplace=True)
            nodesin = list(nodesin)
            for i in range(len(nodesin)):
                nodesin[i] = nodesin[i].upper()

            ### Load file with datetime-filename key
            dffiles = pd.read_csv(
                filekey,
                dtype={'Filename': str, 'Retry': bool, 'one': float},
                parse_dates=['Datetime', 'Datetime-Binned'],
                infer_datetime_format=True
            )
            dffiles.index = dffiles['Datetime-Binned']

            #############
            ### PROCEDURE

            ### Set DST switch times
            springforward_date = pvvm.toolbox.dst_springforward[year]
            fallback_date = pvvm.toolbox.dst_fallback[year]
            springforward_time = pd.to_datetime(
                '{} 03:00:00'.format(springforward_date))
            fallback_time = pd.to_datetime(
                '{} 02:00:00'.format(fallback_date))

            ### Loop over months
            for month in months:
                print('{}-{:02}'.format(year, month))

                ### Make nice 5T timestamps for month
                monthindex = pd.date_range(
                    start='{}-{:02}-01 00:00'.format(year, month),
                    periods = (pvvm.toolbox.monthhours(year, month) * 60 / res),
                    freq = '5T',
                #     tz=pvvm.toolbox.tz_iso[iso]
                )
                monthindex = pd.DataFrame(index=monthindex)

                ### Make date index (for labeling daily output files)
                dates = pd.date_range(
                    start = '{}-{:02}-01'.format(year, month),
                    periods = (pvvm.toolbox.monthhours(year, month) / 24),
                    freq = '1D')

                dates = [date.strftime('%Y%m%d') for date in dates]

                ### Make nice 5T timestamps for year
                yearindex = pd.date_range(
                    start='{}-01-01 00:00'.format(year),
                    periods = int(pvvm.toolbox.yearhours(year) * 60 / res),
                    freq = '{}T'.format(res),
                #     tz=pvvm.toolbox.tz_iso[iso]
                )
                yearindex = pd.DataFrame(index=yearindex)

                ### Create list of files to load
                dfmonth = dffiles.loc[
                    (dffiles['Datetime-Binned']
                        >= (
                            pd.Timestamp('{}-{:02}-01 00:00'.format(year, month))) 
                            - pd.Timedelta('1H'))
                    & (dffiles['Datetime-Binned']
                        <= (
                            pd.Timestamp('{}-{:02}-01 00:00'.format(year, month))) 
                            + pd.Timedelta('{}H'.format(pvvm.toolbox.monthhours(year, month) + 1)))
                ]

                filestoload = list(dfmonth['Filename'])

                ### Load the files
                dfdict = {}
                badzipfilecount = 0

                for i in trange(len(filestoload)):
                    file = filestoload[i]
                    datetime = datetimefromfile(file)

                    try:
                        df = pd.read_csv(
                            file, skiprows=1,
                            usecols=[2,3],
                            names=['node', 'lmp'],
                            dtype={'node': 'category', 'lmp': float})
                        
                        df['datetime'] = datetime
                        df['datetime'] = df['datetime'].map(datetimebin)
                        df.index = df['datetime']
                        
                        dfdict[datetime.strftime('%Y%m%d_%H%M%S')] = df[['node', 'lmp']]
                    except zipfile.BadZipFile as err:
                        badzipfilecount += 1
                        print("zipfile.BadZipFile error number {}".format(badzipfilecount))
                        print(err)
                        print(file)
                        print(datetime)

                ### Concat into one dataframe
                dfall = pd.concat(dfdict, copy=False).reset_index(level=0, drop=True)

                ### Clear dfdict to conserve memory (?)
                dfdict = 0

                ### Categorize nodes (accelerates lookup)
                dfall['node']= dfall['node'].astype('category')

                ### Determine missing nodes and data coverage, and save as one-node files
                missingnodes, datalength, fulldaynodes = [], [], {}

                ### v OR could do "for j in trange(len(dfall['node'].unique()"
                ### v then node = str(uniquenodes[j]), to write all nodes
                ### v (not just those with geographic information)

                for j in trange(len(nodesin)):
                    node = str(nodesin[j])
                    try:
                        dfin = dfall[dfall['node'] == node].copy()
                        
                        ## Add missing timestamps
                        df = dfin.merge(
                            monthindex,
                            how='outer',
                            left_index=True, right_index=True)
                        
                        df = pd.DataFrame(df['lmp'])
                        ## For debugging:
                        ## df['interpolated'] = df['lmp'].isnull()
                        
                        ## Fill gaps using linear interpolation
                        df.interpolate(
                            method='time', 
                            # limit=12, 
                            # limit_direction='both',
                            inplace=True)
                        
                        ## Remove off-5T values
                        dfout = pd.DataFrame(df).merge(
                        monthindex,
                        how='right',
                        left_index=True, right_index=True)
                        
                        ########### Drop Duplicates ############
                        ## Drop duplicates
                        dfout = dfout.reset_index().drop_duplicates('index').copy()
                        dfout.index = dfout['index']
                        dfout = dfout.drop('index', axis=1).copy()
                        ########################################

                        ## Record datapoints
                        numpoints = dfout.notnull().sum().values[0]
                        datalength.append([nodesin[j], numpoints])
                        
                        ## Determine full-data days
                        dfcount = dfout.groupby([dfout.index.month, dfout.index.day]).count()
                        for date in dates:
                            day = int(date[6:])
                            count = dfcount.loc[month].loc[day][0]
                            if count == 288:
                                nodes = fulldaynodes.get(date, [])
                                nodes.append(node)
                                fulldaynodes[date] = nodes
                                                
                        ## Write nodalized file
                        ## ONLY if it contains data
                        if dfout.notnull().sum()[0] > 0:
                            dfout.to_csv(
                                '{}{}-{}{:02}.gz'.format(
                                    filepathout, nodesin[j], year, month),
                                compression='gzip', header=False)
                        else:
                            missingnodes.append(node)
                        
                    except KeyError:
                        missingnodes.append(node)

            #############################
            ### Unmonthify the nodal lmps
            
            ### Set new filepaths
            filepathin = revmpath+'{}/io/lmp-nodal/{}-month/'.format(iso, market)
            filepathout = revmpath+'{}/io/lmp-nodal/{}/'.format(iso, market)
            if not os.path.exists(filepathout): os.makedirs(filepathout)

            ### Make list of all files for year
            filesin = sorted(glob('{}*-{}??.gz'.format(filepathin, year)))

            print('len(filesin) = {}'.format(len(filesin)))
            if len(filesin) == 0:
                raise Exception("No files in filepathin")

            ### Make list of all nodes (with duplicates)
            nodes = []
            for i in range(len(filesin)):
                nodes.append(filesin[i][:filesin[i].find('-{}'.format(year))])

            ### Make list of unique nodes
            uniquenodes = np.unique(np.array(nodes))

            print('len(uniquenodes) = {}'.format(len(uniquenodes)))

            ### Make dict of monthly files for each node
            dictfiles = {}
            for node in uniquenodes:
                out = []
                for file in filesin:
                    if file.find(node) != -1:
                        out.append(file)
                dictfiles[node] = out

            ### Load and concat monthly files for each node, then write as yearly csv
            for node in tqdm(uniquenodes):
                dfdict = {}
                for file in dictfiles[node]:
                    dfin = pd.read_csv(
                        file,
                        header=None,
                        names=['datetime', 'lmp'],
                        parse_dates=['datetime']
                    )
                    dfdict[file[-5:-3]] = dfin
                dfyear = pd.concat(dfdict, ignore_index=True)
                
                dfyear.index = dfyear.datetime
                dfyear.index = dfyear.index.tz_localize(pvvm.toolbox.tz_iso[iso])

                dfyear.to_csv(
                    '{}{}-{}.gz'.format(
                        filepathout, 
                        node[len(filepathin):], 
                        year),
                    index_label=False,
                    header=False,
                    columns=['lmp'],
                    compression='gzip'
                )

    ###### Write summary outputs
    if (iso in ['CAISO', 'ERCOT'] and (market == 'rt')):
        ## List of nodes from nodemap that don't have LMP data
        pd.Series(missingnodes).to_csv(
            revmpath+'{}/io/missingnodes/{}-{}lmp-{}-{}{:02}.csv'.format(
                iso.upper(), iso.lower(), market, submarket, year, month), 
            index=False)

        ## Intervals (hours or 5-min chunks) of LMP data per node
        pd.DataFrame(datalength).to_csv(
            revmpath+'{}/io/datatimes/{}-{}lmp-{}-{}{:02}.csv'.format(
                iso.upper(), iso.lower(), market, submarket, year, month), 
            index=False, header=False)

        ## List of nodes with complete data over year
        fulltime = pd.DataFrame(datalength)
        fulltime = fulltime[
            fulltime[1] == pvvm.toolbox.monthhours(year, month) * 12]
        fulltime.to_csv(
            ### Original
            # revmpath+'{}/io/fulltimenodes/year/{}-{}lmp{}-{}{:02}.csv'.format(
            ### New
            revmpath+'{}/io/fulltimenodes/{}-{}lmp{}-{}{:02}.csv'.format(
                iso.upper(), iso.lower(), market, 
                {None:''}.get(submarket,'-'+submarket), year, month), 
            index=False, header=False, columns=[0])

    else:
        ## List of nodes from nodemap that don't have LMP data
        pd.Series(missingnodes).to_csv(
            revmpath+'{}/io/missingnodes/{}-{}lmp-missing-{}.csv'.format(
                iso.upper(), iso.lower(), market, year), 
            index=False)

        ## Intervals (hours or 5-min chunks) of LMP data per node
        pd.DataFrame(datalength).to_csv(
            revmpath+'{}/io/datatimes/{}-{}lmp-datatimes-{}.csv'.format(
                iso.upper(), iso.lower(), market, year), 
            index=False, header=False)

        ## List of nodes with complete data over year
        fulltime = pd.DataFrame(datalength)
        fulltime = fulltime[fulltime[1] == hours * int(24 * 60 / res)]
        fulltime.to_csv(
            ### Original
            # revmpath+'{}/io/fulltimenodes/year/{}-{}lmp-fulltime-{}.csv'.format(
            ### New
            revmpath+'{}/io/fulltimenodes/{}-{}lmp-fulltime-{}.csv'.format(
                iso.upper(), iso.lower(), market, year), 
            index=False, header=False, columns=[0])

    ###### Write daily nodecounts
    if iso == 'ISONE':
        for i in trange(len(dates)):
            dfcount = dfall.loc[dates[i]]
            dfday = dfcount.groupby(dfcount.node).count()['lmp']
            fulltimenodes = list(dfday[dfday == int(24 * 60 / res)].index)
            pd.Series(fulltimenodes).to_csv(
                revmpath+'{}/io/fulltimenodes/day/{}/{}.csv'.format(
                    iso.upper(), market, dates[i]),
                index=False)

    elif (iso in ['MISO', 'PJM']) or ((iso, market) == ('ERCOT', 'da')):
        if iso == 'ERCOT': dfcount = dfall.reset_index(level=0, drop=True)
        dfcount = dfall.groupby([dfall.index.month, dfall.index.day]).count().copy()
        daterange = pd.date_range(dates[0], periods=int(hours / 24), freq='D')
        for i in range(len(daterange)):
            dfday = dfcount.loc[daterange[i].month].loc[daterange[i].day].copy()
            fulltimenodes = list(dfday[dfday == int(24 * 60 / res)].index)
            pd.Series(fulltimenodes).to_csv(
                revmpath+'{}/io/fulltimenodes/day/{}/{}.csv'.format(
                    iso.upper(), market, dates[i]),
                index=False)

    elif (iso in ['NYISO', 'CAISO']) or ((iso, market) == ('ERCOT', 'rt')):
        for date in fulldaynodes:
            nodes = fulldaynodes.get(date, [])
            pd.Series(nodes).to_csv(
                revmpath+'{}/io/fulltimenodes/day/{}/{}.csv'.format(iso.upper(), market, date),
                index=False)


##############################
### EXTRACT NODE LOCATIONS ###

def nodelocations_pjm():
    """
    """
    ### Set up googlemaps
    gmaps = googlemaps.Client(key=apikeys['googlemaps'])

    ### Test if zip code mapping file exists and download if it does not
    zipnodefile = revmpath+'PJM/in/zip-code-mapping.xls'
    if not os.path.exists(zipnodefile):
        url = 'https://www.pjm.com/-/media/markets-ops/energy/lmp-model-info/zip-code-mapping.ashx'
        filepathout = revmpath+'PJM/in/zip-code-mapping.xls'
        urllib.request.urlretrieve(url, filepathout)

    ## Make a clean csv version
    dfin = pd.read_excel(zipnodefile, skiprows=9, dtype={'Zip Code': 'category'})
    dfin['Zip Code'] = dfin['Zip Code'].map(lambda x: '{:>05}'.format(x))
    dfin.to_csv(revmpath+'PJM/in/zip-code-mapping.csv', index=False)

    ## Save the unique zip codes
    df = pd.read_csv(revmpath+'PJM/in/zip-code-mapping.csv', dtype={'Zip Code': 'category'})
    df['Zip Code'] = df['Zip Code'].map(lambda x: '{:>05}'.format(x))
    zips_unique = pd.Series(df['Zip Code'].unique())
    zips_unique.to_csv(revmpath+'PJM/io/zips-pjm-unique.csv', index=False)

    ###### Look up zipcode centers
    numattempts = 200
    sleeptime = 60

    zipcodes = pd.read_csv(
        revmpath+'PJM/io/zips-pjm-unique.csv', dtype='category', 
        names=['zipcode'], squeeze=True)
    zipcodes = zipcodes.map(lambda x: '{:>05}'.format(x))

    out=[]
    for i, zipcode in enumerate(tqdm(zipcodes)):
        attempts = 0
        while attempts < numattempts:
            try:
                ### Original version only added 'zipcode' for these two zipcodes.
                ### Now, additional googlemaps queries return erroneous locations,
                ### so we append 'zipcode' for everything.
                # if zipcode in ['15775', '15777']:
                #     ### These two zipcodes, if queried alone, return locations
                #     ### outside of the PJM territory
                #     location = gmaps.geocode('zipcode {}'.format(zipcode))
                # else:
                #     location = gmaps.geocode(zipcode)
                location = gmaps.geocode('zipcode {}'.format(zipcode))
                ### Continue
                lat = location[0]['geometry']['location']['lat']
                lon = location[0]['geometry']['location']['lng']
                out.append([lat, lon])
                time.sleep(0.02)
                break
            except HTTPError as err:
                print('Rebuffed for {} on attempt # {} by "{}".'
                      'Will retry in {} seconds.'.format(
                        zipcode, attempts, err, sleeptime))
                attempts += 1
                time.sleep(sleeptime)
                
            if attempts >= numattempts:
                raise Exception('Failed on {} after {} attempts'.format(
                    zipcode, attempts))
    
    dfout = pd.DataFrame(out, columns=['latitude', 'longitude'])
    zipgeo = pd.concat([zipcodes, dfout], axis=1)

    ### NOTE: zip code 45418, when searched in google maps, returns a
    ### location in Mexico. So look up zip code at 
    ### https://www.unitedstateszipcodes.org/45418/
    ### and change by hand.
    zipgeo.loc[zipgeo.zipcode.astype(str) == '45418', ['latitude', 'longitude']] = (39.69, -84.26)
    ### Additional error for 25572 fixed by hand
    zipgeo.loc[zipgeo.zipcode.astype(str) == '25572', ['latitude', 'longitude']] = (38.16, -81.91)

    ## Write zipcode coordinates
    zipgeo.to_csv(revmpath+'PJM/io/zips-latlon-pjm.csv', index=False)

    ###### Determine node locations
    ### Load input files
    zips = pd.read_csv(
        revmpath+'PJM/io/zips-latlon-pjm.csv', 
        dtype={'zipcode': 'category'}, index_col='zipcode')

    dfnodes = pd.read_csv(revmpath+'PJM/in/zip-code-mapping.csv', dtype='category')
    dfnodes['Zip Code'] = dfnodes['Zip Code'].map(lambda x: '{:>05}'.format(x))
    dfnodes.PNODEID = dfnodes.PNODEID.astype(int)
    dfnodes = dfnodes.drop_duplicates().copy()
    dfnodes = dfnodes.merge(zips, left_on='Zip Code', right_index=True, how='left')

    pnodeids = list(dfnodes['PNODEID'].sort_values().unique())
    zipcodes = list(dfnodes['Zip Code'].unique())

    ### Put lat, lon in cartesian coordinates (assuming spherical Earth)

    dfnodes['x'] = dfnodes.apply(rowlatlon2x, axis=1)
    dfnodes['y'] = dfnodes.apply(rowlatlon2y, axis=1)
    dfnodes['z'] = dfnodes.apply(rowlatlon2z, axis=1)

    ### Determine centroid of zipcodes listed for each node
    lats, lons = [], []
    for i, pnode in enumerate(pnodeids):
        x = dfnodes[dfnodes['PNODEID'] == pnode]['x'].mean()
        y = dfnodes[dfnodes['PNODEID'] == pnode]['y'].mean()
        z = dfnodes[dfnodes['PNODEID'] == pnode]['z'].mean()

        outlon = math.atan2(y, x) * 180 / math.pi
        rho = math.sqrt(x*x + y*y)
        outlat = math.atan2(z, rho) * 180 / math.pi
        
        lats.append(outlat)
        lons.append(outlon)

    ### Make output dataframe
    dfout = (dfnodes[['PNODEID', 'PNODENAME']]
             .drop_duplicates()
             .sort_values('PNODEID')
             .reset_index(drop=True)
             # .rename(columns={col:col.lower() for col in ['PNODEID','PNODENAME']})
             .rename(columns={'PNODEID':'node','PNODENAME':'nodename'})
    )
    dfout['latitude'] = lats
    dfout['longitude'] = lons

    ### Identify duplicate (lat,lon) tuples for NSRDB
    latlons = dfout[['latitude', 'longitude']].drop_duplicates().copy()
    latlons['latlonindex'] = range(len(latlons))
    dfout = dfout.merge(latlons, on=['latitude', 'longitude'], how='left')

    dfout.to_csv(revmpath+'PJM/io/pjm-node-latlon.csv', index=False)
    latlons[['latlonindex', 'latitude', 'longitude']].to_csv(
        revmpath+'PJM/io/pjm-pnode-unique-latlons-for-nsrdb.csv', index=False)

    return dfout

def nodelocations_caiso():
    ### Test if nodemap xml exists and download if it does not
    nodemapxml = revmpath+'CAISO/in/GetPriceContourMap.xml'
    if not os.path.exists(nodemapxml):
        print("Need to download the input file by hand from "
            "'http://wwwmobile.caiso.com/Web.Service.Chart/api/v1/ChartService/GetPriceContourMap'"
            " and save it at (revmpath + 'CAISO/in/GetPriceContourMap.xml).")
        raise Exception("Input file not found")
        # ### For some reason this downloades the file in json format. Just do it by hand.
        # url = 'http://wwwmobile.caiso.com/Web.Service.Chart/api/v1/ChartService/GetPriceContourMap'
        # xmlfile = revmpath+'CAISO/in/GetPriceContourMap.xml'
        # urllib.request.urlretrieve(url, xmlfile)

    ### Import xml nodemap
    tree = ET.parse(nodemapxml)
    root = tree.getroot()

    ### Get  node names, areas, types, and latlons
    names, areas, types, latlonsraw = [], [], [], []
    for node in root.iter(tag='{urn:schemas.caiso.com/mobileapp/2014/03}n'):
        names.append(node.text)

    for node in root.iter(tag='{urn:schemas.caiso.com/mobileapp/2014/03}a'):
        areas.append(node.text)

    for node in root.iter(tag='{urn:schemas.caiso.com/mobileapp/2014/03}p'):
        types.append(node.text)

    latlonsraw = []
    for node in root.iter(tag='{http://schemas.microsoft.com/2003/10/Serialization/Arrays}decimal'):
        latlonsraw.append(float(node.text))

    lats = latlonsraw[::2]
    lons = latlonsraw[1::2]

    ### Generate output dataframe
    dfout = pd.DataFrame({'node': names, 'latitude': lats, 'longitude': lons,
                          'area': areas, 'type': types}).sort_values('node')

    ### Clean up output: Drop nodes with erroneous coordinates
    dfclean = dfout.loc[
        (dfout.longitude > -180)
        & (dfout.longitude < 0)
        & (dfout.latitude > 20)
    ].copy()[['node', 'latitude', 'longitude', 'area', 'type']]

    ### Write output
    dfclean.to_csv(revmpath+'CAISO/io/caiso-node-latlon.csv', index=False)
    return dfclean

def nodelocations_miso():
    import geopandas as gpd
    ### Test if nodemap json files exist and download if not
    filepaths = {
        'Nodes': revmpath + 'MISO/in/MISO_GEN_INT_LZN.json',
        'Hubs':  revmpath + 'MISO/in/PNODELMPLabels_2.json',
        'ReserveZones': revmpath + 'MISO/in/ASMZones_2.json',
        'PlanningZones': revmpath + 'MISO/in/Planning_Zones.json',
    }
    if not os.path.exists(filepaths['Nodes']):
        url = ('https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx'
               '?messageType=getvectorsource&nodeTypes=GEN,INT,LZN')
        # urllib.request.urlretrieve(url, filepaths['Nodes'])
        r = requests.get(url, allow_redirects=True)
        with open(filepaths['Nodes'], 'wb') as writer:
            writer.write(r.content)
    if not os.path.exists(filepaths['Hubs']):
        url = 'https://api.misoenergy.org/MISORTWD/map/PNODELMPLabels_2.json'
        # urllib.request.urlretrieve(url, filepaths['Hubs'])
        r = requests.get(url, allow_redirects=True)
        with open(filepaths['Hubs'], 'wb') as writer:
            writer.write(r.content)

    ###### GEN, INT, LZN nodes
    ### Load node file and extract names, types, regions, and locations
    with open(filepaths['Nodes']) as f:
        data = json.load(f)

    dfall = pd.io.json.json_normalize(data)
    proj = data['proj']
    df = pd.io.json.json_normalize(data['f'])

    nodenames = [(i[0]) for i in df.p]
    nodetypes = [(i[1]) for i in df.p]
    noderegions = [(i[2]) for i in df.p]
    nodexy = [tuple(df['g.c'][i]) for i in range(len(df.p))]

    ###### Original version - now kills the kernel
    # ### Convert to lat/lon
    # g = gpd.GeoSeries(
    #     [shapely.geometry.Point(nodexy[i]) for i in range(len(nodexy))])
    # g.crs = proj
    # gnode = g.to_crs({'init': 'epsg:4326'})
    ###### New version
    import pyproj
    import shapely.geometry
    latlonproj = pyproj.CRS.from_epsg(4326)
    misoproj = pyproj.CRS(proj)
    transform = pyproj.Transformer.from_crs(
        crs_from=misoproj, crs_to=latlonproj, always_xy=True)
    gnode = gpd.GeoSeries(
        [shapely.geometry.Point(transform.transform(xy[0], xy[1])) 
         for xy in nodexy])

    ### Generate output dataframe
    dfout = pd.DataFrame(gnode)
    dfout['node'] = nodenames
    dfout['latitude'] = gnode.y
    dfout['longitude'] = gnode.x
    dfout['type'] = nodetypes
    dfout['region'] = noderegions
    dfout.drop(0, axis=1, inplace=True)

    ###### HUB nodes
    with open(filepaths['Hubs']) as f:
        data = json.load(f)

    dfall = pd.io.json.json_normalize(data)
    proj = data['proj']
    df = pd.io.json.json_normalize(data['f'])

    hubnames = [(i[0][:-4]) for i in df.p]
    hubxy = [tuple(df['g.c'][i]) for i in range(len(df.p))]

    ###### Original version - now kills the kernel
    # ### Convert to lat/lon
    # g = gpd.GeoSeries(
    #     [shapely.geometry.Point(hubxy[i]) for i in range(len(hubxy))])
    # g.crs = proj
    # ghub = g.to_crs({'init': 'epsg:4326'})
    ###### New version
    latlonproj = pyproj.CRS.from_epsg(4326)
    misoproj = pyproj.CRS(proj)
    transform = pyproj.Transformer.from_crs(
        crs_from=misoproj, crs_to=latlonproj, always_xy=True)
    ghub = gpd.GeoSeries(
        [shapely.geometry.Point(transform.transform(xy[0], xy[1])) 
         for xy in hubxy])

    ### Generate output dataframe
    hubout = pd.DataFrame(ghub)
    hubout['node'] = [i+'.HUB' for i in hubnames]
    hubout['latitude'] = ghub.y
    hubout['longitude'] = ghub.x
    hubout['type'] = 'Hub'
    hubout['region'] = 'MISO'
    hubout.drop(0, axis=1, inplace=True)

    ### Combine and write output dataframes
    dfout = pd.concat([dfout, hubout], ignore_index=True)
    dfout.to_csv(revmpath + 'MISO/io/miso-node-latlon.csv', index=False)
    return dfout

def nodelocations_isone(filepath_input=None):
    """
    File must be requested from ISONE
    """
    if (filepath_input==None) or (os.path.exists(filepath_input)==False):
        print("Example: revmpath + 'ISONE/in/nepnode_lat_long.xlsx'. "
             "File can be requested from ISONE at: "
             "'https://www.iso-ne.com/participate/support/request-information/'")
        raise Exception("Need filename of nepnode_lat_long.xlsx file.")

    ### Load file, rename columns, and write output
    dfin = pd.read_excel(filepath_input, sheet_name='New England')
    dfin.rename(
        columns={'Node Name': 'node', 'LATITUDE': 'latitude', 'LONGITUDE': 'longitude',
                 'RSP Area': 'area', 'Dispatch Zone': 'zone', 'Reserve ID': 'reserveid',
                 'Zone ID': 'zoneid'}, inplace=True)
    dfout = dfin[['node', 'latitude', 'longitude', 'area', 'zone', 'reserveid', 'zoneid']]
    dfout.to_csv(revmpath+'ISONE/io/isone-node-latlon.csv', index=False)
    return dfout

def nodelocations_ercot(filepath_input=None):
    """
    http://www.ercot.com/services/rq/imre
    https://mis.ercot.com/pps/tibco/mis/Pages/Grid+Information/Long+Term+Planning
    'CRR Network Model (Monthly)' > download and unzip one of the available files 
    and use the filepath as the input value for filepath_input.
    """
    ### Functions
    def latlonify(coordinates):
        foo = coordinates.split(' ')
        bar = []
        for line in foo:
            if line not in bar:
                bar.append(line)
        latitude = 0
        longitude = 0
        for line in bar:
            longitude += float(line.split(',')[0])
            latitude += float(line.split(',')[1])
        latitude = latitude / 4
        longitude = longitude / 4

        return latitude, longitude

    ### Load and parse the input kml file
    if (filepath_input==None) or (os.path.exists(filepath_input)==False):
        print("Missing input file. To get this file, register as an IMRE at "
            "http://www.ercot.com/services/rq/imre. Go to "
            "https://mis.ercot.com/pps/tibco/mis/Pages/Grid+Information/Long+Term+Planning, "
            "then select 'CRR Network Model (Monthly)'. "
            "Download and unzip one of the available files and use the filepath "
            "as the input value for filepath_input. "
            "Example: revmpath+'ERCOT/in/rpt.00011205.0000000000000000.20170530"
            ".140432154.JUL2017MonthlyCRRNetworkModel/"
            "2017.JUL.Monthly.Auction.OneLineDiagram.kml'")
        raise Exception("Need path to CRRNetworkModel OneLineDiagram file")
    tree = ET.parse(filepath_input)
    root = tree.getroot()
    buses = root[0][6]

    ### Extract the node names and coordinates
    names, coordinates, latitudes, longitudes = [], [], [], []
    for node in buses.iter(tag='{http://www.opengis.net/kml/2.2}name'):
        if node.text.find(' ') == -1 and node.text != 'Buses':
            names.append(node.text)

    for node in buses.iter(
        tag='{http://www.opengis.net/kml/2.2}coordinates'):
        coordinates.append(node.text)
        
    for node in coordinates:
        latitudes.append(latlonify(node)[0])
        longitudes.append(latlonify(node)[1])

    ### Extract the areas and zones (optional)
    descriptions = []
    for node in buses.iter(
        tag='{http://www.opengis.net/kml/2.2}description'):
        descriptions.append(node.text)

    areas = []
    for i in range(len(descriptions)):
        foo = descriptions[i]
        bar = foo[(foo.find('Area')+14):]
        out = bar[:bar.find('</td>')]
        areas.append(out)

    zones = []
    for i in range(len(descriptions)):
        foo = descriptions[i]
        bar = foo[(foo.find('Zone')+14):]
        out = bar[:bar.find('</td>')]
        zones.append(out)

    settlementzones = []
    for i in range(len(descriptions)):
        foo = descriptions[i]
        bar = foo[(foo.find('Settlement Zone')+25):]
        out = bar[:bar.find('</td>')]
        settlementzones.append(out)

    ### Make the output dataframe
    dfout = pd.DataFrame({
        'node': names, 'latitude': latitudes, 'longitude': longitudes,
        'area': areas, 'zone': zones, 'settlementzone': settlementzones
    })[['node', 'latitude', 'longitude', 'area', 'zone', 'settlementzone']]

    ### Normalize node names
    dfout['node'] = dfout.node.map(lambda x: str(x).strip().upper())

    ### Identify duplicate (lat,lon) tuples for NSRDB
    latlons = dfout[['latitude', 'longitude']].drop_duplicates().copy()
    latlons['latlonindex'] = range(len(latlons))
    dfout = dfout.merge(latlons, on=['latitude', 'longitude'], how='left')

    ### Write outputs
    dfout.to_csv(revmpath+'ERCOT/io/ercot-node-latlon.csv', index=False)
    latlons[['latlonindex', 'latitude', 'longitude']].to_csv(
        revmpath+'ERCOT/io/ercot-node-latlon-unique.csv', index=False)
    return dfout

def nodelocations_nyiso():
    ###### Download input data
    ### Identify urls
    years = range(2005, 2018)

    ### Old version
    # urlbase = ('http://www.nyiso.com/public/webdocs/markets_operations/'
    #            'services/planning/Documents_and_Resources/'
    #            'Planning_Data_and_Reference_Docs/Data_and_Reference_Docs/')

    # urls = {year: urlbase + '{}_NYCA_Generators.xls'.format(year)
    #         for year in years}
    # urls[2012] = urlbase + '2012_NYCA_Generating_Facilities.xls'
    # urls[2015] = urlbase + '2015_NYCA_Generators_Revised.xls'

    base = 'https://www.nyiso.com/documents/20142/1402024/'
    urls = {
        2005: base+'2005_NYCA_Generators.xls/64f2ffcf-7859-714f-dc9c-cca2519f453a',
        2006: base+'2006_NYCA_Generators.xls/bb67c807-9a27-7039-f3ef-8793d3e72cce',
        2007: base+'2007_NYCA_Generators.xls/2e0da2c4-be90-caa3-b201-f211a3f9389b',
        2008: base+'2008_NYCA_Generators.xls/cb944d0f-84c5-0f46-1ac3-424e3cbac850',
        2009: base+'2009_NYCA_Generators.xls/962f951d-03a0-ccff-1296-5bfedadfeee9',
        2010: base+'2010_NYCA_Generators.xls/ede624bb-40f6-6bd6-6fae-664a819b9058',
        2011: base+'2011_NYCA_Generators.xls/432a163b-8860-99f0-2f61-8a54d2a7c74d',
        2012: base+'2012_NYCA_Generating_Facilities.xls/1bb796f7-7221-2787-d164-9fc669c2ef52',
        2013: base+'2013_NYCA_Generators.xls/58f988d4-d72c-510c-ae2f-2afa9b5dc0b3',
        2014: base+'2014_NYCA_Generators.xls/92af4de1-ffc4-69cb-bab6-bac4afcec0ca',
        2015: base+'2015_NYCA_Generators_Revised.xls/b1dfb906-56d6-b245-1c21-9649038050fd',
        2016: base+'2016_NYCA_Generators.xls/b38728a0-0a95-d4e8-3b7b-fe14b4419e89',
        2017: base+'2017_NYCA_Generators.xls/42b3e346-b89c-4284-3457-30bd73e3ea19',
    }

    ### Download files
    for year in years:
        filepathout = revmpath + 'NYISO/in/' + os.path.basename(urls[year])
        urllib.request.urlretrieve(urls[year], filepathout)

    ### Concat input files into clean csv
    filesin = {year: revmpath + 'NYISO/in/' + os.path.basename(urls[year])
               for year in years}

    ### Set columns
    columns = {}
    columns[2005] = [
        'line_ref_number', 'owner_operator_billing_org', 'station_unit',
        'zone', 'ptid', 'town', 'county', 'state',
        'date_in_service',
        'capability_MW_sum', 'capability_MW_win',
        'dual_cogen', 'unit_type', 'FT', 'CS',
        'fuel_type_1', 'fuel_type_2', 'fuel_type_3',
        'net_energy_MWh_prev_year', 'notes',
    ]
    columns[2006] = columns[2005]

    ### 2007, 2008, 2009
    columns[2007] = [
        'line_ref_number', 'owner_operator_billing_org', 'station_unit',
        'zone', 'ptid', 'town', 'county', 'state',
        'date_in_service', 'nameplate_rating_kW',
        'capability_kW_sum', 'capability_kW_win',
        'dual_cogen', 'unit_type', 'FT', 'CS',
        'fuel_type_1', 'fuel_type_2', 'fuel_type_3',
        'net_energy_MWh_prev_year', 'notes',
    ]
    columns[2008], columns[2009] = columns[2007], columns[2007]

    ### 2010, 2011, 2012, 
    columns[2010] = [
        'line_ref_number', 'owner_operator_billing_org', 'station_unit',
        'zone', 'ptid', 'town', 'county', 'state',
        'date_in_service', 'nameplate_rating_MW', 'CRIS_sum_cap_MW',
        'capability_MW_sum', 'capability_MW_win',
        'dual_cogen', 'unit_type', 'FT', 'CS',
        'fuel_type_1', 'fuel_type_2', 'fuel_type_3',
        'net_energy_GWh_prev_year', 'notes',
    ]
    columns[2011], columns[2012] = columns[2010], columns[2010]

    ### 2013, 2014, 2015, 2016, 2017, 
    columns[2013] = [
        'line_ref_number', 'owner_operator_billing_org', 'station_unit',
        'zone', 'ptid', 'town', 'county', 'state',
        'date_in_service', 'nameplate_rating_MW', 'CRIS_sum_cap_MW',
        'capability_MW_sum', 'capability_MW_win',
        'dual_cogen', 'unit_type',
        'fuel_type_1', 'fuel_type_2', 'fuel_type_3',
        'net_energy_GWh_prev_year', 'notes',
    ]
    columns[2014], columns[2015], columns[2016], columns[2017] = (
        columns[2013], columns[2013], columns[2013], columns[2013])

    ### 2018
    # columns[2018] = [
    #     'line_ref_number', 'owner_operator_billing_org', 'station_unit',
    #     'zone', 'ptid', 'town', 'county', 'state',
    #     'date_in_service', 'nameplate_rating_MW',
    #     'CRIS_sum_cap_MW', 'CRIS_win_cap_MW',
    #     'capability_MW_sum', 'capability_MW_win',
    #     'dual_cogen', 'unit_type',
    #     'fuel_type_1', 'fuel_type_2',
    #     'net_energy_GWh_prev_year', 'notes',
    # ]

    ### Set other spreadsheet loading parameters
    skiprows = {
        2005: 6, 2006: 7, 2007: 7, 2008: 7, 2009: 7, 2010: 6, 2011: 6, 
        2012: 6, 2013: 6, 2014: 6, 2015: 6, 2016: 6, 2017: 6, 
    }
    skip_footer = {
        2005: 2, 2006: 2, 2007: 2, 2008: 2, 2009: 1, 2010: 1, 2011: 2, 
        2012: 2, 2013: 2, 2014: 3, 2015: 3, 2016: 3, 2017: 1, 
    }
    sheet_name = {year: 0 for year in years}
    sheet_name[2016] = 'NYCA_2016'
    sheet_name[2017] = 'NYCA_2017'

    ### Load and concat all dataframes
    dfs = {}
    for year in years:
        df = pd.read_excel(
            filesin[year], skiprows=skiprows[year], sheet_name=sheet_name[year], 
            names=columns[year], usecols=len(columns[year])-1, skip_footer=skip_footer[year])
        dfs[year] = df[['ptid', 'town', 'county', 'state']]
        
    dfgens = pd.concat(dfs, axis=0)

    dfgens = (dfgens.reset_index(level=0).rename(columns={'level_0':'year'})
              .reset_index(drop=True).copy())

    ### Define location codes
    statecodes = {36: 'NY', 42: 'PA', 25: 'MA', 34: 'NJ'}
    codestates = {'NY': 36, 'PA': 42, 'MA': 25, 'NJ': 34}

    dfs = {}
    ### NY
    df = pd.read_excel(filesin[2017], sheet_name='Gen_Codes', skiprows=29, 
                       usecols=[2, 3, 4, 5]).dropna()
    dfs['NY'] = pd.DataFrame(df.values.reshape(len(df)*2,2))
    ### PA
    df = pd.read_excel(filesin[2017], sheet_name='Gen_Codes', skiprows=29, 
                       usecols=[7, 8, 9, 10]).dropna()
    dfs['PA'] = pd.DataFrame(df.values.reshape(len(df)*2,2))
    ### MA
    df = pd.read_excel(filesin[2017], sheet_name='Gen_Codes', skiprows=29, 
                              usecols=[13, 14]).dropna()
    dfs['MA'] = pd.DataFrame(df.values.reshape(len(df),2))
    ### NJ
    df = pd.read_excel(filesin[2017], sheet_name='Gen_Codes', skiprows=29, 
                              usecols=[18, 19]).dropna()
    dfs['NJ'] = pd.DataFrame(df.values.reshape(len(df),2))

    codes = pd.concat(dfs).reset_index(level=0).rename(
        columns={'level_0':'state', 0: 'countycode', 1: 'county'})

    codes['statecode'] = codes['state'].map(lambda x: codestates[x])
    codes['countycode'] = codes['countycode'].map(lambda x: x[:3])
    codes['code_state'] = codes['countycode'].astype(str) + '_' + codes['state']
    codes['county_state'] = codes['county'] + ' county, ' + codes['state']
    # codes = codes[['state', 'county', 'county_state', 'code_state']].copy()
    # codes.to_csv(revmpath + 'NYISO/test/io/nyiso-county-codes.csv')

    ###### Determine unique list of locations
    ### Generate nodes dataframe
    dfnodes = dfgens.drop('year',axis=1).drop_duplicates().dropna(subset=['ptid'])

    ### Turn county codes into counties
    countycodes = dict(zip(codes.code_state, codes.county_state))
    def foo(x):
        if type(x) == int:
            return '{:03}'.format(x)
        elif x == '  -':
            return 'nan'
        elif type(x) == str:
            return x
        else:
            return 'nan'
    dfnodes['county'] = dfnodes.county.map(foo)

    ### Clean up dfnodes
    dfnodes = dfnodes.drop(dfnodes.loc[dfnodes.county == 'nan'].index)

    dfnodes['county, state'] = dfnodes.apply(
        lambda row: countycodes['{}_{}'.format(row['county'], statecodes[row['state']])], axis=1)

    ### Correct some typos
    replace = {
        'Gilboa NY': 'Gilboa',
        'Kittanning PA': 'Kittanning',
        'Linden NJ': 'Linden',
        'LyonsFalls': 'Lyons Falls',
        'Mahwah NJ': 'Mahwah NJ',
        'P Jefferson': 'Port Jefferson',
        'SouthHampton': 'Southampton',
        'South Hampton': 'Southampton',
        'Wappingers Falls': 'Wappingers',
        '': 'nan',
    }
    dfnodes['town'] = dfnodes.town.map(lambda x: str(x).strip())
    dfnodes['town'].replace(replace, inplace=True)

    ### For each row:
    ###     * If a town is listed, use the town.
    ###     * If a town is not listed, use the county.
    def foo(row):
        if row['town'] == 'nan':
            return row['county, state']
        else:
            return '{}, {}'.format(row['town'], statecodes[row['state']])
        
    dfnodes['location'] = dfnodes.apply(foo, axis=1)

    ###### Look up locations
    ### Set up googlemaps
    gmaps = googlemaps.Client(key=apikeys['googlemaps'])

    ### Get centers from googlemaps
    numattempts = 200
    sleeptime = 60

    locations = dfnodes[['location']].drop_duplicates().reset_index(drop=True)

    out=[]
    for i, location in enumerate(tqdm(locations.values)):
        attempts = 0
        while attempts < numattempts:
            try:
                location = gmaps.geocode(location)
                lat = location[0]['geometry']['location']['lat']
                lon = location[0]['geometry']['location']['lng']
                out.append([lat, lon])
                time.sleep(0.02)
                break
            except HTTPError as err:
                print('Rebuffed for {} on attempt # {} by "{}".'
                      'Will retry in {} seconds.'.format(
                        location, attempts, err, sleeptime))
                attempts += 1
                time.sleep(sleeptime)

            if attempts >= numattempts:
                raise Exception('Failed on {} after {} attempts'.format(
                    location, attempts))

    dfout = pd.DataFrame(out, columns=['latitude', 'longitude'])
    geo = pd.concat([locations, dfout], axis=1)

    ###### Average all locations for each node
    ### Add lat,lon info to nodes df
    dfnodes = dfnodes.merge(geo, on='location', how='left')
    dfnodes.drop_duplicates(inplace=True)
    dfnodes.drop(dfnodes.loc[dfnodes.ptid.map(lambda x: type(x) != int)].index,
                 inplace=True)
    dfnodes['x'] = np.nan
    dfnodes['y'] = np.nan
    dfnodes['z'] = np.nan

    ptids = list(dfnodes['ptid'].sort_values().unique())

    ### Put lat, lon in cartesian coordinates (assuming spherical Earth)
    dfnodes['x'] = dfnodes.apply(rowlatlon2x, axis=1)
    dfnodes['y'] = dfnodes.apply(rowlatlon2y, axis=1)
    dfnodes['z'] = dfnodes.apply(rowlatlon2z, axis=1)

    ### Determine centroid of locations listed for each node
    lats, lons = [], []
    for i, pnode in enumerate(ptids):
        x = dfnodes[dfnodes['ptid'] == pnode]['x'].mean()
        y = dfnodes[dfnodes['ptid'] == pnode]['y'].mean()
        z = dfnodes[dfnodes['ptid'] == pnode]['z'].mean()

        outlon = math.atan2(y, x) * 180 / math.pi
        rho = math.sqrt(x*x + y*y)
        outlat = math.atan2(z, rho) * 180 / math.pi

        lats.append(outlat)
        lons.append(outlon)

    ### Make output dataframe
    dfout = pd.DataFrame(
        {'node':ptids, 'latitude': lats, 'longitude': lons}
    )[['node', 'latitude', 'longitude']]

    ### Identify duplicate (lat,lon) tuples for NSRDB
    latlons = dfout[['latitude', 'longitude']].drop_duplicates().copy()
    latlons['latlonindex'] = range(len(latlons))
    dfout = dfout.merge(latlons, on=['latitude', 'longitude'], how='left')

    ### Write outputs
    geo.to_csv(revmpath+'NYISO/io/nyiso-locations-latlon.csv', index=False)
    dfout.to_csv(revmpath+'NYISO/io/nyiso-node-latlon.csv', index=False)
    latlons[['latlonindex', 'latitude', 'longitude']].to_csv(
        revmpath+'NYISO/io/nyiso-node-unique-latlons-for-nsrdb.csv', index=False)
    return dfout

def nodelocations(iso, filein=None):
    """
    """
    if iso.upper() == 'CAISO':
        nodelocations_caiso()
    elif iso.upper() == 'ERCOT':
        nodelocations_ercot(filein=filein)
    elif iso.upper() == 'MISO':
        nodelocations_miso()
    elif iso.upper() == 'PJM':
        nodelocations_pjm()
    elif iso.upper() == 'NYISO':
        nodelocations_nyiso()
    elif iso.upper() == 'ISONE':
        nodelocations_isone()


###########################
### DOWNLOAD NSRDB DATA ###

def lonlat2wkt(lon, lat):
    return 'POINT({:+f}{:+f})'.format(lon, lat)

def lonlats2wkt(lonlats):
    out = ['{}%20{}'.format(lonlat[0], lonlat[1]) for lonlat in lonlats]
    return 'MULTIPOINT({})'.format('%2C'.join(out))

def querify(**kwargs):
    out = ['{}={}'.format(key, kwargs[key]) for key in kwargs]
    return '&'.join(out)

def convertattributes_2to3(attributes):
    attributes_2to3 = {
        'surface_air_temperature_nwp': 'air_temperature',
        'surface_pressure_background': 'surface_pressure',
        'surface_relative_humidity_nwp': 'relative_humidity',
        'total_precipitable_water_nwp': 'total_precipitable_water',
        'wind_direction_10m_nwp': 'wind_direction',
        'wind_speed_10m_nwp': 'wind_speed',
    }
    attributes_in = attributes.split(',')
    attributes_out = [attributes_2to3.get(attribute, attribute)
                      for attribute in attributes_in]
    return ','.join(attributes_out)

def convertattributes_3to2(attributes):
    attributes_3to2 = {
        'air_temperature': 'surface_air_temperature_nwp',
        'surface_pressure': 'surface_pressure_background',
        'relative_humidity': 'surface_relative_humidity_nwp',
        'total_precipitable_water': 'total_precipitable_water_nwp',
        'wind_direction': 'wind_direction_10m_nwp',
        'wind_speed': 'wind_speed_10m_nwp',
    }
    attributes_in = attributes.split(',')
    attributes_out = [attributes_3to2.get(attribute, attribute)
                      for attribute in attributes_in]
    return ','.join(attributes_out)

def postNSRDBsize(
    years,
    lonlats,
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed',
    leap_day='true', 
    interval='30',
    norm=False):
    """
    Determine size of NSRDB POST request
    """
    numyears = len(years)
    numattributes = len(attributes.split(','))
    numintervals = sum([pvvm.toolbox.yearhours(year) * 60 / int(interval)
                        for year in years])
    numsites = len(lonlats)
    if norm:
        return numsites * numattributes * numyears * numintervals / 175000000
    return numsites * numattributes * numyears * numintervals


def postNSRDBfiles(
    years, lonlats, psmversion=3,
    api_key=apikeys['nsrdb'],
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed',
    leap_day='true', interval='30', utc='false'):
    """
    """

    ### Set url based on version of PSM
    if psmversion in [2, '2', 2.]:
        url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.json?api_key={}'.format(
            api_key)
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        url = 'http://developer.nrel.gov/api/solar/nsrdb_psm3_download.json?api_key={}'.format(
            api_key)
        attributes = convertattributes_2to3(attributes)
    else:
        raise Exception("Invalid psmversion; must be 2 or 3")

    names = ','.join([str(year) for year in years])
    wkt = lonlats2multipoint(lonlats)

    payload = querify(
        wkt=wkt, attributes=attributes, 
        names=names, utc=utc, leap_day=leap_day, interval=interval, 
        full_name=nsrdbparams['full_name'], email=nsrdbparams['email'],
        affiliation=nsrdbparams['affiliation'], reason=nsrdbparams['reason'],
        mailing_list=nsrdbparams['mailing_list']
    )
            
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }
    
    response = requests.request("POST", url, data=payload, headers=headers)
    
    output = response.text
    print(output[output.find("errors"):output.find("inputs")], 
      '\n', output[output.find("outputs"):])

def downloadNSRDBfile(
    lat, lon, year, filepath=None,
    nodename='default', filetype='.gz',
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed', 
    leap_day='true', interval='30', utc='false', psmversion=3,
    write=True, return_savename=False, urlonly=False):
    '''
    Downloads file from NSRDB.
    NOTE: PSM v2 doesn't include 'surface_albedo' attribute.
    NOTE: For PSM v3, can use either v2 or v3 version of attribute labels.
    
    Full list of attributes for PSM v2:
    attributes=(
        'dhi,dni,ghi,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,' + 
        'dew_point,surface_air_temperature_nwp,surface_pressure_background,' +
        'surface_relative_humidity_nwp,solar_zenith_angle,' +
        'total_precipitable_water_nwp,wind_direction_10m_nwp,' +
        'wind_speed_10m_nwp,fill_flag')

    Full list of attributes for PSM v3:
    attributes=(
        'dhi,dni,ghi,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,' + 
        'dew_point,air_temperature,surface_pressure,' +
        'relative_humidity,solar_zenith_angle,' +
        'total_precipitable_water,wind_direction,' +
        'wind_speed,fill_flag,surface_albedo')

    Parameters
    ----------
    filename: string
    nodename: string
    lat: numeric
    lon: numeric
    year: numeric
    
    Returns
    -------
    if write == True: # default
        '.csv' file if filetype == '.csv', or '.gz' file if filetype == '.gz'
    if return_savename == False: pandas.DataFrame # default
    if return_savename == True: (pandas.DataFrame, savename) # type(savename) = str
    '''
    ### Check inputs
    if filetype not in ['.csv', '.gz']:
        raise Exception("filetype must be '.csv' or '.gz'.")

    if write not in [True, False]:
        raise Exception('write must be True or False.')

    if return_savename not in [True, False]:
        raise Exception('return_savename must be True or False.')

    ### Set psmversion to 3 if year is 2016 (since v2 doesn't have 2016)
    if year in [2016, '2016', 2016.]:
        psmversion = 3

    ### Remove solar_zenith_angle if year == 'tmy'
    if year == 'tmy':
        attributes = attributes.replace('solar_zenith_angle,','')
        attributes = attributes.replace('solar_zenith_angle','')

    year = str(year)

    ### Set url based on version of PSM
    if psmversion in [2, '2', 2.]:
        urlbase = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?'
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        urlbase = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?'
        attributes = convertattributes_2to3(attributes)
    else:
        raise Exception("Invalid psmversion; must be 2 or 3")
        
    url = (
        urlbase + querify(
            api_key=apikeys['nsrdb'], full_name=nsrdbparams['full_name'], 
            email=nsrdbparams['email'], affiliation=nsrdbparams['affiliation'], 
            reason=nsrdbparams['reason'], mailing_list=nsrdbparams['mailing_list'],
            wkt=lonlat2wkt(lon, lat), names=year, attributes=attributes,
            leap_day=leap_day, utc=utc, interval=interval))
    if urlonly:
        return url
    
    try:
        df = pd.read_csv(url)
    except HTTPError as err:
        print(url)
        print(err)
        raise HTTPError
    df = df.fillna('')
    columns = df.columns

    if write == True:
        if len(filepath) != 0 and filepath[-1] != '/':
            filepath = filepath + '/'

        if nodename in [None, 'default']:
            savename = (filepath + df.loc[0,'Location ID'] + '_' + 
                df.loc[0,'Latitude'] + '_' + 
                df.loc[0,'Longitude'] + '_' + year + filetype)
        else:
            # savename = str(filepath + nodename + '-' + year + filetype)
            savename = os.path.join(
                filepath, '{}-{}{}'.format(nodename, year, filetype))

        ### Determine number of columns to write (used to always write 11)
        numcols = max(len(attributes.split(','))+5, 11)

        ### Write the output
        if filetype == '.gz':
            df.to_csv(savename, columns=columns[0:numcols], index=False, 
                      compression='gzip')
        elif filetype == '.csv':
            df.to_csv(savename, columns=columns[0:numcols], index=False)
    
        if return_savename == True:
            return df, savename
        else:
            return df
    return df


def downloadNSRDBfiles(
    dfin, years, nsrdbpath, namecolumn=None,
    resolution=None, latlonlabels=None,
    filetype='.gz', psmversion=3,
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed',
    wait=0.5, maxattempts=200,):
    """
    """
    ###### Set defaults
    ### Convert attributes if necessary
    if psmversion in [2, '2', 2.]:
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        attributes = convertattributes_2to3(attributes)

    ### Get lat, lon labels
    if ('latitude' in dfin.columns) and ('longitude' in dfin.columns):
        latlabel, lonlabel = 'latitude', 'longitude'
    elif ('Latitude' in dfin.columns) and ('Longitude' in dfin.columns):
        latlabel, lonlabel = 'Latitude', 'Longitude'
    elif ('lat' in dfin.columns) and ('lon' in dfin.columns):
        latlabel, lonlabel = 'lat', 'lon'
    elif ('lat' in dfin.columns) and ('long' in dfin.columns):
        latlabel, lonlabel = 'lat', 'long'
    elif ('x' in dfin.columns) and ('y' in dfin.columns):
        latlabel, lonlabel = 'x', 'y'
    else:
        latlabel, lonlabel = latlonlabels[0], latlonlabels[1]

    ### Loop over years
    for year in years:
        ### Set defaults
        if (resolution == None) and (year == 'tmy'):
            resolution = 60
        elif (resolution == None) and (type(year) == int):
            resolution = 30

        ### Set up output folder
        outpath = nsrdbpath+'{}/{}min/'.format(year, resolution)
        os.makedirs(outpath, exist_ok=True)

        ### Make list of files downloaded so far
        downloaded = glob(outpath + '*') ## or os.listdir(outpath)
        downloaded = [os.path.basename(file) for file in downloaded]

        ### Make list of files to download
        if 'latlonindex' in dfin.columns:
            dfin.drop_duplicates('latlonindex', inplace=True)
            dfin['name'] = dfin['latlonindex'].copy()
            dfin['file'] = dfin['latlonindex'].map(
                lambda x: '{}{}-{}{}'.format(outpath, x, year, filetype))
        elif namecolumn is not None:
            dfin['name'] = dfin[namecolumn].copy()
            dfin['file'] = dfin[namecolumn].map(
                lambda x: '{}{}-{}{}'.format(outpath, x, year, filetype))
        elif namecolumn is None:
            dfin['name'] = None
            dfin['file'] = None

        dfin['todownload'] = dfin['file'].map(
            lambda x: os.path.basename(x) not in downloaded)

        dftodownload = dfin[dfin['todownload']].reset_index(drop=True)

        print('{}: {} done, {} to download'.format(
            year, len(downloaded), len(dftodownload)))

        ### Loop over locations
        for i in trange(len(dftodownload)):
            attempts = 0
            while attempts < maxattempts:
                try:
                    downloadNSRDBfile(
                        lat=dftodownload[latlabel][i],
                        lon=dftodownload[lonlabel][i],
                        year=year,
                        filepath=outpath,
                        nodename=dftodownload['name'][i],
                        interval=str(resolution),
                        psmversion=psmversion,
                        attributes=attributes)
                    break
                except HTTPError as err:
                    if str(err) in ['HTTP Error 504: Gateway Time-out', 
                                    'HTTP Error 500: Internal Server Error']:
                        print(('Rebuffed on attempt # {} at {} by "{}". '
                               'Retry in 5 minutes.').format(
                                    attempts, pvvm.toolbox.nowtime(), err))
                        attempts += 1
                        time.sleep(5 * 60)
                    else:
                        print(('Rebuffed on attempt # {} at {} by "{}". '
                               'Retry in {} hours.').format(
                                    attempts, pvvm.toolbox.nowtime(), err, wait))
                        attempts += 1
                        time.sleep(wait * 60 * 60)                  
            if attempts >= maxattempts:
                print("Something must be wrong. No response after {} attempts.".format(
                    attempts))


def downloadNSRDBfiles_iso(year, resolution=None, 
    isos=['CAISO', 'ERCOT', 'MISO', 'NYISO', 'PJM', 'ISONE'],
    filetype='.gz', wait=0.5, psmversion=3, 
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed'):
    """
    """
    # nodemap = {
    #     'CAISO': os.path.join(revmpath, 'CAISO/io/caiso-node-latlon.csv'),
    #     'ERCOT': os.path.join(revmpath, 'ERCOT/io/ercot-node-latlon.csv'),
    #     'MISO':  os.path.join(revmpath, 'MISO/in/miso-node-map.csv'),
    #     'PJM':   os.path.join(revmpath, 'PJM/io/pjm-pnode-latlon-uniquepoints.csv'),
    #     'NYISO': os.path.join(revmpath, 'NYISO/io/nyiso-node-latlon.csv'),
    #     'ISONE': os.path.join(revmpath, 'ISONE/io/isone-node-latlon.csv') 
    # }[iso]
    ### Set defaults
    if (resolution == None) and (year == 'tmy'):
        resolution = 60
    elif (resolution == None) and (type(year) == int):
        resolution = 30

    ### Convert attributes if necessary
    if psmversion in [2, '2', 2.]:
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        attributes = convertattributes_2to3(attributes)

    for iso in isos:
        nodemap = revmpath + '{}/io/{}-node-latlon.csv'.format(
            iso.upper(), iso.lower())
        ### Load node key
        dfin = pd.read_csv(nodemap)
        dfin.rename(
            columns={'name': 'node', 'pnodename': 'node', 'ptid': 'node'},
            inplace=True)

        ### Set up output folder
        outpath = os.path.join(revmpath, '{}/in/NSRDB/{}/{}min/'.format(
            iso, year, resolution))
        if not os.path.isdir(outpath):
            os.makedirs(outpath)

        ### Make list of files downloaded so far
        downloaded = glob(outpath + '*') ## or os.listdir(outpath)
        downloaded = [os.path.basename(file) for file in downloaded]

        ### Make list of files to download
        if 'latlonindex' in dfin.columns:
            dfin.drop_duplicates('latlonindex', inplace=True)
            dfin['name'] = dfin['latlonindex'].copy()
            dfin['file'] = dfin['latlonindex'].map(
                lambda x: '{}{}-{}{}'.format(outpath, x, year, filetype))
        else:
            dfin['name'] = dfin['node'].copy()
            dfin['file'] = dfin['node'].map(
                lambda x: '{}{}-{}{}'.format(outpath, x, year, filetype))

        dfin['todownload'] = dfin['file'].map(
            lambda x: os.path.basename(x) not in downloaded)

        dftodownload = dfin[dfin['todownload']].reset_index(drop=True)

        print('{} {}: {} done, {} to download'.format(
            iso.upper(), year, len(downloaded), len(dftodownload)))

        for i in trange(len(dftodownload)):
            attempts = 0
            while attempts < 200:
                try:
                    downloadNSRDBfile(
                        lat=dftodownload['latitude'][i],
                        lon=dftodownload['longitude'][i],
                        year=year,
                        filepath=outpath,
                        nodename=str(dftodownload['name'][i]),
                        interval=str(resolution),
                        psmversion=psmversion,
                        attributes=attributes)
                    break
                except HTTPError as err:
                    if str(err) in ['HTTP Error 504: Gateway Time-out', 
                                    'HTTP Error 500: Internal Server Error']:
                        print(('Rebuffed on attempt # {} at {} by "{}". '
                               'Retry in 5 minutes.').format(
                                    attempts, pvvm.toolbox.nowtime(), err))
                        attempts += 1
                        time.sleep(5 * 60)
                    else:
                        print(('Rebuffed on attempt # {} at {} by "{}". '
                               'Retry in {} hours.').format(
                                    attempts, pvvm.toolbox.nowtime(), err, wait))
                        attempts += 1
                        time.sleep(wait * 60 * 60)                  
            if attempts >= 200:
                print("Something must be wrong. No response after {} attempts.".format(
                    attempts))

def postNSRDBfiles_iso(year, yearkey, resolution=None, 
    isos=['CAISO', 'ERCOT', 'MISO', 'NYISO', 'PJM', 'ISONE'],
    filetype='.gz', wait=3, psmversion=2, chunksize=1000,
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed'):
    """
    Notes
    -----
    * This function can only be run after all NSRDB files for a given year have been
    downloaded using downloadNSRDBfiles(), as the POST request scrambles the 
    node-to-NSRDBid correspondence.
    * The files will be sent to settings.nsrdbparams['email']. Need to download and unzip them.
    Default location for unzipped files is revmpath+'USA/in/NSRDB/nodes/{}/'.format(year).
    """
    # nodemap = {
    #     'CAISO': os.path.join(revmpath, 'CAISO/io/caiso-node-latlon.csv'),
    #     'ERCOT': os.path.join(revmpath, 'ERCOT/io/ercot-node-latlon.csv'),
    #     'MISO':  os.path.join(revmpath, 'MISO/in/miso-node-map.csv'),
    #     'PJM':   os.path.join(revmpath, 'PJM/io/pjm-pnode-latlon-uniquepoints.csv'),
    #     'NYISO': os.path.join(revmpath, 'NYISO/io/nyiso-node-latlon.csv'),
    #     'ISONE': os.path.join(revmpath, 'ISONE/io/isone-node-latlon.csv') 
    # }[iso]
    ### Set defaults
    if (resolution == None) and (year == 'tmy'):
        resolution = 60
    elif (resolution == None) and (type(year) == int):
        resolution = 30

    ### Convert attributes if necessary
    if psmversion in [2, '2', 2.]:
        attributes = convertattributes_3to2(attributes)
    elif psmversion in [3, '3', 3.]:
        attributes = convertattributes_2to3(attributes)

    ### Make dataframe of nodes from all ISOs
    dictnodes = {}
    for iso in isos:
        ### Load node key
        nodemap = revmpath + '{}/io/{}-node-latlon.csv'.format(
            iso.upper(), iso.lower())
        dfin = pd.read_csv(nodemap)
        dfin.rename(
            columns={'name': 'node', 'pnodename': 'node', 'ptid': 'node'},
            inplace=True)
        
        inpath = os.path.join(revmpath, '{}/in/NSRDB/{}/{}min/'.format(
            iso, yearkey, resolution))
        
        ### Make list of files downloaded so far
        downloaded = glob(inpath + '*') ## or os.listdir(inpath)

        ### Make list of files to download
        if 'latlonindex' in dfin.columns:
            dfin.drop_duplicates('latlonindex', inplace=True)
            dfin['name'] = dfin['latlonindex'].copy()
            dfin['file'] = dfin['latlonindex'].map(
                lambda x: '{}{}-{}{}'.format(inpath, x, yearkey, filetype))
        else:
            dfin['name'] = dfin['node'].copy()
            dfin['file'] = dfin['node'].map(
                lambda x: '{}{}-{}{}'.format(inpath, x, yearkey, filetype))

        dfin['todownload'] = dfin['file'].map(
            lambda x: x not in downloaded)

        dictnodes[iso] = dfin.copy()

    dfnodes = pd.concat(dictnodes)

    ### Identify locations to include in query
    nsrdbids, nsrdblats, nsrdblons = [], [], []

    for file in tqdm(dfnodes['file'].values):
        df = pd.read_csv(file, nrows=1)
        nsrdbids.append(df['Location ID'][0])
        nsrdblats.append(df['Latitude'][0])
        nsrdblons.append(df['Longitude'][0])

    dfnodes['NSRDBid'] = nsrdbids
    dfnodes['NSRDBlat'] = nsrdblats
    dfnodes['NSRDBlon'] = nsrdblons

    dfnodes = dfnodes.reset_index(level=0).rename(columns={'level_0': 'iso'})
    dfnodes.reset_index(drop=True, inplace=True)

    ### Save dfnodes for use in unpacking
    if not os.path.exists(revmpath+'USA/io/'):
        os.makedirs(revmpath+'USA/io/')
    dfnodes.to_csv(revmpath+'USA/io/nsrdbnodekey-{}.csv'.format(yearkey), index=False)

    ### Post NSRDB requests in 400-unit chunks, dropping duplicate NSRDBids
    dftodownload = dfnodes.drop_duplicates('NSRDBid').copy()
    lonlatstodownload = list(zip(dftodownload['NSRDBlon'], dftodownload['NSRDBlat']))
    for i in range(0,len(lonlatstodownload), chunksize):
        print(i)
        postNSRDBfiles(years=[year], lonlats=lonlatstodownload[i:i+chunksize],
                       psmversion=psmversion, attributes=attributes)
        time.sleep(wait)

def unpackpostNSRDBfiles_iso(year, yearkey, postpath=None,
    isos=['CAISO', 'ERCOT', 'MISO', 'NYISO', 'PJM', 'ISONE'],
    resolution=None, filetypeout='.gz',
    attributes='ghi,dni,dhi,solar_zenith_angle,air_temperature,wind_speed'):
    """
    Notes
    -----
    * This function can only be run after postNSRDBfiles_iso().
    * Default location for unzipped posted files is 
    revmpath+'USA/in/NSRDB/nodes/{}/'.format(year).
    * Defualt location for dfnodes is 
    revmpath+'USA/io/nsrdbnodekey-{}.csv'.format(yearkey)
    """
    ### Set defaults, if necessary
    if postpath==None:
        postpath = revmpath+'USA/in/NSRDB/nodes/{}/'.format(year)
    if (resolution == None) and (year == 'tmy'):
        resolution = 60
    elif (resolution == None) and (type(year) == int):
        resolution = 30

    compression = 'gzip'
    if filetypeout not in ['gzip', '.gz']:
        compression = None

    ### Load dfnodes from default location
    dfnodes = pd.read_csv(revmpath+'USA/io/nsrdbnodekey-{}.csv'.format(yearkey))

    ### Get downloaded file list
    postfiles = glob(postpath + '*')

    ### Extract parameters from filename
    def fileparams(filepath, filetype='.csv'):
        filename = os.path.basename(filepath)
        nsrdbid = filename[:filename.find('_')]
        lat = filename[filename.find('_')+1:filename.find('_-')]
        lon = filename[filename.find('_-')+1:filename.find(filetype)][:-5]
        year = filename[-(len(filetype)+4):-len(filetype)]
        return nsrdbid, lat, lon, year

    dfpostfiles = pd.DataFrame(postfiles, columns=['filepath'])
    dfpostfiles['nsrdbid'] = dfpostfiles.filepath.apply(lambda x: fileparams(x)[0])
    dfpostfiles['lat'] = dfpostfiles.filepath.apply(lambda x: fileparams(x)[1])
    dfpostfiles['lon'] = dfpostfiles.filepath.apply(lambda x: fileparams(x)[2])
    dfpostfiles['year'] = dfpostfiles.filepath.apply(lambda x: fileparams(x)[3])

    latlon2postfilepath = dict(zip(
        [tuple(row) for row in dfpostfiles[['lat', 'lon']].values],
        list(dfpostfiles['filepath'].values)
    ))

    ### Add filepath for POST files to dfnodes
    dfnodes['NSRDBfile'] = dfnodes.apply(
        lambda row: latlon2postfilepath[(str(row['NSRDBlat']), str(row['NSRDBlon']))],
        axis=1)
    dfnodes['NSRDBfilebase'] = dfnodes.NSRDBfile.map(os.path.basename)

    ### Create output file folders
    for iso in isos:
        pathout = revmpath+'{}/in/NSRDB/{}/{}min/'.format(iso, year, resolution)
        if not os.path.isdir(pathout):
            os.makedirs(pathout)

    ### Save corresponding files with correct nodenames
    for i in trange(len(dfnodes)):
        ### Set filenames
        fileout = os.path.join(
            revmpath,
            '{}/in/NSRDB/{}/{}min/'.format(dfnodes.loc[i,'iso'], year, resolution),
            '{}-{}{}'.format(dfnodes.loc[i, 'name'], year, filetypeout)
        )
        filein = dfnodes.loc[i,'NSRDBfile']
        
        ### Load, manipulate, and resave file
        dfin = pd.read_csv(filein, low_memory=False)
        df = dfin.fillna('')
        
        columns = df.columns
        numcols = max(len(attributes.split(','))+5, 11)
        
        df.to_csv(fileout, 
                  columns=columns[:numcols],
                  index=False, compression=compression)


######################
### BUILT-IN FILES ###

def makefolders():
    """
    Create the file system
    """
    os.makedirs(revmpath, exist_ok=True)
    os.makedirs(datapath, exist_ok=True)
    for iso in isos:
        os.makedirs(revmpath+'{}/in/'.format(iso), exist_ok=True)
        os.makedirs(revmpath+'{}/io/'.format(iso), exist_ok=True)
    os.makedirs(revmpath+'io/', exist_ok=True)
    os.makedirs(revmpath+'out/', exist_ok=True)
    os.makedirs(revmpath+'figs/', exist_ok=True)
    os.makedirs(revmpath+'Validation/', exist_ok=True)

def copyfiles():
    """
    Copy the input files to the appropriate directory
    Notes
    -----
    * results files are in USD2017. ALREADY CORRECTED FOR INLATION.
    """
    dirmodule = os.path.dirname(os.path.abspath(__file__)) + '/'

    ### Make the data folders
    os.makedirs(datapath+'BLS/', exist_ok=True)
    os.makedirs(datapath+'CEDM/', exist_ok=True)
    os.makedirs(datapath+'California/', exist_ok=True)
    os.makedirs(datapath+'RGGI/', exist_ok=True)

    ###### Copy the files
    ### BLS (inflation)
    shutil.copy(dirmodule+'data/BLS/inflation_cpi_level.xlsx',
                datapath+'BLS/')
    ### California (CO2 price)
    shutil.copy(dirmodule+'data/California/carb-auction-settlements.xlsx',
                datapath+'California/')
    ### RGGI (CO2 price)
    shutil.copy(dirmodule+'data/RGGI/AllowancePricesAndVolumes.csv',
                datapath+'RGGI/')
    ### Core results
    os.makedirs(revmpath+'out/', exist_ok=True)
    files = glob(dirmodule+'data/results/*')
    for file in files:
        shutil.copy(file, revmpath+'out/')
    ### ISO Load
    zip_ref = zipfile.ZipFile(dirmodule+'data/ISO.zip')
    zip_ref.extractall(datapath)
    zip_ref.close()
    ### ISO capacity prices
    os.makedirs(datapath+'ISO/', exist_ok=True)
    shutil.copy(dirmodule+'data/ISO/capacity-prices.xlsx',
                datapath+'ISO/')
    ### Wind
    for iso in ['ERCOT','MISO','PJM','ISONE']:
        outpath = datapath+'ISO/{}/wind/'.format(iso)
        os.makedirs(outpath, exist_ok=True)
        files = glob(dirmodule+'data/wind_raw/{}/*'.format(iso))
        for file in files:
            shutil.copy(file, outpath)


def copylmps(iso=None, market=None, year=None, product='lmp', test=False):
    """
    Copy the nodalized LMP files to the appropriate directory
    """
    dirmodule = os.path.dirname(os.path.abspath(__file__)) + '/'
    inpath = dirmodule + 'data/lmp/'

    isos = ['CAISO', 'ERCOT', 'MISO', 'PJM', 'NYISO', 'ISONE']
    markets = ['da', 'rt']
    yeardict = {
        ('CAISO', 'da'): range(2010,2018), ('CAISO', 'rt'): range(2010,2018),
        ('ERCOT', 'da'): range(2011,2018), ('ERCOT', 'rt'): range(2011,2018),
        ('MISO',  'da'): range(2010,2018), ('MISO',  'rt'): range(2010,2018),
        ('PJM',   'da'): range(2010,2018), ('PJM',   'rt'): range(2010,2018),
        ('NYISO', 'da'): range(2010,2018), ('NYISO', 'rt'): range(2010,2018),
        ('ISONE', 'da'): range(2010,2018), ('ISONE', 'rt'): range(2011,2018),
    }

    ### Select single iso/market/year if desired
    if iso is not None:
        isos = [iso]
    if market is not None:
        markets = [market]

    ### Write the single-node files
    for market in markets:
        for iso in isos:
            years = yeardict[(iso, market)]
            if year is not None:
                years = [year]
            for i_year in years:
                df = pvvm.io.getdflmp(iso, market, i_year).dropna(axis=1)

                ### Write the single-node files
                if test == True:
                    iterable = df.columns[:5]
                else:
                    iterable = df.columns

                for column in tqdm(iterable, leave=True,
                                   desc='{} {} {}'.format(iso, market, i_year)):
                    filepath = revmpath+'{}/io/lmp-nodal/{}/'.format(
                        iso.upper(), market)
                    os.makedirs(filepath, exist_ok=True)
                    df[[column]].to_csv(
                        filepath+'{}-{}.gz'.format(column, i_year),
                        header=False, compression='gzip')

                ### Write the full-time files
                fulltimeoutpath = revmpath+'{}/io/fulltimenodes/'.format(iso)
                os.makedirs(fulltimeoutpath, exist_ok=True)
                dsout = pd.Series(df.columns)
                dsout.to_csv(
                    fulltimeoutpath+'{}-{}lmp-fulltime-{}.csv'.format(
                        iso.lower(), market, i_year),
                    index=False)

                ### Write the average files
                meanoutpath = revmpath+iso+'/io/lmp-nodal-mean/'
                os.makedirs(meanoutpath, exist_ok=True)
                df.mean(axis=1).to_csv(
                    meanoutpath+'lmp-{}-{}-mean.csv'.format(market, year))
                df.median(axis=1).to_csv(
                    meanoutpath+'lmp-{}-{}-median.csv'.format(market, year))
                percentiles = [0.005, 0.01, 0.025, 0.1, 0.25, 0.5, 
                               0.75, 0.9, 0.975, 0.99, 0.995]
                df.T.describe(percentiles=percentiles).T.to_csv(
                    meanoutpath+'lmp-{}-{}-percentiles.csv'.format(market, year))

# if __name__ == '__main__':
#     copyfiles()
#     copylmps()
#     for iso in ['CAISO', 'MISO', 'PJM', 'NYISO']:
#         nodelocations(iso)