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
import sys, os, site, zipfile, math, time, json, pickle
from glob import glob
from tqdm import tqdm, trange
import scipy, scipy.optimize

import pvlib

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
import pvvm.data
import pvvm.io

############################
### TIME SERIES MANIPULATION

def dropleap(dfin, year, resolution):
    """
    """
    assert len(dfin) % 8784 == 0, "dfin must be a leap year"
    leapstart = pd.Timestamp('{}-02-29 00:00'.format(year))
    dfout = dfin.drop(dfin.loc[
        leapstart
        :(leapstart
          + pd.Timedelta('1D')
          - pd.Timedelta('{}min'.format(resolution)))].index)
    return dfout

def downsample_trapezoid(dfin, output_freq='H', clip=None):
    """
    """
    ### Get label
    if type(dfin) is pd.DataFrame:
        columns = dfin.columns
    elif type(dfin) is pd.Series:
        columns = [dfin.name]
    
    ### Perform trapezoidal integration
    dfin_hours = dfin.iloc[0::2].copy()
    dfin_halfhours = dfin.iloc[1::2].copy()
    
    dfout = pd.DataFrame(
        (dfin_halfhours.values 
         + dfin_hours.rolling(2).mean().shift(-1).fillna(0).values),
        index=dfin_hours.index, columns=columns) / 2
    
    dfout.index.freq = output_freq
    
    ### Clip it, to get rid of small float errors
    if clip is not None:
        dfout = dfout.where(dfout > clip, 0)

    ### Convert back to series if necessary
    if type(dfin) is pd.Series:
        dfout = dfout[dfin.name]
    
    return dfout


def timeserieslineup(
    series1, series2, 
    resolution1=None, resolution2=None,
    tz1=None, tz2=None,
    resamplemethod='ffill', 
    tzout='left', yearout=None,
    mismatch='raise',
    resampledirection='up', clip=None,
    oneyear=True,
    ):
    """
    Inputs
    ------
    * resolution: int [frequency in minutes] or pandas-readable string
    * cliptimezones: in 'left', 'right', [False, 'none', 'neither', None]
    Notes
    -----
    * Both timeseries need to be one year long, and span Jan1 - Dec31
    * Both timeseries need to have timezone information
    * Updated version (20190828) to allow specification of single-year
      or multi-year operation. If oneyear==False, it won't correct 
      leap years.
    """
    ### Make copies
    ds1, ds2 = series1.copy(), series2.copy()
    ### Make sure the time series are well-formatted
    # assert (ds1.index[0].month == 1 and ds1.index[0].day == 1)
    if not (ds1.index[0].month == 1 and ds1.index[0].day == 1):
        print(ds1.head())
        print(ds1.tail())
        print('len(series1) = {}'.format(len(ds1)))
        raise Exception('series1 is not well-formatted')
    # assert (ds2.index[0].month == 1 and ds2.index[0].day == 1)
    if not (ds2.index[0].month == 1 and ds2.index[0].day == 1):
        print(ds2.head())
        print(ds2.tail())
        print('len(series2) = {}'.format(len(ds2)))
        raise Exception('series2 is not well-formatted')
    
    ### Get properties
    tz1, tz2 = ds1.index.tz, ds2.index.tz
    name1, name2 = ds1.name, ds2.name
    if name1 == None: name1 = 0
    if name2 == None: name2 = 0
            
    ### Determine resolutions if not entered
    freq2resolution = {'H': 60, '1H': 60, '60T': 60, 
                       '30T': 30, '15T': 15, 
                       '5T': 5, '1T': 1, 'T': 1,
                       '<Hour>': 60,
                       '<60 * Minutes>': 60, 
                       '<30 * Minutes>': 30, 
                       '<15 * Minutes>': 15, 
                       '<5 * Minutes>': 5, 
                       '<Minute>': 1,
                      }
    
    if resolution1 == None:
        resolution1 = str(ds1.index.freq)
        resolution1 = freq2resolution[resolution1]
    else:
        resolution1 = freq2resolution.get(resolution1, resolution1)
        
    if resolution2 == None:
        resolution2 = str(ds2.index.freq)
        resolution2 = freq2resolution[resolution2]
    else:
        resolution2 = freq2resolution.get(resolution2, resolution2)
        
    ### Get timezones if not entered
    tz1 = ds1.index.tz.__str__()
    tz2 = ds2.index.tz.__str__()
    if ((tz1 == 'None') and (tz2 != 'None')) or ((tz2 == 'None') and (tz1 != 'None')):
        print('tz1 = {}\ntz2 = {}'.format(tz1,tz2))
        raise Exception("Can't align series when one is tz-naive and one is tz-aware")
        
    ### Check if it's tmy, and if so, convert to 2001
    if oneyear:
        ## ds1
        if len(ds1.index.map(lambda x: x.year).unique()) != 1:
            ds1.index = pd.date_range(
                '2001-01-01', '2002-01-01',
                freq='{}T'.format(resolution1), tz=tz1, closed='left')
            year1 = 2001
        else:
            year1 = ds1.index[0].year
        ## ds2
        if len(ds2.index.map(lambda x: x.year).unique()) != 1:
            ds2.index = pd.date_range(
                '2001-01-01', '2002-01-01',
                freq='{}T'.format(resolution2), tz=tz2, closed='left')
            year2 = 2001
        else:
            year2 = ds2.index[0].year
    
    ###### Either upsample or downsample
    if resampledirection.startswith('up'):
        ###### Upsample the lower-resolution dataset, if necessary
        resolution = min(resolution1, resolution2)
        
        ## Upsample ds2
        if resolution1 < resolution2:
            ## Extend past the last value, for interpolation
            # ds2.loc[ds2.index.max() + 1] = ds2.iloc[-1]
            ds2 = ds2.append(
                pd.Series(data=ds2.iloc[-1],
                          index=[ds2.index[-1] + pd.Timedelta(resolution2, 'm')]))
            ## Interpolate
            if resamplemethod in ['ffill', 'forward', 'pad']:
                ds2 = ds2.resample('{}T'.format(resolution1)).ffill()
            elif resamplemethod in ['interpolate', 'time']:
                ### NOTE that this still just ffills the last value
                ds2 = ds2.resample('{}T'.format(resolution1)).interpolate('time')
            else:
                raise Exception("Unsupported resamplemethod: {}".format(resamplemethod))
            ## Drop the extended value
            ds2.drop(ds2.index[-1], inplace=True)
        ## Upsample ds1
        elif resolution1 > resolution2:
            ## Extend past the last value, for interpolation
            # ds1.loc[ds1.index.max() + 1] = ds1.iloc[-1]
            ds1 = ds1.append(
                pd.Series(data=ds1.iloc[-1],
                          index=[ds1.index[-1] + pd.Timedelta(resolution1, 'm')]))
            ## Interpolate
            if resamplemethod in ['ffill', 'forward', 'pad']:
                ds1 = ds1.resample('{}T'.format(resolution2)).ffill()
            elif resamplemethod in ['interpolate', 'time']:
                ### NOTE that this still just ffills the last value
                ds1 = ds1.resample('{}T'.format(resolution2)).interpolate('time')
            else:
                raise Exception("Unsupported resamplemethod: {}".format(resamplemethod))
            ## Drop the extended value
            ds1.drop(ds1.index[-1], inplace=True)
    elif resampledirection.startswith('down'):
        ### Only works for 2x difference in frequency. Check to make sure.
        resmin = min(resolution1, resolution2)
        resmax = max(resolution1, resolution2)
        assert ((resmax % resmin == 0) & (resmax // resmin == 2))
        ### Resample ds1 if it has finer resolution than ds2
        if resolution1 < resolution2:
            ds1 = downsample_trapezoid(
                dfin=ds1, output_freq='{}T'.format(resolution2), clip=clip)
        ### Resample ds2 if it has finer resolution than ds1
        elif resolution2 < resolution1:
            ds2 = downsample_trapezoid(
                dfin=ds2, output_freq='{}T'.format(resolution1), clip=clip)
        
    ### Drop leap days if ds1 and ds2 have different year lengths
    if oneyear:
        year1hours, year2hours = pvvm.toolbox.yearhours(year1), pvvm.toolbox.yearhours(year2)
        if (year1hours == 8784) and (year2hours == 8760):
            # ds1 = dropleap(ds1, year1, resolution1)
            ds1 = dropleap(ds1, year1, resolution)

        elif (year1hours == 8760) and (year2hours == 8784):
            # ds2 = dropleap(ds2, year2, resolution2)
            ds2 = dropleap(ds2, year2, resolution)
    
    ### Check for errors
    if len(ds1) != len(ds2):
        if mismatch == 'raise':
            print('Lengths: {}, {}'.format(len(ds1), len(ds2)))
            print('Resolutions: {}, {}'.format(resolution1, resolution2))
            print(ds1.head(3))
            print(ds1.tail(3))
            print(ds2.head(3))
            print(ds2.tail(3))
            raise Exception('Mismatched lengths')
        elif mismatch == 'verbose':
            print('Lengths: {}, {}'.format(len(ds1), len(ds2)))
            print('Resolutions: {}, {}'.format(resolution1, resolution2))
            print(ds1.head(3))
            print(ds1.tail(3))
            print(ds2.head(3))
            print(ds2.tail(3))
            warn('Mismatched lengths')
        elif mismatch == 'warn':
            warn('Mismatched lengths: {}, {}'.format(len(ds1), len(ds2)))
        
    ###### Align years
    if oneyear:
        yeardiff = year1 - year2
        if yearout in ['left', '1', 1]:
            ### Align to ds1.index
            ds2.index = ds2.index + pd.DateOffset(years=yeardiff)
        elif yearout in ['right', '2', 2]:
            ### Align to ds2.index
            ds1.index = ds1.index - pd.DateOffset(years=yeardiff)
        elif isinstance(yearout, int) and (yearout not in [1,2]):
            ### Align both timeseries to yearout
            year1diff = yearout - year1
            year2diff = yearout - year2
            ds1.index = ds1.index + pd.DateOffset(years=year1diff)
            ds2.index = ds2.index + pd.DateOffset(years=year2diff)
        else:
            pass

    ### Align and clip time zones, if necessary
    if tz1 != tz2:
        if tzout in ['left', 1, '1']:
            ds2 = (pd.DataFrame(ds2.tz_convert(tz1))
                   .merge(pd.DataFrame(index=ds1.index), left_index=True, right_index=True)
                  )[name2]
        elif tzout in ['right', 2, '2']:
            ds1 = (pd.DataFrame(ds1.tz_convert(tz2))
                   .merge(pd.DataFrame(index=ds2.index), left_index=True, right_index=True)
                  )[name1]
            
    return ds1, ds2


############################
### SYSTEM OUTPUT SIMULATION

class PVsystem:
    """
    """
    def __init__(self, 
                 systemtype='track',
                 axis_tilt=None, axis_azimuth=180,
                 max_angle=60, backtrack=True, gcr=1./3.,
                 dcac=1.3, 
                 loss_system=0.14, loss_inverter=0.04,
                 n_ar=1.3, n_glass=1.526,
                 tempcoeff=-0.004, temp_model='open_rack_cell_polymerback',
                 albedo=0.2, 
                 et_method='nrel', diffuse_model='reindl',
                 model_perez='allsitescomposite1990',
                 clip=True
                ):
        ### Defaults
        if (axis_tilt is None) and (systemtype == 'fixed'):
            axis_tilt = 'latitude'
        elif (axis_tilt is None) and (systemtype == 'track'):
            axis_tilt = 0
        ### Parameters
        self.gentype = 'pv'
        self.systemtype = systemtype
        self.axis_tilt = axis_tilt
        self.axis_azimuth = axis_azimuth
        self.max_angle = max_angle
        self.backtrack = backtrack
        self.gcr = gcr
        self.dcac = dcac
        self.loss_system = loss_system
        self.loss_inverter = loss_inverter
        self.n_ar = n_ar
        self.n_glass = n_glass
        self.tempcoeff = tempcoeff
        self.temp_model = temp_model
        self.albedo = albedo
        self.et_method = et_method
        self.diffuse_model = diffuse_model
        self.model_perez = model_perez
        self.clip = clip

    def sim(self, nsrdbfile, year, 
            nsrdbpathin=None, nsrdbtype='.gz',
            resolution='default', 
            output_ac_only=True, output_ac_tz=False,
            return_all=False, query=False, **kwargs
            ):
        """
        Notes
        -----
        * If querying googlemaps and NSRDB with query==True and a latitude/longitude query,
        need to put latitude first, longitude second, and have both in a string.
        E.g. '40, -100' for latitude=40, longitude=-100.
        * Googlemaps query is not foolproof, so should check the output file if you really
        care about getting the right location.
        * Safest approach is to use an explicit nsrdbfile path and query==False.
        """
        if not os.path.exists(
                os.path.join(pvvm.toolbox.pathify(nsrdbpathin), nsrdbfile)):
            if query == True:
                _,_,_,_,fullpath = pvvm.io.queryNSRDBfile(
                    nsrdbfile, year, returnfilename=True, **kwargs)
            elif query == False:
                print(nsrdbpathin)
                print(nsrdbfile)
                raise FileNotFoundError
        else:
            fullpath = nsrdbfile

        return pv_system_sim(
            nsrdbfile=fullpath, year=year, systemtype=self.systemtype, 
            axis_tilt=self.axis_tilt, axis_azimuth=self.axis_azimuth,
            max_angle=self.max_angle, backtrack=self.backtrack, gcr=self.gcr, 
            dcac=self.dcac, loss_system=self.loss_system, 
            loss_inverter=self.loss_inverter, n_ar=self.n_ar, n_glass=self.n_glass,
            tempcoeff=self.tempcoeff, temp_model=self.temp_model, 
            albedo=self.albedo, diffuse_model=self.diffuse_model,
            et_method=self.et_method, model_perez=self.model_perez,
            nsrdbpathin=nsrdbpathin, nsrdbtype=nsrdbtype, resolution=resolution, 
            output_ac_only=output_ac_only, output_ac_tz=output_ac_tz, 
            return_all=return_all, clip=self.clip)

def loss_reflect_abs(aoi, n_glass=1.526, n_ar=1.3, n_air=1, K=4., L=0.002):
    """
    Adapted from pvlib.pvsystem.physicaliam and PVWatts Version 5 section 8
    """
    if isinstance(aoi, pd.Series):
        aoi.loc[aoi <= 1e-6] = 1e-6
    elif isinstance(aoi, float):
        aoi = max(aoi, 1e-6)
    elif isinstance(aoi, int):
        aoi = max(aoi, 1e-6)
    elif isinstance(aoi, np.ndarray):
        aoi[aoi <= 1e-6] = 1e-6

    theta_ar = pvlib.tools.asind(
        (n_air / n_ar) * pvlib.tools.sind(aoi))

    tau_ar = (1 - 0.5 * (
        ((pvlib.tools.sind(theta_ar - aoi)) ** 2)
        / ((pvlib.tools.sind(theta_ar + aoi)) ** 2)
        + ((pvlib.tools.tand(theta_ar - aoi)) ** 2)
        / ((pvlib.tools.tand(theta_ar + aoi)) ** 2)))

    theta_glass = pvlib.tools.asind(
        (n_ar / n_glass) * pvlib.tools.sind(theta_ar))

    tau_glass = (1 - 0.5 * (
        ((pvlib.tools.sind(theta_glass - theta_ar)) ** 2)
        / ((pvlib.tools.sind(theta_glass + theta_ar)) ** 2)
        + ((pvlib.tools.tand(theta_glass - theta_ar)) ** 2)
        / ((pvlib.tools.tand(theta_glass + theta_ar)) ** 2)))

    tau_total = tau_ar * tau_glass

    tau_total = np.where((np.abs(aoi) >= 90) | (tau_total < 0), np.nan, tau_total)

    if isinstance(aoi, pd.Series):
        tau_total = pd.Series(tau_total, index=aoi.index)

    return tau_total

def loss_reflect(
    aoi, n_glass=1.526, n_ar=1.3, n_air=1, K=4., L=0.002,
    fillna=True):
    """
    """
    out = (
        loss_reflect_abs(aoi, n_glass, n_ar, n_air, K, L)
        / loss_reflect_abs(0, n_glass, n_ar, n_air, K, L))

    ####### UPDATED 20180712 ########
    if fillna==True:
        if isinstance(out, (pd.Series, pd.DataFrame)):
            out = out.fillna(0)
        elif isinstance(out, np.ndarray):
            out = np.nan_to_num(out)
    #################################

    return out

def pv_system_sim(
    nsrdbfile, year, systemtype='track', 
    axis_tilt=0, axis_azimuth=180, 
    max_angle=60, backtrack=True, gcr=1./3., 
    dcac=1.3, 
    loss_system=0.14, loss_inverter=0.04, 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    temp_model='open_rack_cell_polymerback',
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990', 
    nsrdbpathin='in/NSRDB/', nsrdbtype='.gz', 
    resolution='default', 
    output_ac_only=True, output_ac_tz=False,
    return_all=False, clip=True):
    """
    Outputs
    -------
    * output_ac: pd.Series of instantaneous Wac per kWac 
        * Divide by 1000 to get instantaneous AC capacity factor [fraction]
        * Take mean to get yearly capacity factor
        * Take sum * resolution / 60 to get yearly energy generation in Wh

    Notes
    -----
    * To return DC (instead of AC) output, set dcac=None and loss_inverter=None
    """
    ### Set resolution, if necessary
    if resolution in ['default', None]:
        if type(year) == int: resolution = 30
        elif str(year).lower() == 'tmy': resolution = 60
        else: raise Exception("year must be 'tmy' or int")

    ### Set NSRDB filepath
    if nsrdbpathin == 'in/NSRDB/':
        nsrdbpath = pathify(
            nsrdbpathin, add='{}/{}min/'.format(year, resolution))
    else:
        nsrdbpath = nsrdbpathin

    ### Load NSRDB file
    dfsun, info, tz, elevation = pvvm.io.getNSRDBfile(
        filepath=nsrdbpath, 
        filename=nsrdbfile, 
        year=year, resolution=resolution, forcemidnight=False)

    ### Select latitude for tilt, if necessary
    latitude = float(info['Latitude'])
    longitude = float(info['Longitude'])

    if axis_tilt == 'latitude':
        axis_tilt = latitude
    elif axis_tilt == 'winter':
        axis_tilt = latitude + 23.437

    ### Set timezone from info in NSRDB file
    timezone = int(info['Time Zone'])
    tz = 'Etc/GMT{:+}'.format(-timezone)

    ### Determine solar position
    times = dfsun.index.copy()
    solpos = pvlib.solarposition.get_solarposition(
        times, latitude, longitude)

    ### Set extra parameters for diffuse sky models
    if diffuse_model in ['haydavies', 'reindl', 'perez']:
        dni_et = pvlib.irradiance.get_extra_radiation(
            times, method=et_method, epoch_year=year)
        airmass = pvlib.atmosphere.get_relative_airmass(
            solpos['apparent_zenith'])
    else:
        dni_et = None
        airmass = None

    ### Get surface tilt, from tracker data if necessary
    if systemtype == 'track':
        ###### ADDED 20190414 ######
        with np.errstate(invalid='ignore'):
            tracker_data = pvlib.tracking.singleaxis(solpos['apparent_zenith'], solpos['azimuth'],
                axis_tilt=axis_tilt, axis_azimuth=axis_azimuth, 
                max_angle=max_angle, backtrack=backtrack, gcr=gcr)
        ############################
        surface_tilt = tracker_data['surface_tilt']
        surface_azimuth = tracker_data['surface_azimuth']
        ###### ADDED 20180712 ######
        surface_tilt = surface_tilt.fillna(axis_tilt).replace(0., axis_tilt)
        surface_azimuth = surface_azimuth.fillna(axis_azimuth)

    elif systemtype == 'fixed':
        surface_tilt = axis_tilt
        surface_azimuth = axis_azimuth

    ### Determine angle of incidence
    aoi = pvlib.irradiance.aoi(
        surface_tilt, surface_azimuth,
        solpos['apparent_zenith'], solpos['azimuth'])

    ### Determine plane-of-array irradiance
    poa_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt, surface_azimuth, 
        solpos['apparent_zenith'], solpos['azimuth'], 
        dfsun['DNI'], dfsun['GHI'], dfsun['DHI'], 
        dni_extra=dni_et, airmass=airmass, albedo=albedo,
        model=diffuse_model, model_perez=model_perez)

    ### Correct for reflectance losses
    poa_irrad['poa_global_reflectlosses'] = (
        (poa_irrad['poa_direct'] * loss_reflect(aoi, n_glass, n_ar)) 
        + poa_irrad['poa_diffuse'])
    poa_irrad.fillna(0, inplace=True)

    ### Correct for temperature losses
    celltemp = pvlib.pvsystem.sapm_celltemp(
        poa_irrad['poa_global_reflectlosses'], dfsun['Wind Speed'], 
        dfsun['Temperature'], temp_model)['temp_cell']

    output_dc_loss_temp = pvlib.pvsystem.pvwatts_dc(
        g_poa_effective=poa_irrad['poa_global_reflectlosses'], 
        temp_cell=celltemp, pdc0=1000, 
        gamma_pdc=tempcoeff, temp_ref=25.)

    ### Corect for total DC system losses
    # output_dc_loss_all = output_dc_loss_temp * eta_system
    output_dc_loss_all = output_dc_loss_temp * (1 - loss_system)

    ### Correct for inverter
    ##################################
    ### Allow for DC output (20180821)
    if (dcac is None) and (loss_inverter is None):
        output_ac = output_dc_loss_all
    else:
        ##############################
        output_ac = pvlib.pvsystem.pvwatts_ac(
            pdc=output_dc_loss_all*dcac,
            ##### UPDATED 20180708 #####
            # pdc0=1000,
            pdc0=1000/(1-loss_inverter),
            ############################
            # eta_inv_nom=eta_inverter,
            eta_inv_nom=(1-loss_inverter),
            eta_inv_ref=0.9637).fillna(0)
        if clip is True:                     ### Added 20181126
            output_ac = output_ac.clip(0)

    ### Return output
    if output_ac_only:
        return output_ac
    if output_ac_tz:
        return output_ac, tz
    if (return_all == True) and (systemtype == 'track'):
        return (dfsun, solpos, aoi, poa_irrad,
            celltemp, output_dc_loss_temp,
            output_dc_loss_all, output_ac, tracker_data)
    if (return_all == True) and (systemtype == 'fixed'):
        return (dfsun, solpos, aoi, poa_irrad, 
            celltemp, output_dc_loss_temp, 
            output_dc_loss_all, output_ac)

def pv_system_sim_fast(
    axis_tilt_and_azimuth,
    dfsun, info, tznode, elevation,
    solpos, dni_et, airmass,
    year, systemtype, 
    max_angle=60, backtrack=True, gcr=1./3., 
    dcac=1.3, 
    loss_system=0.14, loss_inverter=0.04, 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    temp_model='open_rack_cell_polymerback',
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990',
    clip=True):
    """
    Calculate ac_out after solar resource file has already been loaded
    """
    ### Unpack axis_tilt_and_azimuth
    axis_tilt, axis_azimuth = axis_tilt_and_azimuth

    ### Get surface tilt, from tracker data if necessary
    if systemtype == 'track':
        tracker_data = pvlib.tracking.singleaxis(solpos['apparent_zenith'], solpos['azimuth'],
            axis_tilt=axis_tilt, axis_azimuth=axis_azimuth, 
            max_angle=max_angle, backtrack=backtrack, gcr=gcr)
        surface_tilt = tracker_data['surface_tilt']
        surface_azimuth = tracker_data['surface_azimuth']
        ###### ADDED 20180712 ######
        surface_tilt = surface_tilt.fillna(axis_tilt).replace(0., axis_tilt)
        surface_azimuth = surface_azimuth.fillna(axis_azimuth)

    elif systemtype == 'fixed':
        surface_tilt = axis_tilt
        surface_azimuth = axis_azimuth

    ### Determine angle of incidence
    aoi = pvlib.irradiance.aoi(
        surface_tilt, surface_azimuth,
        solpos['apparent_zenith'], solpos['azimuth'])

    ### Determine plane-of-array irradiance
    poa_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt, surface_azimuth, 
        solpos['apparent_zenith'], solpos['azimuth'], 
        dfsun['DNI'], dfsun['GHI'], dfsun['DHI'], 
        dni_extra=dni_et, airmass=airmass, albedo=albedo,
        model=diffuse_model, model_perez=model_perez)

    ### Correct for reflectance losses
    poa_irrad['poa_global_reflectlosses'] = (
        (poa_irrad['poa_direct'] * loss_reflect(aoi, n_glass, n_ar)) 
        + poa_irrad['poa_diffuse'])
    poa_irrad.fillna(0, inplace=True)

    ### Correct for temperature losses
    celltemp = pvlib.pvsystem.sapm_celltemp(
        poa_irrad['poa_global_reflectlosses'], dfsun['Wind Speed'], 
        dfsun['Temperature'], temp_model)['temp_cell']

    output_dc_loss_temp = pvlib.pvsystem.pvwatts_dc(
        g_poa_effective=poa_irrad['poa_global_reflectlosses'], 
        temp_cell=celltemp, pdc0=1000, 
        gamma_pdc=tempcoeff, temp_ref=25.)

    ### Corect for total DC system losses
    output_dc_loss_all = output_dc_loss_temp * (1 - loss_system)

    ### Correct for inverter
    output_ac = pvlib.pvsystem.pvwatts_ac(
        pdc=output_dc_loss_all*dcac,
        ##### UPDATED 20180708 #####
        # pdc0=1000,
        pdc0=1000/(1-loss_inverter),
        ############################
        eta_inv_nom=(1-loss_inverter),
        eta_inv_ref=0.9637).fillna(0)
    if clip is True:
        output_ac = output_ac.clip(0)

    return output_ac


######################
### VALUE ANALYSIS ###

################
### Energy value

def solarvalue(yearlmp, yearsun, 
    isos=['caiso', 'ercot', 'miso', 'pjm', 'nyiso', 'isone'],
    market='da', submarket=None, 
    monthly=False, daily=False,
    pricecutoff=0,
    systemtype='track', dcac=1.3, 
    axis_tilt=None, axis_azimuth=180,
    max_angle=60, backtrack=True, gcr=1./3., 
    loss_system=0.14, loss_inverter=0.04,
    n_ar=1.3, n_glass=1.526, tempcoeff=-0.004,
    temp_model='open_rack_cell_polymerback',
    albedo=0.2, diffuse_model='reindl',
    et_method='nrel', model_perez='allsitescomposite1990',
    savemod='', write='default', compress=False, runtype='full',
    nsrdbpathin='default', nsrdbtype='.gz',
    lmppath='default', lmptype='.gz',
    outpath='out',
    savesafe=True, product='lmp', clip=True, opt_pricecutoff='no'): 
    """
    Inputs
    ------

    Outputs
    -------

    """

    ### Normalize inputs
    ## Put isos in lower case
    isos = [iso.lower() for iso in isos]
    ## Cut ERCOT if year < 2011
    if (yearlmp <= 2010) and ('ercot' in isos):
        isos.remove('ercot')
    ## Cut ISONE if year < 2011 and market == 'rt'
    if (yearlmp <= 2010) and (market == 'rt') and ('isone' in isos):
        isos.remove('isone')
    ## Set resolutions
    if yearsun == 'tmy':
        resolutionsun = 60
    elif type(yearsun) == int:
        resolutionsun = 30
    else:
        print(yearsun)
        raise Exception("yearsun must by 'tmy' or int")
    if axis_tilt is None:
        if systemtype == 'track':
            axis_tilt = 0
        elif systemtype == 'fixed':
            axis_tilt = 'latitude'

    ## Check other inputs
    if write not in ['default', False] and write.find('.') == -1:
        raise Exception('write (i.e. savename) needs a "."')
    for iso in isos:
        if iso not in ['caiso', 'ercot', 'miso', 
            'pjm', 'nyiso', 'isone']:
            raise Exception("Invalid iso")
    if (product is None) or (product in ['lmp','LMP']):
        product = 'lmp'
    elif product.lower() in ['mcl','mlc','loss','losses']:
        product = 'mcl'
    elif product.lower() in ['mcc','congestion']:
        product = 'mcc'
    elif product.lower() in ['mce','mec','energy']:
        product = 'mce'
    else:
        raise Exception("invalid product: {}".format(product))

    ### Get the optimized orientations
    if type(axis_tilt) == str:
        axis_tilt = axis_tilt.lower()
    if type(axis_azimuth) == str:
        axis_azimuth = axis_azimuth.lower()

    if (axis_tilt in ['optrev','optcf']) or (axis_azimuth in ['optrev','optcf']):
        infile = ('{}io/CEMPNInodes-lmp_optimized_orientations-tmysun'
                  '-2010_2017lmp-{}-{}-{}cutoff.p'.format(
                    revmpath, market, systemtype, opt_pricecutoff))
        with open(infile, 'rb') as p:
            dictorient = pickle.load(p)

    ### savenames
    if write:
        if compress:
            compression, saveend = ('gzip', '.gz')
        else:
            compression, saveend = (None, '.csv')

        abbrevs = {
            'caiso': 'C',
            'ercot': 'E',
            'miso': 'M',
            'pjm': 'P',
            'nyiso': 'N',
            'isone': 'I'}
        abbrev = ''.join([abbrevs[i.lower()] for i in isos])

        if write == 'default':
            # savename = '{}/PVvalueV7-{}-{}-{}lmp-{}sun-{}-{}ILR-{}m-{}d-{}cutoff{}{}'.format(
            #     outpath, abbrev, market, yearlmp, yearsun, 
            #     systemtype, 
            #     int((1-loss_system)*(1-loss_inverter)*dcac*100), 
            #     int(monthly), int(daily),
            #     pricecutoff,
            #     savemod, saveend)
            ### V9: Allowing for optimized orientation
            savename = '{}/PVvalueV9-{}-{}-{}-{}lmp-{}sun-{}-{}cutoff{}{}{}{}{}'.format(
                outpath, abbrev, market, product,
                yearlmp, yearsun, 
                systemtype, 
                pricecutoff,
                '-{}'.format(axis_azimuth) if axis_azimuth in ['optrev','optcf'] else '',
                '-monthly' if monthly is True else '',
                '-daily' if daily is True else '',
                '-{}'.format(savemod) if savemod is not None else '-0',
                saveend)
        else:
            savename = os.path.join(outpath, write)

        describename = os.path.splitext(savename)[0] + '-describe.txt'
        if savesafe == True:
            savename = pvvm.toolbox.safesave(savename)
            describename = pvvm.toolbox.safesave(describename)
        print(savename)
        sys.stdout.flush()
        ### Make sure the folder exists
        if not os.path.isdir(os.path.dirname(savename)):
            raise Exception('{} does not exist'.format(os.path.dirname(savename)))


    ### Convenience variables
    hours = min(pvvm.toolbox.yearhours(yearlmp), pvvm.toolbox.yearhours(yearsun))

    ### Set up results containers
    results_to_concat = {}

    ### Loop over ISOs
    for iso in isos:
        ### Set ISO parameters
        tzlmp = pvvm.toolbox.tz_iso[iso]
        resolutionlmp = pvvm.data.resolutionlmps[(iso.upper(), market)]

        ### Glue together different ISO labeling formalisms
        (nsrdbindex, lmpindex, pnodeid,
            latlonindex, pnodename) = pvvm.io.glue_iso_columns(iso)

        ### Get nodes with lat/lon info and full lmp data for yearlmp
        dfin = pvvm.io.get_iso_nodes(iso, market, yearlmp, merge=True)

        ### Set NSRDB filepath
        if nsrdbpathin == 'default':
            nsrdbpath = '{}{}/in/NSRDB/{}/{}min/'.format(
                revmpath, iso.upper(), yearsun, resolutionsun)
        else:
            nsrdbpath = nsrdbpathin

        ### Set LMP filepath
        if lmppath == 'default':
            ### DO: Modify this when you want to allow monthly value
            # if (market == 'rt') and (iso == 'ercot'):
            #     lmpfilepath = '{}{}/io/lmp-nodal/{}/'.format(
            #         revmpath, iso.upper(), 'rt-year')
            # elif (market == 'rt') and (iso == 'caiso'):
            #     lmpfilepath = '{}{}/io/lmp-nodal/{}/'.format(
            #         revmpath, iso.upper(), 'rt/rtm-yearly')
            # else:
            lmpfilepath = '{}{}/io/lmp-nodal/{}/'.format(
                revmpath, iso.upper(), market)
        else:
            lmpfilepath = lmppath

        ### Make NSRDB file list
        nsrdbfiles = list(dfin[nsrdbindex])
        for i in range(len(nsrdbfiles)):
            nsrdbfiles[i] = str(nsrdbfiles[i]) + '-{}{}'.format(yearsun, nsrdbtype)

        ### Make lmp file list
        lmpfiles = list(dfin[pnodeid])
        for i in range(len(lmpfiles)):
            lmpfiles[i] = str(lmpfiles[i]) + '-{}{}'.format(yearlmp, lmptype)

        ### Make isonode list
        isonodes = list(iso.upper() + ':' + dfin[pnodeid].astype(str))

        ### Determine runlength
        if runtype == 'full':
            runlength = len(dfin)
        elif runtype == 'test':
            runlength = 5
        elif type(runtype) == int:
            runlength = runtype
        else:
            raise Exception("Invalid runtype. Must be 'full', 'test', or int.")

        #######################
        ### CALCULATE VALUE ###

        ### Loop over nodes
        for i in trange(runlength, desc=iso):
            isonode = isonodes[i]

            ### Get optimized orientations, if necessary
            if axis_tilt == 'optrev':
                axis_tilt_node = dictorient[isonode, yearlmp]['OptRev_Tilt']
            elif axis_tilt == 'optcf':
                axis_tilt_node = dictorient[isonode, yearlmp]['OptCF_Tilt']
            else:
                axis_tilt_node = axis_tilt

            if axis_azimuth == 'optrev':
                axis_azimuth_node = dictorient[isonode, yearlmp]['OptRev_Azimuth']
            elif axis_azimuth == 'optcf':
                axis_azimuth_node = dictorient[isonode, yearlmp]['OptCF_Azimuth']
            else:
                axis_azimuth_node = axis_azimuth

            output_ac, tznode = pv_system_sim(
                # nsrdbfile=str(dfin.loc[i, latlonindex]), 
                nsrdbfile=nsrdbfiles[i],
                year=yearsun, systemtype=systemtype,
                resolution=resolutionsun, dcac=dcac, 
                axis_tilt=axis_tilt_node, axis_azimuth=axis_azimuth_node, 
                max_angle=max_angle, backtrack=backtrack, gcr=gcr, 
                loss_system=loss_system, loss_inverter=loss_inverter,
                n_ar=n_ar, n_glass=n_glass,
                tempcoeff=tempcoeff, temp_model=temp_model,
                albedo=albedo, nsrdbpathin=nsrdbpath, nsrdbtype=nsrdbtype, 
                et_method=et_method, diffuse_model=diffuse_model, 
                model_perez=model_perez, 
                output_ac_only=False, output_ac_tz=True, clip=clip)     

            ### Get LMP data
            dflmp = pvvm.io.getLMPfile(lmpfilepath, lmpfiles[i], tzlmp, product=product)[product]

            ### Drop leap days if yearlmp and yearsun have different year lengths
            if pvvm.toolbox.yearhours(yearlmp) == 8784:
                if (yearsun == 'tmy') or (pvvm.toolbox.yearhours(yearsun) == 8760):
                    ## Drop lmp leapyear
                    dflmp = dropleap(dflmp, yearlmp, resolutionlmp)
            elif ((pvvm.toolbox.yearhours(yearlmp) == 8760) 
                  and (pvvm.toolbox.yearhours(yearsun) == 8784)):
                ## Drop sun leapyear
                output_ac = dropleap(output_ac, yearsun, resolutionsun)

            ### Reset indices if yearsun != yearlmp
            if yearsun != yearlmp:
                output_ac.index = pd.date_range(
                    '2001-01-01', 
                    periods=(8760 * 60 / resolutionsun), 
                    freq='{}min'.format(resolutionsun), 
                    tz=tznode
                ).tz_convert(tzlmp)
                
                dflmp.index = pd.date_range(
                    '2001-01-01', 
                    periods=(8760 * 60 / resolutionlmp),
                    freq='{}min'.format(resolutionlmp), 
                    tz=tzlmp
                )

            ### upsample solar data if resolutionlmp == 5
            if resolutionlmp == 5:
                ### Original verion - no longer works given recent pandas update
                # output_ac.loc[output_ac.index.max() + 1] = output_ac.iloc[-1]
                ### New version
                output_ac = output_ac.append(
                    pd.Series(data=output_ac.iloc[-1],
                              index=[output_ac.index[-1] + pd.Timedelta(resolutionsun, 'm')]))
                ### Continue
                output_ac = output_ac.resample('5T').interpolate(method='time')
                output_ac.drop(output_ac.index[-1], axis=0, inplace=True)

            ### upsample LMP data if resolutionsun == 30
            if (resolutionlmp == 60) and (resolutionsun == 30):
                dflmp = dflmp.resample('30T').ffill()
                ### Original verion - no longer works given recent pandas update
                # dflmp.loc[dflmp.index.max() + 1] = dflmp.iloc[-1]
                ### New version
                dflmp = dflmp.append(
                    pd.Series(
                        data=dflmp.iloc[-1], name=product,
                        index=[dflmp.index[-1] + pd.Timedelta(resolutionsun, 'm')]))
                ### Continue

            if len(output_ac) != len(dflmp):
                print('Something probably went wrong with leap years')
                print('NSRDBfile = {}'.format(dfin.loc[i, latlonindex]))
                print('lmpfile = {}'.format(lmpfiles[i]))
                print('len(output_ac.index) = {}'.format(len(output_ac.index)))
                print('len(dflmp.index) = {}'.format(len(dflmp.index)))
                raise Exception('Mismatched lengths for LMP and NSRDB files')
            
            ### put dflmp into same timezone as output_ac
            ### (note that this will drop point(s) from dflmp)
            if tznode != tzlmp:
                dflmp = (
                    pd.DataFrame(dflmp.tz_convert(tznode))
                    .merge(pd.DataFrame(index=output_ac.index), 
                           left_index=True, right_index=True)
                )[product]
            
            ### determine final resolution
            resolution = min(resolutionlmp, resolutionsun)
            
            ############################
            ##### PERFORM ANALYSIS #####
            
            ### Yearly
            
            ## determine dispatch schedule
            dispatchall = dflmp.map(lambda x: True).rename('dispatchall')
            if product == 'lmp':
                dispatch = dflmp.map(lambda x: x > pricecutoff).rename('dispatch')
            ### Determine dispatch schedule based on LMP-derived dispatch
            else:
                dfprice = getLMPfile(lmpfilepath, lmpfiles[i], tzlmp, product='lmp')['lmp']
                ###### Duplicate dflmp munging above for dfprice
                ### Drop leap days if yearlmp and yearsun have different year lengths
                if pvvm.toolbox.yearhours(yearlmp) == 8784:
                    if (yearsun == 'tmy') or pvvm.toolbox.yearhours(yearsun) == 8760:
                        ## Drop lmp leapyear
                        dfprice = dropleap(dflmp, yearlmp, resolutionlmp)
                ### Reset indices if yearsun != yearlmp
                if yearsun != yearlmp:
                    dfprice.index = pd.date_range(
                        '2001-01-01', 
                        periods=(8760 * 60 / resolutionlmp),
                        freq='{}min'.format(resolutionlmp), 
                        tz=tzlmp
                    )
                ### upsample LMP data if resolutionsun == 30
                if (resolutionlmp == 60) and (resolutionsun == 30):
                    dfprice = dfprice.resample('30T').ffill()
                    dfprice.loc[dfprice.index.max() + 1] = dfprice.iloc[-1]

                if len(output_ac) != len(dfprice):
                    print('Something probably went wrong with leap years')
                    print('NSRDBfile = {}'.format(dfin.loc[i, latlonindex]))
                    print('lmpfile = {}'.format(lmpfiles[i]))
                    print('len(output_ac.index) = {}'.format(len(output_ac.index)))
                    print('len(dfprice.index) = {}'.format(len(dfprice.index)))
                    print('dfprice.shape = {}'.format(dfprice.shape))
                    print('output_ac.shape = {}'.format(output_ac.shape))
                    print('dfprice:')
                    print(dfprice.head())
                    print(dfprice.tail())
                    print('output_ac:')
                    print(output_ac.head())
                    print(output_ac.tail())
                    raise Exception('Mismatched lengths for LMP and NSRDB files')

                ### put dfprice into same timezone as output_ac
                ### (note that this will drop point(s) from dfprice)
                if tznode != tzlmp:
                    dfprice = (
                        pd.DataFrame(dfprice.tz_convert(tznode))
                        .merge(pd.DataFrame(index=output_ac.index), 
                               left_index=True, right_index=True)
                    )['lmp']

                ### Determine dispatch based on price
                dispatch = dfprice.map(lambda x: x > pricecutoff).rename('dispatch')
            
            output_ac = (output_ac * dispatchall).rename('output_ac')
            output_dispatched = (output_ac * dispatch).rename('output_dispatched')
            
            ## generate results
            price_average = dflmp.mean()
            
            generation_yearly = 0.001 * output_ac.sum() * resolution / 60
            generation_yearly_dispatched = 0.001 * output_dispatched.sum() * resolution / 60

            capacity_factor = generation_yearly / hours
            capacity_factor_dispatched = generation_yearly_dispatched / hours

            revenue_timestep = (
                0.001 * output_ac * dflmp * resolution / 60
            ).rename('revenue_timestep')

            revenue_timestep_dispatched = (
                0.001 * output_dispatched * dflmp * resolution / 60
            ).rename('revenue_timestep_dispatched')
            
            revenue_yearly = revenue_timestep.sum() / 1000 # $/kWac-yr
            revenue_yearly_dispatched = revenue_timestep_dispatched.sum() / 1000 # $/kWac-yr

            value_average = revenue_yearly / generation_yearly * 1000
            value_average_dispatched = revenue_yearly_dispatched / generation_yearly_dispatched * 1000

            ### Correct for zero division (can occur for mcc)
            try:
                value_factor = value_average / price_average
            except ZeroDivisionError:
                value_factor = np.nan
            try:
                value_factor_dispatched = value_average_dispatched / price_average
            except ZeroDivisionError:
                value_factor_dispatched = np.nan

            results_node_year = pd.DataFrame(
                data=([[
                    0, 0, 
                    axis_tilt_node, axis_azimuth_node,
                    price_average, 
                    capacity_factor, capacity_factor_dispatched, 
                    revenue_yearly, revenue_yearly_dispatched, 
                    value_average, value_average_dispatched,
                    value_factor, value_factor_dispatched,
                ]]),
                columns=([
                    'Month', 'Day', 
                    'Tilt', 'Azimuth',
                    'PriceAverage', 
                    'CapacityFactor', 'CapacityFactor_dispatched', 
                    'Revenue', 'Revenue_dispatched', 
                    'ValueAverage', 'ValueAverage_dispatched', 
                    'ValueFactor', 'ValueFactor_dispatched',
                ]))


            ### Monthly
            if monthly:
                ## generation [MWh]
                generation_monthly = (
                    output_ac.groupby(output_ac.index.month)
                    .sum().rename('Generation')
                ) / 1000 * resolution / 60

               ## generation dispatched [MWh]
                generation_monthly_dispatched = (
                    output_dispatched.groupby(output_dispatched.index.month)
                    .sum().rename('Generation_dispatched')
                ) / 1000 * resolution / 60
                
                ### Have to do it this way because .groupby().mean() doesn't like np.nan
                ## capacity factor [fraction]
                capacity_factor = (
                    output_ac.groupby([output_ac.index.month]).sum()
                    / output_ac.groupby([output_ac.index.month]).count()
                ).rename('CapacityFactor') / 1000
                
                ## capacity factor dispatched [fraction]
                capacity_factor_dispatched = (
                    output_dispatched.groupby([output_dispatched.index.month]).sum()
                    / output_dispatched.groupby([output_dispatched.index.month]).count()
                ).rename('CapacityFactor_dispatched') / 1000
                            
                ## average price [$/MWh]
                price_average = (
                    dflmp.groupby(dflmp.index.month)
                    .mean().rename('PriceAverage')
                )      

                ## daily revenue [$/kWac-day]
                revenue_outstep = (
                    revenue_timestep.groupby(revenue_timestep.index.month)
                    .sum().rename('Revenue')
                ) / 1000
                
               ## daily revenue dispatched [$/kWac-day]
                revenue_outstep_dispatched = (
                    revenue_timestep_dispatched.groupby(revenue_timestep_dispatched.index.month)
                    .sum().rename('Revenue_dispatched')
                ) / 1000

               ## average value [$/MWh]
                value_average = (
                    revenue_outstep * 1000
                    / generation_monthly).rename('ValueAverage')
    
               ## average value dispatched [$/MWh]
                value_average_dispatched = (
                    revenue_outstep_dispatched * 1000
                    / generation_monthly_dispatched).rename('ValueAverage_dispatched')

                ## value factor [fraction]
                value_factor = (value_average / price_average).rename('ValueFactor')
            
               ## value factor dispatched [fraction]
                value_factor_dispatched = (
                    value_average_dispatched / price_average).rename('ValueFactor_dispatched')


                results_node_month = (pd.concat(
                    [price_average, 
                     capacity_factor, capacity_factor_dispatched, 
                     revenue_outstep, revenue_outstep_dispatched, 
                     value_average, value_average_dispatched,
                     value_factor, value_factor_dispatched],
                    axis=1))

                results_node_month.reset_index(inplace=True)
                results_node_month.rename(columns={'index':'Month'}, inplace=True)
                results_node_month.insert(1, 'Day', 0)


            ### Daily
            if daily:
                ### Have to do it this way because .groupby().mean() doesn't like np.nan
                ## capacity factor [fraction]
                capacity_factor = (
                    output_ac.groupby((output_ac.index.month, output_ac.index.day)).sum()
                    / output_ac.groupby((output_ac.index.month, output_ac.index.day)).count()
                ).rename('CapacityFactor') / 1000
                
                ## capacity factor dispatched [fraction]
                capacity_factor_dispatched = (
                    output_dispatched.groupby(
                        [output_dispatched.index.month, output_dispatched.index.day]).sum()
                    / output_dispatched.groupby(
                        [output_dispatched.index.month, output_dispatched.index.day]).count()
                ).rename('CapacityFactor_dispatched') / 1000
                
                ## average price [$/MWh]
                price_average = (
                    dflmp
                    .groupby((dflmp.index.month, dflmp.index.day))
                    .mean()
                    .rename('PriceAverage')
                )

                ## daily revenue [$/kWac-day]
                revenue_daily = (
                    revenue_timestep
                    .groupby(
                        (revenue_timestep.index.month, revenue_timestep.index.day))
                    .sum()
                    .rename('Revenue')
                ) / 1000
                
                ## daily revenue dispatched [$/kWac-day]
                revenue_daily_dispatched = (
                    revenue_timestep_dispatched
                    .groupby(
                        (revenue_timestep_dispatched.index.month, 
                         revenue_timestep_dispatched.index.day))
                    .sum()
                    .rename('Revenue_dispatched')
                ) / 1000

                ## average value [$/MWh]
                value_average = (
                    revenue_daily * 1000
                    / (capacity_factor * 24)).rename('ValueAverage')
                
                ## average value dispatched [$/MWh]
                value_average_dispatched = (
                    revenue_daily_dispatched * 1000
                    / (capacity_factor_dispatched * 24)).rename('ValueAverage_dispatched')

                ## value factor [fraction]
                value_factor = (value_average / price_average).rename('ValueFactor')
                
                ## value factor dispatched [fraction]
                value_factor_dispatched = (
                    value_average_dispatched / price_average).rename('ValueFactor_dispatched')

                ## combine into dataframe
                results_node_day = pd.concat(
                    [price_average, 
                     capacity_factor, capacity_factor_dispatched, 
                     revenue_daily, revenue_daily_dispatched, 
                     value_average, value_average_dispatched, 
                     value_factor, value_factor_dispatched],
                    axis=1)

                ## change leapyear to np.nan (to make checksums come out right)
                # if pvvm.toolbox.yearhours(yearlmp) == 8784 and pvvm.toolbox.yearhours(yearsun) == 8760:
                #   results_node_day.loc[2].loc[29, 'CapacityFactor'] = np.nan

                ## correct indices
                results_node_day.reset_index(inplace=True)
                results_node_day.rename(columns={'level_0':'Month', 'level_1':'Day'},inplace=True)

            ### create nodal results df
            if daily and monthly:
                results_node = pd.concat([results_node_year, results_node_month, results_node_day])
            elif monthly:
                results_node = pd.concat([results_node_year, results_node_month])
            elif daily:
                results_node = pd.concat([results_node_year, results_node_day])
            else:
                results_node = results_node_year.copy()
            
            results_node.insert(0, 'ISO', iso.upper())
            results_node.insert(1, 'ISO:Node', isonodes[i])
            results_node.insert(2, 'NodeID', dfin[pnodeid][i])
            results_node.insert(3, 'NodeName', dfin[pnodename][i])
            results_node.insert(4, 'Latitude', dfin['latitude'][i])
            results_node.insert(5, 'Longitude', dfin['longitude'][i])
            results_node.insert(6, 'LatLonIndex', dfin[latlonindex][i])
            results_node.insert(7, 'Timezone', tznode)

            ### Write results to output dictionary
            results_to_concat[isonodes[i]] = results_node

    ### Write output dataframe from output dictionary
    dfout = pd.concat(results_to_concat)

    if write == False:
        return dfout

    ### Write results
    dfout.to_csv(savename, index=False, compression=compression)
    ### Write summary yearly results if monthly and daily
    if monthly and daily:
        ### Make newsavename with 0m, 0d, csv
        oldnamelist = savename.split('-')
        newnamelist = ['0m' if i == '1m' else i for i in oldnamelist]
        newnamelist = ['0d' if i == '1d' else i for i in newnamelist]
        newname = '-'.join(newnamelist)
        newnamelist = newname.split('.')
        newnamelist = ['csv' if i == 'gz' else i for i in newnamelist]
        newname = '.'.join(newnamelist)
        ### Save the yearly results
        dfout[dfout.Month == 0].to_csv(newname, index=False, compression=None)

    with open(describename, 'w') as f:
        f.write('datetime of run = {}\n'.format(pvvm.toolbox.nowtime()))
        f.write('savename of run = {}\n'.format(savename))
        f.write('script =          {}\n'.format(os.path.basename(__file__)))
        f.write('\n')
        f.write('isos =            {}\n'.format(abbrev))
        f.write('product =         {}\n'.format(product))
        f.write('yearlmp =         {}\n'.format(yearlmp))
        f.write('yearsun =         {}\n'.format(yearsun))
        f.write('market =          {}\n'.format(market))
        f.write('submarket =       {}\n'.format(submarket))
        f.write('pricecutoff =     {}\n'.format(pricecutoff))
        f.write('resolutionsun =   {}\n'.format(resolutionsun))
        f.write('\n')
        f.write('monthly =         {}\n'.format(monthly))
        f.write('daily =           {}\n'.format(daily))
        f.write('\n')
        f.write('systemtype =      {}\n'.format(systemtype))
        f.write('dcac =            {}\n'.format(dcac))
        f.write('loss_system =     {}\n'.format(loss_system))
        f.write('loss_inverter =   {}\n'.format(loss_inverter))
        f.write('n_ar =            {}\n'.format(n_ar))
        f.write('n_glass =         {}\n'.format(n_glass))
        f.write('tempcoeff =       {}\n'.format(tempcoeff))
        f.write('temp_model =      {}\n'.format(temp_model))
        f.write('\n')
        f.write('axis_tilt =       {}\n'.format(axis_tilt))
        f.write('axis_azimuth =    {}\n'.format(axis_azimuth))
        f.write('max_angle =       {}\n'.format(max_angle))
        f.write('backtrack =       {}\n'.format(backtrack))
        f.write('gcr =             {}\n'.format(gcr))
        f.write('\n')
        f.write('albedo =          {}\n'.format(albedo))    
        f.write('diffuse_model =   {}\n'.format(diffuse_model))
        f.write('et_method =       {}\n'.format(et_method))
        f.write('model_perez =     {}\n'.format(model_perez))
        f.write('\n')
        f.write('clip =            {}\n'.format(clip))
        f.write('opt_pricecutoff = {}\n'.format(opt_pricecutoff))
        f.write('\n')
        f.write('revmpath =        {}\n'.format(revmpath))
        f.write('write =           {}\n'.format(write))
        f.write('runtype =         {}\n'.format(runtype))
        f.write('nsrdbpath =       {}\n'.format(nsrdbpath))
        f.write('nsrdbtype =       {}\n'.format(nsrdbtype))
        f.write('lmppath =         {}\n'.format(lmppath))
        f.write('lmptype =         {}\n'.format(lmptype))
        f.write('\n')
        f.write("results_units =   'NodeID': int\n")
        f.write("results_units =   'NodeName': str\n")
        f.write("results_units =   'Latitude': float\n")
        f.write("results_units =   'Longitude': float\n")
        f.write("results_units =   'LatLonIndex': int\n")
        f.write("results_units =   'Timezone': str (Etc/GMT+{})\n")
        f.write("results_units =   'CapacityFactor': fraction\n")
        f.write("results_units =   'PriceAverage': $/MWh\n")
        f.write("results_units =   'ValueAverage': $/MWh\n")
        f.write("results_units =   'RevenueYearly': $/kWac\n")
        f.write("results_units =   'ValueFactor': fraction\n")

def calculate_value(dfval, dfgen, resolution=60, scaler=1000, returntype=tuple):
    """
    """
    if (type(dfval) == pd.Series) and (type(dfgen) == pd.Series):
        assert all(dfval.index == dfgen.index)
    elif (type(dfval) == np.ndarray) and (type(dfgen) == np.ndarray):
        assert dfval.shape == dfgen.shape
    else:
        raise Exception('dfval, dfarry must be pd.Series or np.ndarray')

    hours = len(dfval) * resolution / 60
    
    capacity_factor = dfgen.mean()
    price_average = dfval.mean()
    generation_yearly = dfgen.sum() * resolution / 60 / scaler
    revenue_timestep = dfgen * dfval * resolution / 60 / scaler
    revenue_yearly = revenue_timestep.sum() / 1000 # $/kWac-yr
    with np.errstate(invalid='ignore'):
        value_average = revenue_yearly / generation_yearly * 1000
    try:
        with np.errstate(invalid='ignore'):
            value_factor = value_average / price_average
    except ZeroDivisionError:
        value_factor = np.nan
    
    if returntype == tuple:
        return (capacity_factor, price_average, revenue_yearly, 
                value_average, value_factor)
    elif returntype == dict:
        return {
            'capacity_factor':capacity_factor, 'price': price_average,
            'revenue': revenue_yearly, 'value': value_average,
            'value_factor': value_factor
        }

##################
### Capacity value

def get_iso_critical_hours(iso, year, resolution='H'):
    """
    MISO: https://www.misoenergy.org/legal/business-practice-manuals/
    PJM: https://www.pjm.com/library/manuals.aspx
    NYISO: https://www.nyiso.com/manuals-tech-bulletins-user-guides
    ISONE: https://www.iso-ne.com/participate/rules-procedures/tariff/market-rule-1
    """
    ### Set up dummy year
    crityear = pd.date_range(
        '{}-01-01'.format(year),
        '{}-01-01'.format(year+1),
        freq=resolution, closed='left', 
        tz=pvvm.toolbox.tz_iso[iso])
    crityear = pd.Series(index=crityear)

    if iso == 'ISONE':
        ### Summer
        crityear.loc[
            (crityear.index.month.isin([6,7,8,9]))
            & (crityear.index.hour.isin([13,14,15,16,17]))
        ] = 1
        ### Winter
        crityear.loc[
            (crityear.index.month.isin([10,11,12,1,2,3,4,5]))
            & (crityear.index.hour.isin([17,18]))
        ] = 1        
    elif iso == 'MISO':
        ### Summer
        crityear.loc[
            (crityear.index.month.isin([6,7,8]))
            & (crityear.index.hour.isin([14,15,16]))
        ] = 1
    elif iso == 'NYISO':
        ### Summer
        crityear.loc[
            (crityear.index.month.isin([6,7,8]))
            & (crityear.index.hour.isin([14,15,16,17]))
        ] = 1
        ### Winter
        crityear.loc[
            (crityear.index.month.isin([12,1,2,]))
            & (crityear.index.hour.isin([16,17,18,19]))
        ] = 1
    elif iso == 'PJM':
        ### Summer
        crityear.loc[
            (crityear.index.month.isin([6,7,8]))
            & (crityear.index.hour.isin([14,15,16,17]))
        ] = 1

    ### Otherwise 0
    crityear = crityear.fillna(0).astype(int)

    return crityear

def nhours(percent, year):
    """
    Determine number of hours corresponding to given percent of year
    """
    return int(np.around(0.01*percent*pvvm.toolbox.yearhours(year)))

# def getcriticalhours(iso, year, percent=10, zone=None, 
#     dropleap=False, numhours=None, 
#     source='ferc', dfferc=None):
#     """
#     Determine mask for top load hours, given ISO, year, and percent
#     NOTE: This doesn't give you highest NET load, only highest load
#     """
#     ### Setup
#     dfload = pvvm.io.getload_ferc(iso, year, dfferc=dfferc).copy()
#     if (dropleap is True) and (pvvm.toolbox.yearhours(year)==8784):
#         dfload.drop(dfload.loc['{}-02-29'.format(year)].index, inplace=True)
#     index = dfload.index.copy()
#     dfloadi = dfload.reset_index(drop=True)
#     indexi = dfloadi.index.copy()

#     ### Get number of hours if using percent
#     if numhours is None:
#         numhours = nhours(percent, year)
#         ### Correct if dropping leap year
#         if dropleap is True:
#             numhours = nhours(percent, 2001)
#     else:
#         assert type(numhours) is int
#     ### Get mask of hours of largest load
#     mask = dfloadi.nlargest(numhours).index
#     ### Return boolean mask as dictout
#     df = pd.Series(indexi.map(lambda x: x in mask))    
#     df.index = index

#     return df

def getcriticalhours(
    dsload, percent=3.21, year=None,
    numhours=None, dropleap=False,
    resolution=60,
    ):
    """
    Notes
    * IMPORTANT: Need to use timeserieslineup first!
    """
    ### Take year from middle of dsload if necessary
    if year is None:
        year = dsload.index[int(len(dsload)/2)].year
    ### Identify number of critical hours if using percent
    if numhours is None:
        numhours = nhours(percent, year) * int(resolution/60)
        ### Correct if dropping leap year
        if dropleap is True:
            numhours = nhours(percent, 2001) * int(resolution/60)
    ### Otherwise make sure numhours is an integer
    else:
        assert type(numhours) is int
    ### Make copy of original index to use at end
    index = dsload.index.copy()
    ### Drop the index
    dsloadi = dsload.reset_index(drop=True)
    ### Get the new, ordered-integer index
    indexi = dsloadi.index.copy()
    ### Identify the integer indices of the largest values
    mask = dsloadi.nlargest(numhours).index
    ### Make a boolean series: True if index is in list of largest values
    mask = pd.Series(indexi.map(lambda x: x in mask))
    ### Switch back to the original datetime index
    mask.index = index

    return mask

def get_capacity_price(isonode, year, mask, 
    dictcapacityprice=None, returndictcapprice=False):
    """
    Notes
    -----
    * Returns results in $/kWh, so that when summed over year they give $/kWac-yr
    """
    ### Load dictcapacityprice if necessary
    if dictcapacityprice is None:
        cappricepath = (
            revmpath+'io/capacityprice-CEMPNI-2010_2017-nominal-dict.p')
        with open(cappricepath, 'rb') as p:
            dictcapacityprice = pickle.load(p)
    
    if returndictcapprice is True:
        return dictcapacityprice
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    dfcapacityprice = mask.astype(bool).copy()
    for i, month in enumerate(months):
        imonth = i+1
        dfcapacityprice.loc['{} {}'.format(month, year)] = (
            mask.loc['{} {}'.format(month, year)]
            * dictcapacityprice[isonode, year, imonth])
    ### Divide by the number of critical intervals
    if mask.sum() > 0:
        dfcapacityprice = dfcapacityprice / mask.sum()
    else:
        dfcapacityprice = mask.copy() * 0.
    
    return dfcapacityprice.astype(float)

def capacityrevenue_pv_allnodes(
    year, PVsystem,
    isos=['CAISO','ERCOT','MISO','PJM','NYISO','ISONE'],
    percents=sorted([0.1, 0.2, 0.5, 1.14, 3.21, 0.11] 
                     + [12.51, 3.15, 4.20, 8.31, 7.04, 0.37]
                     + np.arange(1,31,1).tolist()),
    net='vre', resolution=60,
    savemod=None, runtype='full',
    outpath='out', savesafe=True, compress=False,
    ):
    """
    Inputs
    ------
    * criticalmethod: 

    Outputs
    -------

    Notes
    -----
    * 100 hrs is 1.14%, 10 hrs is 0.11%, 
    ISONE = 1096 hrs (12.51%), MISO = 276 hrs (3.15%),
    PJM = 368 hrs (4.20%), NYISO = 728 hrs (8.31%),
    average across these four is 7.04%,
    MISO wind is 32 hours (0.3653%)

    """

    ### Normalize inputs
    ## Put isos in lower case
    isos = [iso.upper() for iso in isos]
    ## Cut ERCOT if year < 2011
    if (year != 'tmy'):
        if (year <= 2010) and ('ercot' in isos):
            isos.remove('ercot')
    ## Check other inputs
    for iso in isos:
        iso = iso.upper()
        assert iso in ['CAISO', 'ERCOT', 'MISO', 'PJM', 'NYISO', 'ISONE']
    if savemod is not None:
        savemod = '-{}'.format(savemod)
    else:
        savemod = ''
    if resolution in [30, '30T', '30min', '30']:
        resolution = 30
    elif resolution in [60, '60T', '60min', '60', 'H', '1H', '1hour', 'hour']:
        resolution = 60
    else:
        raise Exception("resolution must be in [30, 60]")
    ## net
    net = net.lower()
    if net == 'solar':
        net = 'pv'
    elif net in ['both']:
        net = 'vre'
    elif net in ['pv', 'wind', 'vre']:
        pass
    else:
        raise Exception("net must be in ['pv','wind','vre']")

    ### savenames
    abbrevs = {
        'CAISO': 'C', 'ERCOT': 'E', 'MISO': 'M',
        'PJM': 'P', 'NYISO': 'N', 'ISONE': 'I',
        'APS': 'A', 'NV': 'V'}
    abbrev = ''.join([abbrevs[i.upper()] for i in isos])

    ### V0 means CA-only for CAISO
    savename = '{}/PVcaprevV1-{}-{}loadsun-{}-{}tilt-{}pcts-{}clip-{}min-net{}{}.csv'.format(
        outpath, abbrev, year,
        PVsystem.systemtype, PVsystem.axis_tilt,
        len(percents), PVsystem.clip,
        resolution, net,
        savemod)

    describename = os.path.splitext(savename)[0] + '-describe.txt'
    if compress is True:
        savename = savename+'.gz'
    if savesafe is True:
        savename = pvvm.toolbox.safesave(savename)
        describename = pvvm.toolbox.safesave(describename)
    print(savename)
    sys.stdout.flush()
    ### Make sure the folder exists
    if not os.path.isdir(os.path.dirname(savename)):
        raise Exception('{} does not exist'.format(os.path.dirname(savename)))

    ### Convenience variables
    hours = pvvm.toolbox.yearhours(year)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ### Load list of all nodes
    dfnodes = pd.read_csv(
        revmpath+'io/CEMPNI-nodes-eGRID-region.csv')
    
    ###### Load all capacity value data
    ### Load dictionary of capacity values by isonode
    ### Keys are (isonode, year, month)
    cappricepath = (
        revmpath+'io/capacityprice-CEMPNI-2010_2017-nominal-dict.p')
    with open(cappricepath, 'rb') as p:
        dictcapacityprice = pickle.load(p)

    ### Set up results containers
    results_to_concat = []

    ### Loop over ISOs
    for iso in isos:
        ### Glue together different ISO labeling formalisms
        (nsrdbindex, lmpindex, pnodeid,
            latlonindex, pnodename) = pvvm.io.glue_iso_columns(iso)

        ### Load node key (same for all isos and markets; assume 'da')
        dfinput = pvvm.io.get_iso_nodes(iso=iso)
        ### CAISO: Only do CA area
        if iso == 'CAISO':
            dfinput = dfinput.loc[dfinput.area=='CA'].reset_index(drop=True)

        ### Set NSRDB filepath
        nsrdbpath = '{}{}/in/NSRDB/{}/30min/'.format(
            revmpath, iso.upper(), year)

        ### Make NSRDB file list
        nsrdbfiles = list(dfinput[nsrdbindex])
        for i in range(len(nsrdbfiles)):
            nsrdbfiles[i] = str(nsrdbfiles[i]) + '-{}.gz'.format(year)

        ### Make isonode list
        isonodes = list(iso.upper() + ':' + dfinput[pnodeid].astype(str))

        ###### Get load and system-wide PV and store as dicts
        ### Load
        # dfload_in = dictload[iso,year].copy()
        dfload = pd.read_csv(
            revmpath+'io/iso-load_ferc/load-MWac-FERC2018-{}-{}.csv'.format(
                iso if iso != 'MISO' else 'MISO_SMEPA_Entergy_CLECO', 
                year),
            header=None, parse_dates=True, names=[iso], squeeze=True,
        ).tz_localize('UTC').tz_convert(pvvm.toolbox.tz_iso[iso])

        ### Net load
        inpath = {
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
        }[net]
        dfnetload = pd.read_csv(
            inpath,
            header=None, parse_dates=True, names=[iso], squeeze=True,
        ).tz_localize('UTC').tz_convert(pvvm.toolbox.tz_iso[iso])

        # ### PV generation
        # dfsystemsolar_in = dictsystemsolar[iso,year][iso].copy()

        # ### Lineup
        # dfload, dfsystemsolar = timeserieslineup(
        #     dfload_in, dfsystemsolar_in, resamplemethod='interpolate', 
        #     resolution1=60, resolution2=30,
        #     tzout=None)

        ###### Also get  ISO-defined critical hours
        if iso in ['MISO', 'PJM', 'NYISO', 'ISONE']:
            mask_iso = get_iso_critical_hours(
                iso, year, resolution='{}min'.format(resolution))
            
        ###### Make peak-load and peak-net-load masks
        ###### (timezone is ISO default, index is from load)
        ### Loop over percents
        masks, masks_netvre = {}, {}
        for percent in percents:
            ### Critical hours
            numhours = nhours(percent, year) * int(60/resolution)

            ### Total load
            index = dfload.index.copy()
            dfloadi = dfload.reset_index(drop=True)
            indexi = dfloadi.index.copy()
            mask = dfloadi.nlargest(numhours).index
            mask = pd.Series(indexi.map(lambda x: x in mask))
            mask.index = index

            ### Net load
            # dfnetload = dfload - dfsystemsolar
            index = dfnetload.index.copy()
            dfload_netvrei = dfnetload.reset_index(drop=True)
            indexi = dfload_netvrei.index.copy()
            mask_netvre = dfload_netvrei.nlargest(numhours).index
            mask_netvre = pd.Series(indexi.map(lambda x: x in mask_netvre))
            mask_netvre.index = index
            
            ### Store in dict
            masks['{:.2f}'.format(percent)] = mask
            masks_netvre['{:.2f}'.format(percent)] = mask_netvre

        #######################
        ### CALCULATE VALUE ###

        ### Determine runlength
        if runtype == 'full':
            runlength = len(dfinput)
        elif runtype == 'test':
            runlength = 5
        elif type(runtype) == int:
            runlength = runtype
        else:
            raise Exception("Invalid runtype. Options: ['full', 'test', int]")

        ### Loop over nodes
        for node in trange(runlength, desc=iso.upper()):
            ### Get info about row
            row = dfinput.loc[node]
            isonode = isonodes[node]
            
            ###### IMPORTANT: Bypass this node if it isn't contained in 
            ###### capacity value dictionary (because that means we 
            ###### don't have a capacity value for it)
            if (isonode, year, 1) not in dictcapacityprice.keys():
                continue
            ###########################################################
            
            ### Simulate output
            output_ac, tznode = PVsystem.sim(
                nsrdbfile=nsrdbfiles[node], year=year, nsrdbpathin=nsrdbpath,
                output_ac_only=False, output_ac_tz=True
            )
            ### Convert to CF
            output_ac = output_ac / 1000

            ####### Downsample to 30min resolution, if necessary
            if resolution is 60:
                output_ac = downsample_trapezoid(
                    output_ac, output_freq='H')
            elif resolution is 30:
                pass
            
            ### Get the average cap val in case we need it later
            capvalmean = sum(
                [dictcapacityprice[isonode, year, i] for i in range(1,13)]
            ) / 12
            
            ###### Calculate results for percentage of top load hours
            for percent in percents:
                ### Set up masks
                mask = masks['{:.2f}'.format(percent)]
                mask_netvre = masks_netvre['{:.2f}'.format(percent)]
                
                ###### Set up capacity value series
                ### Start with the masks
                capval = mask.copy()
                capval_netvre = mask_netvre.copy()
                
                ### Set each month equal to historical capacity value
                for i, month in enumerate(months):
                    imonth = i+1
                    capval.loc['{} {}'.format(month, year)] = (
                        mask.loc['{} {}'.format(month, year)]
                        * dictcapacityprice[isonode, year, imonth]
                    )
                    capval_netvre.loc['{} {}'.format(month, year)] = (
                        mask_netvre.loc['{} {}'.format(month, year)]
                        * dictcapacityprice[isonode, year, imonth]
                    )
                ### Divide by the number of critical intervals
                capval = capval / mask.sum()
                capval_netvre = capval_netvre / mask_netvre.sum()
                
                ###### Easy way, if both are in same timezone
                try:
                    ### Capacity credit, for reference
                    capcredit = output_ac.loc[mask].mean()
                    capcredit_netvre = output_ac.loc[mask_netvre].mean()
                    
                    ### Calculate the yearly capacity revenue in $/kWac-yr
                    caprev = (output_ac * capval).sum()
                    caprev_netvre = (output_ac * capval_netvre).sum()
                    
                ###### Hard/slow way, if in different timezones
                except pd.core.indexing.IndexingError:
                    ###### Capacity credit, for reference
                    dfcap = (pd.DataFrame(output_ac)
                     .tz_convert(pvvm.toolbox.tz_iso[iso])
                     .merge(pd.DataFrame(mask), 
                            left_index=True, right_index=True, how='inner',
                            suffixes=('_gen','_crit'))
                    )
                    capcredit = dfcap.loc[
                        dfcap['0_crit'], '0_gen'].mean()
                    
                    dfcap_netvre = (pd.DataFrame(output_ac)
                     .tz_convert(pvvm.toolbox.tz_iso[iso])
                     .merge(pd.DataFrame(mask_netvre), 
                            left_index=True, right_index=True, how='inner',
                            suffixes=('_gen','_crit'))
                    )
                    capcredit_netvre = dfcap_netvre.loc[
                        dfcap_netvre['0_crit'], '0_gen'].mean()
                    
                    ###### Capacity revenue
                    dfrev = (pd.DataFrame(output_ac)
                     .tz_convert(pvvm.toolbox.tz_iso[iso])
                     .merge(pd.DataFrame(capval), 
                            left_index=True, right_index=True, how='inner',
                            suffixes=('_gen','_crit'))
                    )
                    caprev = (dfrev['0_gen'] * dfrev['0_crit']).sum()
                    
                    dfrev_netvre = (pd.DataFrame(output_ac)
                     .tz_convert(pvvm.toolbox.tz_iso[iso])
                     .merge(pd.DataFrame(capval_netvre), 
                            left_index=True, right_index=True, how='inner',
                            suffixes=('_gen','_crit'))
                    )
                    caprev_netvre = (
                        dfrev_netvre['0_gen'] * dfrev_netvre['0_crit']).sum()

                ### Write results to output list
                results_node = [iso.upper(), isonode, 
                                dfinput.loc[node,pnodeid],
                                dfinput.loc[node,pnodename], 
                                row['latitude'], row['longitude'], 
                                dfinput.loc[node,latlonindex],
                                tznode, 
                                percent,
                                capcredit, 
                                capcredit_netvre,
                                caprev, 
                                caprev_netvre,
                                dictcapacityprice[isonode, year, 6],
                                capvalmean,
                               ]

                results_to_concat.append(results_node)

            ###### Do it all again for ISO-defined critical hours
            if iso in ['MISO', 'PJM', 'NYISO', 'ISONE']:
                ### Start with the mask
                capval_iso = mask_iso.copy()
                
                ### Set each month equal to historical capacity value
                for i, month in enumerate(months):
                    imonth = i+1
                    capval_iso.loc['{} {}'.format(month, year)] = (
                        mask_iso.loc['{} {}'.format(month, year)]
                        * dictcapacityprice[isonode, year, imonth]
                    )
                ### Divide by the number of critical intervals
                capval_iso = capval_iso / mask_iso.sum()
                
                ### Easy way, if both are in same timezone
                try:
                    ### Capacity credit, for reference
                    capcredit_iso = output_ac.loc[mask_iso.astype(bool)].mean()
                    
                    ### Calculate the yearly capacity revenue in $/kWac-yr
                    caprev_iso = (output_ac * capval_iso).sum()

                ### Hard/slow way, if in different timezones
                except pd.core.indexing.IndexingError:                    
                    ###### Capacity credit, for reference
                    dfcap_iso = (pd.DataFrame(output_ac)
                     .tz_convert(pvvm.toolbox.tz_iso[iso])
                     .merge(pd.DataFrame(mask_iso), 
                            left_index=True, right_index=True, how='inner',
                            suffixes=('_gen','_crit'))
                    )
                    capcredit_iso = dfcap_iso.loc[
                        dfcap_iso['0_crit'].astype(bool), '0_gen'].mean()
                    
                    ###### Capacity revenue
                    dfrev_iso = (pd.DataFrame(output_ac)
                     .tz_convert(pvvm.toolbox.tz_iso[iso])
                     .merge(pd.DataFrame(capval_iso), 
                            left_index=True, right_index=True, how='inner',
                            suffixes=('_gen','_crit'))
                    )
                    caprev_iso = (
                        dfrev_iso['0_gen'] * dfrev_iso['0_crit']).sum()

                ### Write results to output list
                results_node = [iso.upper(), isonode, 
                                dfinput.loc[node,pnodeid],
                                dfinput.loc[node,pnodename], 
                                row['latitude'], row['longitude'], 
                                dfinput.loc[node,latlonindex],
                                tznode, 
                                '', # percent
                                capcredit_iso, 
                                '', # capcredit_netvre
                                caprev_iso,
                                '', # caprev_netvre
                                dictcapacityprice[isonode, year, 6],
                                capvalmean,
                               ]

                results_to_concat.append(results_node)
            
    ### Write output dataframe from output dictionary
    columns = [
        'ISO', 'ISO:Node', 'PNodeID', 'PNodeName',
        'Latitude', 'Longitude', 'LatLonIndex', 'Timezone', 
        'Percent', 'CapCredit', 'CapCreditNetVRE',
        'CapRevenue', 'CapRevenueNetVRE',
        'CapValJun', 'CapValMonthlyMean',]
    dfout = pd.DataFrame(data=np.array(results_to_concat), columns=columns)

    ###### Write results
    ### Results
    if compress is False:
        dfout.to_csv(savename, index=False)
    elif compress is True:
        dfout.to_csv(savename, index=False, compression='gzip')
    ### Description
    pvdict = PVsystem.__dict__
    with open(describename, 'w') as f:
        f.write('datetime of run = {}\n'.format(pvvm.toolbox.nowtime()))
        f.write('savename of run = {}\n'.format(savename))
        f.write('script =          {}\n'.format(os.path.basename(__file__)))
        f.write('\n')
        f.write('isos =            {}\n'.format(','.join(isos)))
        f.write('year =            {}\n'.format(year))
        f.write('resolution =      {}\n'.format(resolution))
        f.write('percents =        {}\n'.format(
            ','.join([str(percent) for percent in percents])))
        f.write('net =             {}\n'.format(net))
        f.write('resolution =      {}\n'.format(resolution))
        f.write('\n')
        f.write('\nPVsystem\n')
        f.write('--------\n')
        for key in pvdict:
            if type(pvdict[key]) in [str, int, float, bool]:
                f.write('{:<18} = {}\n'.format(key, pvdict[key]))
        f.write('\n')
        f.write('revmpath =        {}\n'.format(revmpath))
        f.write('datapath =        {}\n'.format(datapath))
        f.write('runtype =         {}\n'.format(runtype))
        f.write('\n')
        f.write("results_units =   'PNodeID': int\n")
        f.write("results_units =   'PNodeName': str\n")
        f.write("results_units =   'Latitude': float\n")
        f.write("results_units =   'Longitude': float\n")
        f.write("results_units =   'LatLonIndex': int\n")
        f.write("results_units =   'Timezone': str (Etc/GMT+{})\n")
        f.write("results_units =   'CapCredit': fraction\n")
        f.write("results_units =   'CapCreditNetvre': fraction\n")
    
    ### Return results
    return dfout


###################
### Emissions value

def getemissions(
    year=None, region=None, pollutant='co2',
    emissions='marginal', measurement='emissions',
    source='easiur',
    tz='UTC', dollaryear=2017,
    dfemissions=None, returndfemissions=False,
    dfemissionspath=None):
    """
    Inputs
    ------
    year: int in 2006..2017
    region: str in [
        'AZNM', 'CAMX', 'ERCT', 'FRCC', 'MROE', 
        'MROW', 'NEWE', 'NWPP', 'NYCW', 'NYLI', 
        'NYUP', 'RFCE', 'RFCM', 'RFCW', 'RMPA', 
        'SPNO', 'SPSO', 'SRMV', 'SRMW', 'SRSO', 
        'SRTV', 'SRVC']
    pollutant: str in ['co2','so2','nox','pm25']
    emissions: str in ['marginal', 'average']
    measurement: str in ['emissions', 'damages']
    tz: str. Examples: ['UTC', 'Etc/GMT+5']
    dollaryear: int. Default is 2017.
    dfemissions: pd.DataFrame (3x faster if provided)
    returndfemissions: bool in [True, False]
    dfemissionspath: None or str
    
    Units
    -----
    emissions: [kg/MWh]
    damages: output in [dollaryear$/MWh] (input from CEDM in [2010$/MWh])
    
    Source
    ------
    https://cedm.shinyapps.io/MarginalFactors/
    Azevedo IL, Horner NC, Siler-Evans K, Vaishnav PT (2017). 
    Electricity Marginal Factor Estimates. 
    Center For Climate and Energy Decision Making. 
    Pittsburgh: Carnegie Mellon University. http://cedmcenter.org

    Notes from source (updated 20181124)
    * Emissions factor values are in kg/MWh.
    * Damage factor values are in $/MWh (reported in 2010 dollars).
    * Hour of day and seasonal hour of day estimates are reported in 
        UTC-5 (NOT the local time zone).
    * The breakdown for seasonal factors is: 
        Winter (Nov-Mar), Summer (May-Sep), and Transition (Apr & Oct).

    Procedure
    ---------
    * Download files from site listed above for all regions, years, pollutants 
    and save in (datapath+'CEDM/') folder as:
    [
        'Generation.MAR.DAMAP2.egrid.bySeasonalTOD.csv',
        'Generation.MAR.DAMEASIUR.egrid.bySeasonalTOD.csv',
        'Generation.MAR.EMIT.egrid.bySeasonalTOD.csv',
        'Generation.AVG.DAMAP2.egrid.bySeasonalTOD.csv',
        'Generation.AVG.DAMEASIUR.egrid.bySeasonalTOD.csv',
        'Generation.AVG.EMIT.egrid.bySeasonalTOD.csv',
    ]

    """
    ### Load dfemissions if necessary
    if dfemissions is None:
        ### Set default path for input dataframe
        if dfemissionspath is None:
            dfemissionspath = (
                datapath+'CEDM/egrid-emissions_damages-seasonal_tod.p')

        if not os.path.exists(dfemissionspath):
            ###### Save the pickled dataframe from raw CEDM files
            ###### (should only have to do this once)
            ### Load each file
            dfmoer = pd.read_csv(
                datapath+'CEDM/Generation.MAR.EMIT.egrid.bySeasonalTOD.csv',
                usecols=list(range(1,11)))
            dfmoer['emissions'] = 'marginal'
            dfmoer['measurement'] = 'emissions'

            dfaer = pd.read_csv(
                datapath+'CEDM/Generation.AVG.EMIT.egrid.bySeasonalTOD.csv',
                usecols=list(range(1,7)))
            dfaer['emissions'] = 'average'
            dfaer['measurement'] = 'emissions'

            dfmdam_easiur = pd.read_csv(
                datapath+'CEDM/Generation.MAR.DAMEASIUR.egrid.bySeasonalTOD.csv',
                usecols=list(range(1,11)))
            dfmdam_easiur['emissions'] = 'marginal'
            dfmdam_easiur['measurement'] = 'damages'
            dfmdam_easiur['source'] = 'easiur'

            dfmdam_ap2 = pd.read_csv(
                datapath+'CEDM/Generation.MAR.DAMAP2.egrid.bySeasonalTOD.csv',
                usecols=list(range(1,11)))
            dfmdam_ap2['emissions'] = 'marginal'
            dfmdam_ap2['measurement'] = 'damages'
            dfmdam_ap2['source'] = 'ap2'

            dfadam_easiur = pd.read_csv(
                datapath+'CEDM/Generation.AVG.DAMEASIUR.egrid.bySeasonalTOD.csv',
                usecols=list(range(1,7)))
            dfadam_easiur['emissions'] = 'average'
            dfadam_easiur['measurement'] = 'damages'
            dfadam_easiur['source'] = 'easiur'

            dfadam_ap2 = pd.read_csv(
                datapath+'CEDM/Generation.AVG.DAMAP2.egrid.bySeasonalTOD.csv',
               usecols=list(range(1,7)))
            dfadam_ap2['emissions'] = 'average'
            dfadam_ap2['measurement'] = 'damages'
            dfadam_ap2['source'] = 'ap2'

            dfin = pd.concat([dfmoer, dfaer, dfmdam_easiur, dfmdam_ap2,
                              dfadam_easiur, dfadam_ap2],
                             sort=False, ignore_index=True)
            dfin.warn = dfin.warn.fillna(0).astype(int)
            for datum in ['region','season','pollutant','emissions','measurement']:
                dfin[datum] = dfin[datum].astype('category').copy()

            ### Save it
            savename = datapath+'CEDM/egrid-emissions_damages-seasonal_tod'
            with open(savename+'.p', 'wb') as p:
                pickle.dump(dfin, p)

            dfin.to_csv(savename+'.csv', index=False)

        ### Load it
        with open(dfemissionspath, 'rb') as p:
            dfemissions = pickle.load(p)

    ### Return full dictionary if desired
    if returndfemissions is True:
        return dfemissions

    ### Check inputs
    assert year in range(2006,2018)
    assert region in [
        'AZNM', 'CAMX', 'ERCT', 'FRCC', 'MROE', 
        'MROW', 'NEWE', 'NWPP', 'NYCW', 'NYLI', 
        'NYUP', 'RFCE', 'RFCM', 'RFCW', 'RMPA', 
        'SPNO', 'SPSO', 'SRMV', 'SRMW', 'SRSO', 
        'SRTV', 'SRVC']
    assert pollutant in ['co2', 'so2', 'nox', 'pm25']
    assert emissions in ['marginal', 'average']
    assert measurement in ['emissions', 'damages']
    if measurement == 'damages': assert source in ['easiur', 'ap2']

    ### Get subset dataframe to tile
    dfsub = dfemissions.loc[(dfemissions.year==year)
                     &(dfemissions.region==region)
                     &(dfemissions.pollutant==pollutant)
                     &(dfemissions.emissions==emissions)
                     &(dfemissions.measurement==measurement)
                    ].sort_values('hour')
    ### If necessary, get damages from specific source
    if measurement == 'damages':
        dfsub = dfsub.loc[dfsub.source==source]
    
    ### chunk1: 12/31 - 2/28
    index = pd.date_range(
        '{}-12-31'.format(year-1), 
        '{}-04-01'.format(year),
        freq='H', closed='left', tz='Etc/GMT+5')
    assert len(index) % 24 == 0, 'len(index) % 24 != 0'
    data = np.tile(dfsub.loc[dfsub.season=='Winter','factor'].values,
                     int(len(index)/24))
    chunk1 = pd.Series(index=index, data=data)

    ### chunk2: 3/1 - 4/30
    index = pd.date_range(
        '{}-04-01'.format(year), 
        '{}-05-01'.format(year),
        freq='H', closed='left', tz='Etc/GMT+5')
    assert len(index) % 24 == 0, 'len(index) % 24 != 0'
    data = np.tile(dfsub.loc[dfsub.season=='Trans','factor'].values,
                     int(len(index)/24))
    chunk2 = pd.Series(index=index, data=data)

    ### chunk3: 5/1 - 8/31
    index = pd.date_range(
        '{}-05-01'.format(year), 
        '{}-10-01'.format(year),
        freq='H', closed='left', tz='Etc/GMT+5')
    assert len(index) % 24 == 0, 'len(index) % 24 != 0'
    data = np.tile(dfsub.loc[dfsub.season=='Summer','factor'].values,
                     int(len(index)/24))
    chunk3 = pd.Series(index=index, data=data)

    ### chunk4: 9/1 - 11/30
    index = pd.date_range(
        '{}-10-01'.format(year), 
        '{}-11-01'.format(year),
        freq='H', closed='left', tz='Etc/GMT+5')
    assert len(index) % 24 == 0, 'len(index) % 24 != 0'
    data = np.tile(dfsub.loc[dfsub.season=='Trans','factor'].values,
                     int(len(index)/24))
    chunk4 = pd.Series(index=index, data=data)

    ### chunk5: 12/1 - 1/1
    index = pd.date_range(
        '{}-11-01'.format(year), 
        '{}-01-02'.format(year+1),
        freq='H', closed='left', tz='Etc/GMT+5')
    assert len(index) % 24 == 0, 'len(index) % 24 != 0'
    data = np.tile(dfsub.loc[dfsub.season=='Winter','factor'].values,
                     int(len(index)/24))
    chunk5 = pd.Series(index=index, data=data)

    ### Join it
    dsout = pd.concat([chunk1, chunk2, chunk3, chunk4, chunk5])

    ### Convert to ISO timezone
    isoindex = pd.date_range(
        '{}-01-01'.format(year), '{}-01-01'.format(year+1),
        freq='H', closed='left', tz=tz)
    isoindex = pd.DataFrame(index=isoindex)

    ### Take chunk that overlaps with year of interest
    dsout = pd.DataFrame(dsout.rename(pollutant)).merge(
        isoindex, left_index=True, right_index=True, how='right'
    )[pollutant].tz_convert(tz)

    ### Convert dollars to dollaryear if necessary
    if measurement == 'damages':
        dsout = dsout * inflate(2010, dollaryear)
    
    return dsout

def emissionsoffset(
    dsgen, region=None, pollutant='co2',
    emissions='marginal', measurement='emissions',
    source='easiur',
    tz=None, dollaryear=2017,
    dfemissions=None, dfemit=None, 
    resolutiongen=None,
    resamplemethod='interpolate',
    mismatch='raise',
    results='offset'):
    """
    Inputs
    ------
    dsgen: timeseries. Should be in AC CF, with max of 1.

    Outputs
    -------
    yearlyoffset: float. Yearly emissions displacement in kg/kWac-yr
        or damages offset in $/kWac-yr.

    Notes
    -----
    * dsgen and dfemit should be tz-aware
    * dfemit should come from getemissions()
    * year is pulled from dsgen index, so if using TMY need to reindex
    """
    
    ### Get year (from middle of dsgen)
    year = dsgen.index[int(len(dsgen)/2)].year
    hours = pvvm.toolbox.yearhours(year)
    
    ### Get tz from dsgen if necessary
    if tz is None:
        tz = str(dfpv.index.tz)
        
    ### Check inputs
    assert year in range(2006,2018)
    assert region in [
        'AZNM', 'CAMX', 'ERCT', 'FRCC', 'MROE', 
        'MROW', 'NEWE', 'NWPP', 'NYCW', 'NYLI', 
        'NYUP', 'RFCE', 'RFCM', 'RFCW', 'RMPA', 
        'SPNO', 'SPSO', 'SRMV', 'SRMW', 'SRSO', 
        'SRTV', 'SRVC']
    assert pollutant in ['co2', 'so2', 'nox', 'pm25']
    assert emissions in ['marginal', 'average']
    assert measurement in ['emissions', 'damages']
    if measurement == 'damages': assert source in ['easiur', 'ap2']
    
    ### Get emissions timeseries if necessary
    if dfemit is None:
        dfemit = getemissions(
            year=year, region=region, pollutant=pollutant,
            emissions=emissions, measurement=measurement,
            source=source, tz=tz, dollaryear=dollaryear,
            dfemissions=dfemissions)
    
    ### Line up the data series
    dssol, dsemit = timeserieslineup(
        dsgen, dfemit, 
        resolution1=resolutiongen, resolution2='H',
        resamplemethod=resamplemethod, tzout='none',
        mismatch=mismatch)
    # assert dsgen.index.tz == dfemit.index.tz, \
    # "Mismatched tz: {}, {}".format(dsgen.index.tz, dfemit.index.tz)

    ### Calculate results
    ## [(kW/kWac) * (kg / MWh) * (1 MW / 1000 kW) * (hr/sample) per yr] = [kg / kWac-yr]
    ## [(kW/kWac) * ( $ / MWh) * (1 MW / 1000 kW) * (hr/sample) per yr] = [ $ / kWac-yr]
    offset_yearly = ((dssol * dsemit).sum() 
                    / 1000 * hours / len(dsgen))
    
    ### Return this number if that's all you want
    if results in ['offset','offset_yearly','single','scalar','yearlyoffset']:
        return offset_yearly

    ### Otherwise return dict of interesting values
    elif results in ['all','dict','full']:
        ## Average emissions [kg/MWh]
        emissions_average = dsemit.mean()
        ## Capacity factor [MWh/MWac/h]
        generation_yearly = dssol.sum() * hours / len(dsgen) 
        ## ^ same as multiplying by resolutiongen / 60
        capacity_factor = generation_yearly / hours
        ## Average emissions offset [kg/MWh]
        offset_average = offset_yearly / generation_yearly * 1000
        ## Value factor [(kg/MWh)/(kg/MWh)]
        try:
            value_factor = offset_average / emissions_average
        except ZeroDivisionError:
            value_factor = np.nan
        ### Return output dictionary
        out = {'offset_yearly': offset_yearly,
               'capacity_factor': capacity_factor,
               'emissions_average': emissions_average,
               'offset_average': offset_average,
               'value_factor': value_factor}
        return out

def emissionsoffset_allnodes(
    yearemissions, PVsystem,
    isos=['CAISO', 'ERCOT', 'MISO', 'PJM', 'NYISO', 'ISONE'],
    pollutant='co2',
    emissions='marginal', measurement='emissions',
    source='easiur', dollaryear=2017,
    yearsun=None,
    savemod=None, runtype='full',
    outpath='out', savesafe=True,
    ):
    """
    Inputs
    ------

    Outputs
    -------

    """

    ### Normalize inputs
    ## Put isos in lower case
    isos = [iso.lower() for iso in isos]
    ## Set solar resolution
    if yearsun is None:
        yearsun = yearemissions
    if yearsun == 'tmy':
        resolution = 60
    elif type(yearsun) == int:
        resolution = 30
    else:
        print(year)
        raise Exception("year must by 'tmy' or int")
    ## Cut ERCOT if year < 2011
    if (yearsun != 'tmy'):
        if (yearsun <= 2010) and ('ercot' in isos):
            isos.remove('ercot')
    ## Check other inputs
    for iso in isos:
        assert iso in ['caiso', 'ercot', 'miso', 'pjm', 'nyiso', 'isone']
    assert pollutant in ['co2', 'so2', 'nox', 'pm25']
    assert emissions in ['marginal', 'average']
    assert measurement in ['emissions', 'damages']
    if measurement == 'damages': assert source in ['easiur', 'ap2']
    if savemod is not None:
        savemod = '-{}'.format(savemod)
    else:
        savemod = ''

    ### savenames
    abbrevs = {
        'caiso': 'C', 'ercot': 'E', 'miso': 'M',
        'pjm': 'P', 'nyiso': 'N', 'isone': 'I'}
    abbrev = ''.join([abbrevs[i.lower()] for i in isos])

    ### v0: only yearly offset
    ### v1: yearly offset, average offset, average emissions, CF, VF
    savename = '{}/PVoffset_CEDMv1-{}-{}em-{}sun-{}-{}tilt-{}-{}-{}{}.csv'.format(
        outpath, abbrev, yearemissions, yearsun,
        PVsystem.systemtype, PVsystem.axis_tilt,
        pollutant,
        {'marginal':'marg','average':'ave'}[emissions],
        {'emissions':'em','damages':'dam_{}'.format(source)}[measurement],
        savemod)

    describename = os.path.splitext(savename)[0] + '-describe.txt'
    if savesafe == True:
        savename = pvvm.toolbox.safesave(savename)
        describename = pvvm.toolbox.safesave(describename)
    print(savename)
    sys.stdout.flush()
    ### Make sure the folder exists
    if not os.path.isdir(os.path.dirname(savename)):
        raise Exception('{} does not exist'.format(os.path.dirname(savename)))

    ### Convenience variables
    hours = pvvm.toolbox.yearhours(yearemissions)

    ### Load node-egrid region lineup
    dfnodes = pd.read_csv(
        revmpath+'io/CEMPNI-nodes-eGRID-region.csv')
    ### Make lineup dict
    isonode2egrid = dict(zip(dfnodes['ISO:Node'],dfnodes['eGRID']))
    
    ### Load all emissions data for faster function operation
    dfemissions = getemissions(returndfemissions=True)

    ### Set up results containers
    results_to_concat = []

    ### Loop over ISOs
    for iso in isos:
        ### Glue together different ISO labeling formalisms
        (nsrdbindex, lmpindex, pnodeid,
            latlonindex, pnodename) = pvvm.io.glue_iso_columns(iso)

        ### Load node key (same for all isos and markets; assume 'da')
        dfinput = pvvm.io.get_iso_nodes(iso=iso)

        ### Set NSRDB filepath
        nsrdbpath = '{}{}/in/NSRDB/{}/{}min/'.format(
            revmpath, iso.upper(), yearsun, resolution)

        ### Make NSRDB file list
        nsrdbfiles = list(dfinput[nsrdbindex])
        for i in range(len(nsrdbfiles)):
            nsrdbfiles[i] = str(nsrdbfiles[i]) + '-{}.gz'.format(yearsun)

        ### Make isonode list
        isonodes = list(iso.upper() + ':' + dfinput[pnodeid].astype(str))

        ### Get new index if year == 'tmy'
        if yearsun == 'tmy':
            newindex = pd.date_range(
                '{}-01-01'.format(yearemissions), 
                '{}-01-01'.format(yearemissions+1),
                freq='H', closed='left', tz=pvvm.toolbox.tz_iso[iso])
            ### Drop leap day if necessary
            if pvmm.toolbox.yearhours(yearemissions) == 8784:
                newindex = dropleap(
                    dfin=newindex, year=yearemissions, resolution=60)

        ### Determine runlength
        if runtype == 'full':
            runlength = len(dfinput)
        elif runtype == 'test':
            runlength = 5
        elif type(runtype) == int:
            runlength = runtype
        else:
            raise Exception("Invalid runtype. Must be 'full', 'test', or int.")

        #######################
        ### CALCULATE VALUE ###

        ### Loop over nodes
        for i in trange(runlength, desc=iso.upper()):
            ### Get info about row
            row = dfinput.loc[i]
            isonode = isonodes[i]
            region = isonode2egrid[isonode]
            
            ### Simulate output
            output_ac, tznode = PVsystem.sim(
                nsrdbfile=nsrdbfiles[i], year=yearsun, nsrdbpathin=nsrdbpath,
                output_ac_only=False, output_ac_tz=True
            )
            ### Convert to CF
            output_ac = output_ac / 1000
            
            ### If year == 'tmy', convert to 2001
            if yearsun == 'tmy':
                output_ac.index = newindex
            
            ### Calculate results
            ########## DEBUG
            try:
                offsets = emissionsoffset(
                    dsgen=output_ac, region=region, pollutant=pollutant,
                    emissions=emissions, measurement=measurement,
                    source=source,
                    tz=pvvm.toolbox.tz_iso[iso], dollaryear=dollaryear,
                    dfemissions=dfemissions, dfemit=None, 
                    resolutiongen=resolution,
                    resamplemethod='interpolate',
                    mismatch='raise', results='all')
            except ValueError:
                print(isonode, tznode, tz_iso[iso], region)
                print(output_ac.index.map(lambda x: x.year).unique())
                print(output_ac.head())
                print(output_ac.tail())
                raise Exception('Something is messed up')

            results_node = [iso.upper(), isonode, 
                            dfinput.loc[i,pnodeid],
                            dfinput.loc[i,pnodename], 
                            row['latitude'], row['longitude'], 
                            dfinput.loc[i,latlonindex],
                            tznode, region, 
                            offsets['capacity_factor'],
                            offsets['offset_yearly'], 
                            offsets['emissions_average'],
                            offsets['offset_average'],
                            offsets['value_factor'],
                            ]

            ### Write results to output list
            results_to_concat.append(results_node)

    ### Write output dataframe from output dictionary
    columns = [
        'ISO', 'ISO:Node', 'PNodeID', 'PNodeName',
        'Latitude', 'Longitude', 'LatLonIndex', 'Timezone', 'Region',
        'CapacityFactor', 'OffsetYearly', 'EmissionsAverage', 
        'OffsetAverage', 'ValueFactor']
    dfout = pd.DataFrame(data=np.array(results_to_concat), columns=columns)
    ### Add additional fields
    dfout['pollutant'] = pollutant
    dfout['emissions'] = emissions
    dfout['measurement'] = measurement
    dfout['source']= source

    ###### Write results
    ### Results
    dfout.to_csv(savename, index=False)
    ### Description
    pvdict = PVsystem.__dict__
    with open(describename, 'w') as f:
        f.write('datetime of run = {}\n'.format(pvvm.toolbox.nowtime()))
        f.write('savename of run = {}\n'.format(savename))
        f.write('script =          {}\n'.format(os.path.basename(__file__)))
        f.write('\n')
        f.write('isos =            {}\n'.format(abbrev))
        f.write('yearemissions =   {}\n'.format(yearemissions))
        f.write('yearsun =         {}\n'.format(yearsun))
        f.write('resolution =      {}\n'.format(resolution))
        f.write('\n')
        f.write('\nPVsystem\n')
        f.write('--------\n')
        for key in pvdict:
            if type(pvdict[key]) in [str, int, float, bool]:
                f.write('{:<18} = {}\n'.format(key, pvdict[key]))
        f.write('\n')
        f.write('pollutant =       {}\n'.format(pollutant))
        f.write('emissions =       {}\n'.format(emissions))
        f.write('measurement =     {}\n'.format(measurement))
        f.write('source =          {}\n'.format(source))
        f.write('dollaryear =      {}\n'.format(dollaryear))
        f.write('\n')
        f.write('revmpath =        {}\n'.format(revmpath))
        f.write('datapath =        {}\n'.format(datapath))
        f.write('runtype =         {}\n'.format(runtype))
        f.write('\n')
        f.write("results_units =   'PNodeID': int\n")
        f.write("results_units =   'PNodeName': str\n")
        f.write("results_units =   'Latitude': float\n")
        f.write("results_units =   'Longitude': float\n")
        f.write("results_units =   'LatLonIndex': int\n")
        f.write("results_units =   'Timezone': str (Etc/GMT+{})\n")
        f.write("results_units =   'CapacityFactor': [MWh/MWac/h]\n")
        if measurement == 'emissions':
            f.write("results_units =   'OffsetYearly': [kg/kWac-yr]\n")
            f.write("results_units =   'EmissionsAverage': [kg/MWh]\n")
            f.write("results_units =   'OffsetAverage': [kg/MWh]\n")
            f.write("results_units =   'ValueFactor': [(kg/MWh)/(kg/MWh)]\n")
        elif measurement == 'damages':
            f.write("results_units =   'OffsetYearly': [$/kWac-yr]\n")
            f.write("results_units =   'EmissionsAverage': [$/MWh]\n")
            f.write("results_units =   'OffsetAverage': [$/MWh]\n")
            f.write("results_units =   'ValueFactor': [($/MWh)/($/MWh)]\n")
    
    ### Return results
    return dfout

#############
### STATS ###

def errorstats(calc, meas, returndict=False):
    """
    Inputs
    ------
    calc: calculated/simulated values
    meas: actual measured values

    Returns
    -------
    tuple of:
    * n: Number of observations
    * CC: Pearson correlation coefficient [fraction]
    * MAE: Mean absolute error
    * MBE: Mean bias error
    * rMBE: Relative mean bias error
    * RMSE: Root mean squared error
    * rRMSE: Relative room mean squared error

    Dataframe columns
    -----------------
    columns = ['n', 'CC', 'MAE', 'MBE', 'rMBE', 'RMSE', 'rRMSE']
    columnlabels = ['n [# obs]', 'CC [fraction]', 
                    'MAE [% CF]', 'MBE [% CF]',
                    'rMBE [%]', 'RMSE [% CF]', 'rRMSE [%]']
    """
    error = calc - meas
    rerror = (calc - meas) / np.abs(meas)
    n = len(error)

    data = pd.DataFrame(
        [meas, calc, error, rerror], 
        index=['meas', 'calc', 'error', 'rerror']).T
    data.drop(data[data.rerror == np.inf].index, inplace=True)

    corrcoef = np.corrcoef(meas, calc)[0][1]  # Pearson correlation coefficient
    mae = np.abs(error).sum() / n             # mean absolute error (MAE)
    mbe = error.sum() / n                     # mean bias error (MBE)
    rmse = np.sqrt((error**2).sum() / n)      # root mean square error (RMSE)
    rmbe = mbe / meas.sum() * n               # relative MBE (NREL 2017)
    rrmse = rmse / np.sqrt((meas**2).sum() / n)  # relative RMSE (NREL 2017)

    out = (n, corrcoef, mae, mbe, rmbe*100, rmse, rrmse*100)
    if returndict:
        return dict(zip(
            ('n', 'cc', 'mae', 'mbe', 'rmbe', 'rmse', 'rrmse'),
            out))
    else:
        return out

##################
### FINANCIALS ###

##########
### Common

def crf(wacc, lifetime):
    """
    Inputs
    -----
    wacc: expressed as a percent, not a fraction
    lifetime: econommic lifetime in years
    """
    out = ((wacc*0.01 * (1 + wacc*0.01) ** lifetime) 
           / ((1 + wacc*0.01) ** lifetime - 1))
    return out

def inflate(yearin=None, yearout=2017, value=None):
    """
    License (public domain)
    ------
    https://www.bls.gov/bls/linksite.htm

    Usage
    -----
    * Download input data from https://data.bls.gov/timeseries/CUUR0000SA0.
    Select years from 1913--2018 and include annual averages, then 
    download .xlsx file and save at revmpath+'Data/BLS/inflation_cpi_level.xlsx'
    """
    ### Set filepaths
    file_inflation_level = datapath + 'BLS/inflation_cpi_level.xlsx'
    
    ### Load the file
    try:
        dsinflation = pd.read_excel(
            file_inflation_level, skiprows=11, index_col=0)['Annual']
    except FileNotFoundError as err:
        print("Download input data from https://data.bls.gov/timeseries/CUUR0000SA0. "
              "Select years from 1913--2018 and include annual averages, then download "
              ".xlsx file and save at revmpath+'Data/BLS/inflation_cpi_level.xlsx'")
        print(err)
        raise FileNotFoundError
    
    ### Return the ds 
    if (yearin is None) and (value is None):
        return dsinflation
    ### Or return the ratio
    elif (value is None):
        return dsinflation[yearout] / dsinflation[yearin]
    ### Or return the inflated value
    else:
        return value * dsinflation[yearout] / dsinflation[yearin]

def depreciation(year, schedule='macrs', period=5):
    """
    Function
    --------
    Returns nominal depreciation rate [%] given choice of 
        depreciation schedule.
        
    Inputs
    ------
    year: int year of operation                [year]
    schedule: str in ['macrs', 'sl', 'none']   (default 'macrs')
    period: int length of depreciation period  [years] (default 5)
    
    Returns
    -------
    float: Nominal depreciation rate [%]. If year is
        outside the schedule, returns 0.
        
    References
    ----------
    * https://www.irs.gov/forms-pubs/about-publication-946
    * https://www.irs.gov/pub/irs-pdf/p946.pdf
    """
    if year == 0:
        # raise Exception('year must be >= 1')
        return 0
    index = int(year) - 1
        
    if schedule == 'macrs':
        if period not in [3, 5, 7, 10, 15, 20]:
            raise Exception("Invalid period")
        rate = {
            3: [33.33, 44.45, 14.81, 7.41],
            5: [20.00, 32.00, 19.20, 11.52, 11.52, 
                 5.76],
            7: [14.29, 24.49, 17.49, 12.49, 8.93, 
                 8.92,  8.93,  4.46],
            10: [10.00, 18.00, 14.40, 11.52, 9.22, 
                  7.37,  6.55,  6.55,  6.56, 6.55, 
                  3.28],
            15: [5.00, 9.50, 8.55, 7.70, 6.93, 
                 6.23, 5.90, 5.90, 5.91, 5.90, 
                 5.91, 5.90, 5.91, 5.90, 5.91,
                 2.95],
            20: [3.750, 7.219, 6.677, 6.177, 5.713,
                 5.285, 4.888, 4.522, 4.462, 4.461,
                 4.462, 4.461, 4.462, 4.461, 4.462,
                 4.461, 4.462, 4.461, 4.462, 4.461,
                 2.231],
        }
        
    elif schedule in ['sl', 'straight', 'straightline']:
        rate = {period: np.ones(period) * 100 / period}
        
    elif schedule in [None, 'none', False, 'no']:
        rate = {period: [0]}
        
    else:
        raise Exception("Invalid schedule")
        
    try:
        return rate[period][index]
    except IndexError:
        return 0

def lcoe(lifetime, discount, capex, cf, fom, itc=0, degradation=0):
    """
    Inputs
    ------
    lifetime:    economic lifetime [years]
    discount:    discount rate [fraction]
    capex:       year-0 capital expenditures [$/kWac]
    cf:          capacity factor [fraction]
    fom:         fixed O&M costs [$/kWac-yr]
    itc:         investment tax credit [fraction]
    degradation: output degradation per year [fraction]

    Assumptions
    -----------
    * 8760 hours per year
    """
    ### Index
    years = np.arange(0,lifetime+0.1,1)
    ### Discount rate
    discounts = np.array([1/((1+discount)**year) for year in years])
    ### Degradation
    degrades = np.array([(1-degradation)**year for year in years])
    ### FOM costs
    costs = np.ones(len(years)) * fom
    ### Add capex cost to year 0 and remove FOM
    costs[0] = capex * (1 - itc)
    ### Discount costs
    costs_discounted = costs * discounts
    ### Energy generation, discounted and degraded
    energy_discounted = cf * 8760 * discounts * degrades
    ### Set first-year generation to zero
    energy_discounted[0] = 0
    ### Sum and return
    out = costs_discounted.sum() / energy_discounted.sum()
    return out

############
### NREL ATB

def confinfactor(costschedule=None, taxrate=28, interest_nominal=3.7,
    taxrate_federal=None, taxrate_state=None,
    deduct_state_taxrate=False):
    """
    Inputs
    ------
    costschedule: List with percent of construction costs spent in 
        each year. Must sum to 100. 
        Examples: PV is [100], natural gas is [80, 10, 10]
    
    Source
    ------
    NREL ATB 2018: https://atb.nrel.gov/, 
    https://data.nrel.gov/files/89/2018-ATB-data-interim-geo.xlsm
    """
    ### Calculate taxrate if taxrate_federal and taxrate_state
    if (taxrate_federal is not None) and (taxrate_state is not None):
        ### Note: in 2018, the ability to fully deduct state taxes from 
        ### federal income tax was repealed for individuals. 
        ### Not sure about corporations, but probably.
        ### So we simply add the federal and state tax rates.
        if deduct_state_taxrate is False:
            taxrate = taxrate_federal + taxrate_state
        elif deduct_state_taxrate is True:
            taxrate = (taxrate_federal*0.01 * (1 - taxrate_state*0.01) 
                       + taxrate_state*0.01) * 100
    
    ### Assume all construction costs are in first year if no schedule supplied
    if costschedule is None:
        costschedule = [100]
    if sum(costschedule) != 100:
        raise Exception(
            'costschedule elements must sum to 100 but sum to {}'.format(
                sum(costschedule)))
        
    ### Years of construction
    years = range(len(costschedule))
    
    ### Sum costs of debt
    yearlyinterest = [
        (
            costschedule[year]*0.01 
            * (1 + (1 - taxrate*0.01)
                   * ((1 + interest_nominal*0.01)**(year + 0.5) - 1)
              )
        )
        for year in years
    ]
    
    return sum(yearlyinterest)

def depreciation_present_value(
    wacc=7, inflationrate=2.5, schedule='macrs', period=5):
    """
    Notes
    -----
    * Inputs are in percent (so 10% is 10, not 0.1)
    
    Inputs
    ------
    wacc: REAL weighted average cost of capital
    
    Source
    ------
    NREL ATB 2018: https://atb.nrel.gov/, 
    https://data.nrel.gov/files/89/2018-ATB-data-interim-geo.xlsm
    """
    ### Years over which depreciation applies
    ### (note that MACRS has one extra year; we keep it for straight-line
    ### because it's zero after the end of the depreciation period anyway)
    years = range(1,period+2)
    
    ### Sum discounted values of depreciation
    out = sum([
        (
            depreciation(year, schedule, period) 
            / (((1 + wacc*0.01)*(1+inflationrate*0.01))**year)
        )
        for year in years
    ])*0.01 ### Convert to fraction instead of percent

    return out

def projfinfactor(
    wacc=7, taxrate=28, 
    inflationrate=2.5,
    schedule='macrs', period=5,
    taxrate_federal=None, taxrate_state=None,
    deduct_state_taxrate=False,):
    """
    Notes
    -----
    * Inputs are in percent (so 10% is 10, not 0.1)
    
    Inputs
    ------
    wacc: REAL weighted average cost of capital
    
    Source
    ------
    NREL ATB 2018: https://atb.nrel.gov/, 
    https://data.nrel.gov/files/89/2018-ATB-data-interim-geo.xlsm
    """
    ### Calculate taxrate if taxrate_federal and taxrate_state
    if (taxrate_federal is not None) and (taxrate_state is not None):
        ### Note: in 2018, the ability to fully deduct state taxes from 
        ### federal income tax was repealed for individuals. 
        ### Not sure about corporations, but probably.
        ### So we simply add the federal and state tax rates.
        if deduct_state_taxrate is False:
            taxrate = taxrate_federal + taxrate_state
        elif deduct_state_taxrate is True:
            taxrate = (taxrate_federal*0.01 * (1 - taxrate_state*0.01) 
                       + taxrate_state*0.01) * 100
    
    ### Get present value of depreciation
    dpv = depreciation_present_value(
        wacc=wacc, inflationrate=inflationrate, 
        schedule=schedule, period=period)
    
    ### Calculate factor according to NREL ATB
    out = (1 - taxrate*0.01 * dpv) / (1 - taxrate*0.01)
    
    return out

def lcoe_atb(
    overnightcost, cf, wacc, lifetime, fom=0, itc=0,
    taxrate=28, interest_nominal=3.7,
    taxrate_federal=None, taxrate_state=None,
    deduct_state_taxrate=False,
    inflationrate=2.5,
    schedule='macrs', period=5,
    costschedule=None,):
    """
    Notes
    -----
    * All inputs are in percents, not fractions

    Source
    ------
    NREL ATB 2018: https://atb.nrel.gov/, 
    https://data.nrel.gov/files/89/2018-ATB-data-interim-geo.xlsm
    """
    ### Calculate taxrate if taxrate_federal and taxrate_state
    if (taxrate_federal is not None) and (taxrate_state is not None):
        ### Note: in 2018, the ability to fully deduct state taxes from 
        ### federal income tax was repealed for individuals. 
        ### Not sure about corporations, but probably.
        ### So we simply add the federal and state tax rates.
        if deduct_state_taxrate is False:
            taxrate = taxrate_federal + taxrate_state
        elif deduct_state_taxrate is True:
            taxrate = (taxrate_federal*0.01 * (1 - taxrate_state*0.01) 
                       + taxrate_state*0.01) * 100
    
    out = (
        (
            crf(wacc=wacc, lifetime=lifetime)
            * projfinfactor(
                wacc=wacc, taxrate=taxrate, inflationrate=inflationrate, 
                schedule=schedule, period=period,
                taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
                deduct_state_taxrate=deduct_state_taxrate)
            * confinfactor(
                costschedule=costschedule, taxrate=taxrate, 
                interest_nominal=interest_nominal, 
                taxrate_federal=taxrate_federal, taxrate_state=taxrate_state, 
                deduct_state_taxrate=deduct_state_taxrate)
            * overnightcost
            * (1 - itc*0.01)
            + fom
        ) / (cf * 8760)
    )
    
    return out

#########
### Solar

def npv(
    revenue, carboncost, carbontons, cost_upfront,
    wacc, lifetime, degradationrate,
    cost_om, cost_om_units, 
    inflationrate=2.5,
    taxrate=None, taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Function
    --------
    Calculates net present value given yearly revenue and
        lots of other financial assumptions
    
    Inputs
    ------
    revenue: numeric or np.array            [$/kWac-yr]
    wacc: numeric                           [percent]
    lifetime: numeric                       [years]
    cost_om: numeric                        [$/kWac-yr]
                              OR            [percent of cost_upfront]
    cost_upfront: numeric                   [$/kWac]
    carboncost: numeric                     [$/ton]
    carbontons: numeric                     [tons/MWac-yr]
    cost_om_units: str in ['$', '%']      
    inflationrate: numeric                  [percent] (default = 2.5)
    taxrate: numeric                        [percent]
    taxrate_federal: numeric or None        [percent]
    taxrate_state: numeric or None          [percent]
    schedule: str in ['macrs', 'sl', 'none] {input to depreciation()}
                                                (default = 'macrs')
    period: int                             {input to depreciation()}
                                                (default = 5)
    itc: numeric                            [percent] (default = 0)
    
    Outputs
    -------
    npv: numeric or np.array                [$/kWac]
    
    Reference for formulation
    -------------------------
    * http://dx.doi.org/10.3390/en10081100 (Hogan2017)

    Assumptions
    -----------
    References:
    * (Jordan2013) Jordan, D.C.; Kurtz, S.R. "Photovoltaic Degradation
    Rates - an Analytical Review", Prog.Photovolt: Res. Appl. 2013, 21:12-29
    dx.doi.org/10.1002/pip.1182
    * (NREL2017) Fu, R.; Feldman, D.; Margolis, R.; Woodhouse, M.; Ardani, K.
    "U.S. Solar Photovoltaic System Cost Benchmark: Q1 2017"
    NREL/TP-6A20-68925
    * (Hogan2017) Salles, M.B.C.; Huang, J.; Aziz, M.J.; Hogan, W.W.
    "Potential Arbitrage Revenue of Energy Storage Systems in PJM"
    Energies 2017, 10, 1100; dx.doi.org/10.3390/en10081100
    
    Degradation rate:
    * Jordan2013
        * 0.5% median value across ~2000 modules and systems
    * NREL2017
        * 0.75% assumed for utility-scale 2017
    
    O&M cost:
    * NREL2017
        * $15.4/kW-yr for utility-scale fixed-tilt for 2017
        * $18.5/kW-yr for utility-scale 1-ax track for 2017
        * $15/kW-yr for commercial for 2017
                
    Lifetime:
    * NREL2017
        * 30 years for all systems
        
    Tax rate:
    * Hogan 2017
        * 38%
    * NREL2017
        * 35%
        
    Other assumptions for utility-scale 2017:
    * NREL2017
        * Pre-inverter derate (1 - loss_system) = 90.5%
        * Inverter efficiency (1 - loss_inverter) = 2.0%
        * Inflation rate = 2.5%
        * Equity discount rate (real) = 6.3%
        * Debt interest rate = 4.5%
        * Debt fraction = 40%
        * IRR target = 6.46%


    """
    ### Make year index
    years = np.arange(1, lifetime+1, 1)
    ### Calculate cost_om based on cost_om_units, if necessary
    if cost_om_units in ['percent', '%', '%/yr', '%peryr']:
        cost_om_val = cost_om*0.01 * cost_upfront
    elif cost_om_units in ['dollars', '$', '$/kWac-yr', '$perkWac-yr']:
        cost_om_val = cost_om
    else:
        raise Exception("Invalid cost_om_units; try '$' or '%'")
    ### Calculate taxrate if taxrate_federal and taxrate_state
    if (taxrate_federal is not None) and (taxrate_state is not None):
        ### Note: in 2018, the ability to fully deduct state taxes from 
        ### federal income tax was repealed for individuals. 
        ### Not sure about corporations, but probably.
        ### So we simply add the federal and state tax rates.
        taxrate = taxrate_federal + taxrate_state
        ### Old way, with state income tax deduction
        ## taxrate = (taxrate_federal * (1 - taxrate_state / 100) 
        ##            + taxrate_state)
        
    
    npv = (
        sum(
            [(  (  (  (  (revenue + (carboncost * carbontons / 1000)) 
                         * (1 - degradationrate*0.01)**year)
                      - cost_om_val)
                   * (1 - taxrate*0.01)
                   + (  (depreciation(year, schedule, period))*0.01
                        /  (1 + inflationrate*0.01)**year
                        * cost_upfront
                        * taxrate*0.01
                     ) 
                )
                / ((1 + wacc*0.01)**year)
             )
             for year in years]
        )
        - cost_upfront * (1 - itc*0.01)
    )
        
    return npv


def npv_upfrontcost(
    cost_upfront, revenue, 
    carboncost=50, carbontons=0, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Inputs
    ------
    Same as npv()
    
    Outputs
    -------
    npv: numeric or np.array       [$/kWac]
    """
    out = npv(
        revenue=revenue, carboncost=carboncost, carbontons=carbontons, 
        cost_upfront=cost_upfront, wacc=wacc, lifetime=lifetime, 
        degradationrate=degradationrate, 
        cost_om=cost_om, cost_om_units=cost_om_units, 
        inflationrate=inflationrate, taxrate=taxrate,
        taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
        schedule=schedule, period=period, itc=itc)
    return out

def npv_carbon(
    carboncost, revenue, 
    carbontons=0, cost_upfront=1400, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Inputs
    ------
    Same as npv()
    
    Outputs
    -------
    npv: numeric or np.array       [$/kWac]
    """
    out = npv(
        revenue=revenue, carboncost=carboncost, carbontons=carbontons, 
        cost_upfront=cost_upfront, wacc=wacc, lifetime=lifetime, 
        degradationrate=degradationrate, 
        cost_om=cost_om, cost_om_units=cost_om_units, 
        inflationrate=inflationrate, taxrate=taxrate,
        taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
        schedule=schedule, period=period, itc=itc)
    return out

def npv_wacc(
    wacc, revenue, 
    carboncost=50, carbontons=0, cost_upfront=1400, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Inputs
    ------
    Same as npv()
    
    Outputs
    -------
    npv: numeric or np.array       [$/kWac]
    """
    out = npv(
        revenue=revenue, carboncost=carboncost, carbontons=carbontons, 
        cost_upfront=cost_upfront, wacc=wacc, lifetime=lifetime, 
        degradationrate=degradationrate, 
        cost_om=cost_om, cost_om_units=cost_om_units, 
        inflationrate=inflationrate, taxrate=taxrate,
        taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
        schedule=schedule, period=period, itc=itc)
    return out

def npv_revenue(
    revenue, cost_upfront=1443,
    carboncost=50, carbontons=0, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0):
    """
    Inputs
    ------
    Same as npv()
    
    Outputs
    -------
    npv: numeric or np.array       [$/kWac]
    """
    out = npv(
        revenue=revenue, carboncost=carboncost, carbontons=carbontons, 
        cost_upfront=cost_upfront, wacc=wacc, lifetime=lifetime, 
        degradationrate=degradationrate, 
        cost_om=cost_om, cost_om_units=cost_om_units, 
        inflationrate=inflationrate, taxrate=taxrate,
        taxrate_federal=taxrate_federal, taxrate_state=taxrate_state,
        schedule=schedule, period=period, itc=itc)
    return out

def breakeven_upfrontcost(
    revenue, 
    carboncost=50, carbontons=0, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40,
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0, 
    maxiter=1000, xtol='default', 
    ab=(-1000, 100000), **kwargs):
    """
    Inputs
    ------
    revenue: numeric or np.array      [$/kWac-yr]
    carbontons:                       [tons/MWac-yr]
    cost_upfront: numeric or np.array [$/kWac-yr]
    wacc: numeric                     [percent]
    lifetime: numeric                 [years]
    degradationrate: numeric          [percent/yr]
    cost_om: numeric                  [$/kWac-yr]
                                 OR   [(percent of cost_upfront) / yr]
    cost_om_units: str in ['dollars', 'percent']
    taxrate: numeric                  [percent]
    
    Outputs
    -------
    carboncost: numeric               [$/ton]
    """
    ### Brent method
    result = scipy.optimize.brentq(
        npv_upfrontcost, 
        a=ab[0],
        b=ab[1],
        args=(revenue, carboncost, carbontons, wacc,
              lifetime, degradationrate, 
              cost_om, cost_om_units, 
              inflationrate, taxrate, 
              taxrate_federal, taxrate_state,
              schedule, period, itc),
        maxiter=maxiter,
    )
    return result

def breakeven_carboncost(
    revenue, 
    carbontons=0, cost_upfront=1400, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0, 
    maxiter=1000, xtol='default', 
    ab=(-1000, 100000), **kwargs):
    """
    Inputs
    ------
    revenue: numeric or np.array      [$/kWac-yr]
    carbontons:                       [tons/MWac-yr]
    cost_upfront: numeric or np.array [$/kWac-yr]
    wacc: numeric                     [percent]
    lifetime: numeric                 [years]
    degradationrate: numeric          [percent/yr]
    cost_om: numeric                  [$/kWac-yr]
                                 OR   [(percent of cost_upfront) / yr]
    cost_om_units: str in ['dollars', 'percent']
    taxrate: numeric                  [percent]
    
    Outputs
    -------
    carboncost: numeric               [$/ton]
    """
    ### Brent method
    result = scipy.optimize.brentq(
        npv_carbon, 
        a=ab[0],
        b=ab[1],
        args=(revenue, carbontons, cost_upfront, wacc,
              lifetime, degradationrate, 
              cost_om, cost_om_units, 
              inflationrate, taxrate, 
              taxrate_federal, taxrate_state,
              schedule, period, itc),
        maxiter=maxiter,
    )
    return result

def breakeven_wacc(
    revenue, 
    carboncost=50, carbontons=0, cost_upfront=1400, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40, 
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5,
    maxiter=1000, xtol='default', 
    ab=(-80, 10000), **kwargs):
    """
    Inputs
    ------
    revenue: numeric or np.array      [$/kWac-yr]
    carbontons:                       [tons/MWac-yr]
    cost_upfront: numeric or np.array [$/kWac-yr]
    wacc: numeric                     [percent]
    lifetime: numeric                 [years]
    degradationrate: numeric          [percent/yr]
    cost_om: numeric                  [$/kWac-yr]
                                 OR   [(percent of cost_upfront) / yr]
    cost_om_units: str in ['dollars', 'percent']
    taxrate: numeric                  [percent]
    
    Outputs
    -------
    carboncost: numeric               [$/ton]
    """
    ### Brent method
    result = scipy.optimize.brentq(
        npv_wacc, 
        a=ab[0],
        b=ab[1],
        args=(revenue, carboncost, carbontons, cost_upfront,
              lifetime, degradationrate, 
              cost_om, cost_om_units, 
              inflationrate, taxrate, 
              taxrate_federal, taxrate_state,
              schedule, period, itc),
        maxiter=maxiter,
    )
    return result

def breakeven_revenue(
    cost_upfront, 
    carboncost=50, carbontons=0, wacc=7, lifetime=30, 
    degradationrate=0.5, cost_om=15, cost_om_units='$',
    inflationrate=2.5, taxrate=40,
    taxrate_federal=None, taxrate_state=None,
    schedule='macrs', period=5, itc=0, 
    maxiter=1000, xtol='default', 
    ab=(-1000, 100000), **kwargs):
    """
    Inputs
    ------
    revenue: numeric or np.array      [$/kWac-yr]
    carbontons:                       [tons/MWac-yr]
    cost_upfront: numeric or np.array [$/kWac-yr]
    wacc: numeric                     [percent]
    lifetime: numeric                 [years]
    degradationrate: numeric          [percent/yr]
    cost_om: numeric                  [$/kWac-yr]
                                 OR   [(percent of cost_upfront) / yr]
    cost_om_units: str in ['dollars', 'percent']
    taxrate: numeric                  [percent]
    
    Outputs
    -------
    carboncost: numeric               [$/ton]
    """
    ### Brent method
    result = scipy.optimize.brentq(
        npv_revenue, 
        a=ab[0],
        b=ab[1],
        args=(cost_upfront, carboncost, carbontons, wacc,
              lifetime, degradationrate, 
              cost_om, cost_om_units, 
              inflationrate, taxrate, 
              taxrate_federal, taxrate_state,
              schedule, period, itc),
        maxiter=maxiter,
    )
    return result

################################
### ORIENTATION OPTIMIZATION ###

def pv_optimize_orientation_objective(
    axis_tilt_and_azimuth,
    objective,
    dfsun, info, tznode, elevation,
    solpos, dni_et, airmass,
    systemtype, 
    yearsun, resolutionsun, 
    dflmp=None, yearlmp=None, resolutionlmp=None, tzlmp=None,
    pricecutoff=None,
    max_angle=60, backtrack=True, gcr=1./3., 
    dcac=1.3, 
    loss_system=0.14, loss_inverter=0.04, 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    temp_model='open_rack_cell_polymerback',
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990',
    axis_tilt_constant=None, axis_azimuth_constant=None,
    clip=True,
    ):
    """
    """
    ###### Repackage input variables as necessary
    ### If axis_tilt_and_azimuth is numeric and length 1, require that
    ### only axis_tilt_constant or axis_azimuth_constant be specifed.
    ### axis_tilt_and_azimuth will then be taken as whichever of 
    ### axis_tilt_constant or axis_azimuth_constant is not specified.
    
    ### Make sure at least one of axis_tilt_constant, axis_azimuth_constant is None
    if (axis_tilt_constant is not None) and (axis_azimuth_constant is not None):
        print('axis_tilt_and_azimuth: {}, {}'.format(
            axis_tilt_and_azimuth, type(axis_tilt_and_azimuth)))
        print('axis_tilt_constant: {}, {}'.format(
            axis_tilt_constant, type(axis_tilt_constant)))
        print('axis_azimuth_constant: {}, {}'.format(
            axis_azimuth_constant, type(axis_azimuth_constant)))
        raise Exception("At least one of (axis_tilt_constant, "
                        "axis_azimuth_constant) must be None.")
    ### Azimuth-only optimization case
    elif axis_tilt_constant is not None:
        axis_tilt_and_azimuth = (axis_tilt_constant, float(axis_tilt_and_azimuth))
    ### Tilt-only optimization case
    elif axis_azimuth_constant is not None:
        axis_tilt_and_azimuth = (float(axis_tilt_and_azimuth), axis_azimuth_constant)
    ### Full optimization orientation
    elif (axis_azimuth_constant is None) and (axis_tilt_constant is None):
        pass
    
    ### Continue as usual
    output_ac = pv_system_sim_fast(
        axis_tilt_and_azimuth=axis_tilt_and_azimuth,
        dfsun=dfsun, info=info, tznode=tznode, elevation=elevation,
        solpos=solpos, dni_et=dni_et, airmass=airmass,
        year=yearsun, systemtype=systemtype, 
        max_angle=max_angle, backtrack=backtrack, gcr=gcr,
        dcac=dcac,
        loss_system=loss_system, loss_inverter=loss_inverter,
        n_ar=n_ar, n_glass=n_glass,
        tempcoeff=tempcoeff, 
        temp_model=temp_model,
        albedo=albedo, diffuse_model=diffuse_model, 
        et_method=et_method, model_perez=model_perez,
        clip=clip)

    if objective.lower() in ['cf', 'capacityfactor']:
        capacityfactor = (
            0.001 * output_ac.sum() / len(output_ac))
        return -capacityfactor

    elif objective.lower() in ['rev', 'revenue']:
        ### Drop leap days if yearlmp andyearsun have different year lengths
        if pvvm.toolbox.yearhours(yearlmp) == 8784:
            if (yearsun == 'tmy') or pvvm.toolbox.yearhours(yearsun) == 8760:
                ## Drop lmp leapyear
                dflmp = dropleap(dflmp, yearlmp, resolutionlmp)
        elif pvvm.toolbox.yearhours(yearlmp) == 8760 and (pvvm.toolbox.yearhours(yearsun) == 8784):
            ## Drop sun leapyear
            output_ac = dropleap(output_ac, yearsun, resolutionsun)

        ### Reset indices if yearsun != yearlmp
        if yearsun != yearlmp:
            output_ac.index = pd.date_range(
                '2001-01-01', 
                periods=(8760 * 60 / resolutionsun), 
                freq='{}min'.format(resolutionsun), 
                tz=tznode
            ).tz_convert(tzlmp)

            dflmp.index = pd.date_range(
                '2001-01-01', 
                periods=(8760 * 60 / resolutionlmp),
                freq='{}min'.format(resolutionlmp), 
                tz=tzlmp
            )

        ### upsample solar data if resolutionlmp == 5
        if resolutionlmp == 5:
            ### Original version - no longer works given recent pandas update
            # output_ac.loc[output_ac.index.max() + 1] = output_ac.iloc[-1]
            ### New version
            output_ac = output_ac.append(
                pd.Series(data=output_ac.iloc[-1],
                          index=[output_ac.index[-1] + pd.Timedelta(resolutionsun, 'm')]))
            ### Continue
            output_ac = output_ac.resample('5T').interpolate(method='time')
            output_ac.drop(output_ac.index[-1], axis=0, inplace=True)

        ### upsample LMP data if resolutionsun == 30
        if (resolutionlmp == 60) and (resolutionsun == 30):
            dflmp = dflmp.resample('30T').ffill()
            dflmp.loc[dflmp.index.max() + 1] = dflmp.iloc[-1]

        if len(output_ac) != len(dflmp):
            print('Something probably went wrong with leap years')
            print('len(output_ac.index) = {}'.format(len(output_ac.index)))
            print('len(dflmp.index) = {}'.format(len(dflmp.index)))
            raise Exception('Mismatched lengths for LMP and output_ac files')

        ### put dflmp into same timezone as output_ac
        ### (note that this will drop point(s) from dflmp)
        if tznode != tzlmp:
            dflmp = (
                pd.DataFrame(dflmp.tz_convert(tznode))
                .merge(pd.DataFrame(index=output_ac.index), 
                       left_index=True, right_index=True)
            )['lmp']

        ### Determine final resolution
        resolution = min(resolutionlmp, resolutionsun)
        

        ### Calculate revenue
        if pricecutoff is None:
            revenue_timestep = 0.001 * output_ac * dflmp * resolution / 60 
        ### DO: Would be faster to do this outside of the function
        else:
            dispatch = dflmp.map(lambda x: x > pricecutoff)
            output_dispatched = (output_ac * dispatch)
            revenue_timestep = (
                0.001 * output_dispatched * dflmp * resolution / 60)

        revenue_yearly = revenue_timestep.sum() / 1000 # $/kWac-yr
        return -revenue_yearly

def solarvalue_compute(
    nsrdbpath, nsrdbfile, yearsun, nsrdbtype, resolutionsun,
    lmpfilepath, lmpfile, yearlmp, tzlmp, resolutionlmp, 
    pricecutoff=None,
    systemtype='track', axis_tilt='latitude', axis_azimuth=0,
    dcac=1.3, loss_system=0.14, loss_inverter=0.04, 
    max_angle=60, backtrack=True, gcr=1./3., 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    temp_model='open_rack_cell_polymerback',
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990',
    clip=True,
    ):
    """
    """
    ### Convenience variables
    hours = min(pvvm.toolbox.yearhours(yearlmp), pvvm.toolbox.yearhours(yearsun))

    ### Calculate hourly PV output
    output_ac, tznode = pv_system_sim(
        # nsrdbfile=str(dfin.loc[i, latlonindex]), 
        nsrdbfile=nsrdbfile,
        year=yearsun, systemtype=systemtype,
        resolution=resolutionsun, dcac=dcac, 
        axis_tilt=axis_tilt, axis_azimuth=axis_azimuth, 
        max_angle=max_angle, backtrack=backtrack, gcr=gcr, 
        loss_system=loss_system, loss_inverter=loss_inverter,
        n_ar=n_ar, n_glass=n_glass,
        tempcoeff=tempcoeff, temp_model=temp_model,
        albedo=albedo, nsrdbpathin=nsrdbpath, nsrdbtype=nsrdbtype, 
        et_method=et_method, diffuse_model=diffuse_model, 
        model_perez=model_perez, 
        output_ac_only=False, output_ac_tz=True, clip=clip)

    ### Get LMP data
    dflmp = pvvm.io.getLMPfile(lmpfilepath, lmpfile, tzlmp)['lmp']

    ### Drop leap days if yearlmp andyearsun have different year lengths
    if pvvm.toolbox.yearhours(yearlmp) == 8784:
        if (yearsun == 'tmy') or pvvm.toolbox.yearhours(yearsun) == 8760:
            ## Drop lmp leapyear
            dflmp = dropleap(dflmp, yearlmp, resolutionlmp)
    elif pvvm.toolbox.yearhours(yearlmp) == 8760 and (pvvm.toolbox.yearhours(yearsun) == 8784):
        ## Drop sun leapyear
        output_ac = dropleap(output_ac, yearsun, resolutionsun)

    ### Reset indices if yearsun != yearlmp
    if yearsun != yearlmp:
        output_ac.index = pd.date_range(
            '2001-01-01', 
            periods=(8760 * 60 / resolutionsun), 
            freq='{}min'.format(resolutionsun), 
            tz=tznode
        ).tz_convert(tzlmp)

        dflmp.index = pd.date_range(
            '2001-01-01', 
            periods=(8760 * 60 / resolutionlmp),
            freq='{}min'.format(resolutionlmp), 
            tz=tzlmp
        )

    ### upsample solar data if resolutionlmp == 5
    if resolutionlmp == 5:
        ### Original version - no longer works given recent pandas update
        # output_ac.loc[output_ac.index.max() + 1] = output_ac.iloc[-1]
        ### New version
        output_ac = output_ac.append(
            pd.Series(data=output_ac.iloc[-1],
                      index=[output_ac.index[-1] + pd.Timedelta(resolutionsun, 'm')]))
        ### Continue
        output_ac = output_ac.resample('5T').interpolate(method='time')
        output_ac.drop(output_ac.index[-1], axis=0, inplace=True)

    ### upsample LMP data if resolutionsun == 30
    if (resolutionlmp == 60) and (resolutionsun == 30):
        dflmp = dflmp.resample('30T').ffill()
        dflmp.loc[dflmp.index.max() + 1] = dflmp.iloc[-1]

    if len(output_ac) != len(dflmp):
        print('Something probably went wrong with leap years')
        print('NSRDBfile = {}'.format(dfin.loc[i, latlonindex]))
        print('lmpfile = {}'.format(lmpfile))
        print('len(output_ac.index) = {}'.format(len(output_ac.index)))
        print('len(dflmp.index) = {}'.format(len(dflmp.index)))
        raise Exception('Mismatched lengths for LMP and NSRDB files')

    ### put dflmp into same timezone as output_ac
    ### (note that this will drop point(s) from dflmp)
    if tznode != tzlmp:
        dflmp = (
            pd.DataFrame(dflmp.tz_convert(tznode))
            .merge(pd.DataFrame(index=output_ac.index), 
                   left_index=True, right_index=True)
        )['lmp']

    ### determine final resolution
    resolution = min(resolutionlmp, resolutionsun)

    ##### perform analysis
    ### Yearly
    if pricecutoff is not None:
        dispatch = dflmp.map(lambda x: x > pricecutoff)
        output_ac = (output_ac * dispatch)

    revenue_timestep = 0.001 * output_ac * dflmp * resolution / 60 
    revenue_yearly = revenue_timestep.sum() / 1000 # $/kWac-yr
    generation_yearly = 0.001 * output_ac.sum() * resolution / 60      
    capacity_factor = generation_yearly / hours
    value_average = revenue_yearly / generation_yearly * 1000
    price_average = dflmp.mean()
    value_factor = value_average / price_average

    return (capacity_factor, price_average, 
        value_average, revenue_yearly, value_factor)

def pv_optimize_orientation(
    objective,
    dfsun, info, tznode, elevation,
    solpos, dni_et, airmass,
    systemtype, 
    yearsun, resolutionsun, 
    dflmp=None, yearlmp=None, resolutionlmp=None, tzlmp=None,
    pricecutoff=None,
    max_angle=60, backtrack=True, gcr=1./3., 
    dcac=1.3, 
    loss_system=0.14, loss_inverter=0.04, 
    n_ar=1.3, n_glass=1.526, 
    tempcoeff=-0.004, 
    temp_model='open_rack_cell_polymerback',
    albedo=0.2, diffuse_model='reindl', 
    et_method='nrel', model_perez='allsitescomposite1990',
    ranges='default', full_output=False,
    optimize='both', 
    axis_tilt_constant=None, axis_azimuth_constant=None,
    clip=True,
    ):
    """
    """
    ###### Check inputs, adapt to different choices of optimization
    if optimize in ['both', 'default', None, 'orient', 'orientation']:
        assert (axis_tilt_constant is None) and (axis_azimuth_constant is None)
        if ranges == 'default':
            ranges = (slice(0, 100, 10), slice(90, 280, 10))
        else:
            assert len(ranges) == 2
            assert (isinstance(ranges[0], slice) 
                    and (isinstance(ranges[1], slice)))
    elif optimize in ['azimuth', 'axis_azimuth_constant']:
        assert axis_azimuth_constant is None
        if ranges == 'default':
            ranges = (slice(90, 271, 1),)
        else:
            assert isinstance(ranges, tuple)
            assert len(ranges) == 1
    elif optimize in ['tilt', 'axis_tilt_constant']:
        assert axis_tilt_constant is None
        if ranges == 'default':
            ranges = (slice(0, 91, 1),)
        else:
            assert isinstance(ranges, tuple)
            assert len(ranges) == 1
    else:
        raise Exception("optimize must be in 'both', 'azimuth', 'tilt'")

    params = (
        objective, dfsun, info, tznode, elevation, solpos, dni_et, 
        airmass, systemtype, yearsun, resolutionsun, 
        dflmp, yearlmp, resolutionlmp, tzlmp,
        pricecutoff,
        max_angle, backtrack, gcr, dcac, loss_system, loss_inverter, 
        n_ar, n_glass, tempcoeff, temp_model, albedo, 
        diffuse_model, et_method, model_perez, 
        axis_tilt_constant, axis_azimuth_constant,
        clip)

    results = scipy.optimize.brute(
        pv_optimize_orientation_objective,
        ranges=ranges,
        args=params,
        full_output=True,
        disp=False,
        finish=scipy.optimize.fmin)

    ### Unpack and return results
    opt_orient = results[0]
    opt_objective = results[1]
    opt_input_grid = results[2]
    opt_output_grid = results[3]
    
    if full_output:
        return (opt_orient, -opt_objective, 
                opt_input_grid, opt_output_grid)
    else:
        return opt_orient, -opt_objective

def solarvalue_orientopt(yearlmp, yearsun, 
    isos=['caiso', 'ercot', 'miso', 'pjm', 'nyiso', 'isone'],
    market='da', submarket=None, timezonelmp='default', 
    set_resolutionlmp='default',
    ranges='default',
    monthly=False, daily=False,
    pricecutoff=0,
    systemtype='fixed', dcac=1.3, 
    axis_tilt_constant=None, axis_azimuth_constant=None, optimize='both', 
    max_angle=60, backtrack=True, gcr=1./3., 
    loss_system=0.14, loss_inverter=0.04,
    n_ar=1.3, n_glass=1.526, tempcoeff=-0.004,
    temp_model='open_rack_cell_polymerback',
    albedo=0.2, diffuse_model='reindl',
    et_method='nrel', model_perez='allsitescomposite1990',
    savemod='', write='default', compress=False, 
    nsrdbpathin='default', nsrdbtype='.gz',
    lmppath='default', lmptype='.gz',
    outpath=None, savesafe=True,
    runbegin=0, runend=1000000,
    clip=True):
    """
    """
    ### Normalize inputs
    ## Put isos in lower case
    isos = [iso.lower() for iso in isos]
    ## Cut ERCOT if year < 2011
    if (yearlmp <= 2010) and ('ercot' in isos):
        isos.remove('ercot')
    ## Cut ISONE if year < 2011 and market == 'rt'
    if (yearlmp <= 2010) and (market == 'rt') and ('isone' in isos):
        isos.remove('isone')
    ## Set solar resolution
    if yearsun == 'tmy':
        resolutionsun = 60
    elif type(yearsun) == int:
        resolutionsun = 30
    else:
        print(resolution)
        raise Exception("yearsun must by 'tmy' or int")
    ## Check other inputs
    if write not in ['default', False] and write.find('.') == -1:
        raise Exception('write (i.e. savename) needs a "."')
    for iso in isos:
        if iso not in ['caiso', 'ercot', 'miso', 
            'pjm', 'nyiso', 'isone']:
            raise Exception("Invalid iso")
    if outpath is None:
        outpath = revmpath + 'out/'
    os.makedirs(outpath, exist_ok=True)
    ## Set defaults
    if systemtype == 'fixed':
        default_axis_tilt, default_axis_azimuth = 'latitude', 180
    elif systemtype == 'track':
        default_axis_tilt, default_axis_azimuth = 0, 180
    ###### Set default ranges based on optimize value
    if optimize in ['both', 'default', None, 'orient', 'orientation']:
        assert (axis_tilt_constant is None) and (axis_azimuth_constant is None)
        if ranges == 'default':
            ranges = (slice(0, 100, 10), slice(90, 280, 10))
        else:
            assert len(ranges) == 2
            assert (isinstance(ranges[0], slice) 
                    and (isinstance(ranges[1], slice)))
    elif optimize in ['azimuth', 'axis_azimuth_constant']:
        assert axis_azimuth_constant is None
        if ranges == 'default':
            ranges = (slice(90, 271, 1),)
        else:
            assert isinstance(ranges, tuple)
            assert len(ranges) == 1
    elif optimize in ['tilt', 'axis_tilt_constant']:
        assert axis_tilt_constant is None
        if ranges == 'default':
            ranges = (slice(0, 91, 1),)
        else:
            assert isinstance(ranges, tuple)
            assert len(ranges) == 1
    else:
        raise Exception("optimize must be in 'both', 'azimuth', 'tilt'")

    ### savenames
    if write:
        if compress:
            compression, saveend = ('gzip', '.gz')
        else:
            compression, saveend = (None, '.csv')

        abbrevs = {
            'caiso': 'C',
            'ercot': 'E',
            'miso': 'M',
            'pjm': 'P',
            'nyiso': 'N',
            'isone': 'I'}
        abbrev = ''.join([abbrevs[i.lower()] for i in isos])

        if write == 'default':
            if pricecutoff is None:
                pricecutoffwrite = 'no'
            else:
                pricecutoffwrite = '{:.0f}'.format(pricecutoff)
            ### Add modifiers for optimization routine
            if optimize not in ['both', 'orient', 'default', 'orientation', None]:
                savemodextra = {
                    'tilt': 'tiltopt_{}az_'.format(axis_azimuth_constant),
                    'azimuth': 'azopt_{}tilt_'.format(int(axis_tilt_constant)),
                }
                savemodextra['axis_tilt'] = savemodextra['tilt']
                savemodextra['axis_azimuth'] = savemodextra['azimuth']
                savemod = savemodextra[optimize] + savemod
            savename = '{}PVvalueOptV4-{}-{}-{}lmp-{}sun-{}-{}ILR-{}cutoff-{}_{}-{}{}'.format(
                outpath, abbrev, market, yearlmp, yearsun, 
                systemtype, 
                int((1-loss_system)*(1-loss_inverter)*dcac*100), 
                pricecutoffwrite, runbegin, runend,
                savemod, saveend)
        else:
            savename = write

        describename = os.path.splitext(savename)[0] + '-describe.txt'
        if savesafe == True:
            savename = pvvm.toolbox.safesave(savename)
            describename = pvvm.toolbox.safesave(describename)
        print(savename)
        sys.stdout.flush()
        ### Make sure the folder exists
        if not os.path.isdir(os.path.dirname(savename)):
            raise Exception('{} does not exist'.format(os.path.dirname(savename)))

    ### Convenience variables
    hours = min(pvvm.toolbox.yearhours(yearlmp), pvvm.toolbox.yearhours(yearsun))

    ### Set up results containers
    results = []
    results_to_concat = {}

    ### Loop over ISOs
    for iso in isos:
        
        print(iso.upper())
        ### Timezone in which LMPs are reported
        if timezonelmp == 'default':
            tzlmp = pvvm.toolbox.tz_iso[iso]
        else:
            tzlmp = 'Etc/GMT{0:+}'.format(-timezonelmp)

        ### Set default resolution
        if set_resolutionlmp == 'default':
            if market == 'da':
                resolutionlmp = 60
            elif (market == 'rt') and (iso in ['ercot', 'nyiso', 'caiso']):
                resolutionlmp = 5
            elif (market == 'rt') and (iso in ['isone', 'miso', 'pjm']):
                resolutionlmp = 60

        ### Glue together different ISO labeling formalisms
        (nsrdbindex, lmpindex, pnodeid,
            latlonindex, pnodename) = pvvm.io.glue_iso_columns(iso)

        ### Get nodes with lat/lon info and full lmp data for yearlmp
        dfin = pvvm.io.get_iso_nodes(iso, market, yearlmp, merge=True)

        ### Set NSRDB filepath
        if nsrdbpathin == 'default':
            nsrdbpath = '{}{}/in/NSRDB/{}/{}min/'.format(
                revmpath, iso.upper(), yearsun, resolutionsun)
        else:
            nsrdbpath = nsrdbpathin

        ### Set LMP filepath
        if lmppath == 'default':
            ### DO: Modify this when you want to allow monthly value
            if (market == 'rt') and (iso == 'ercot'):
                lmpfilepath = '{}{}/io/lmp-nodal/{}/'.format(
                    revmpath, iso.upper(), 'rt-year')
            elif (market == 'rt') and (iso == 'caiso'):
                lmpfilepath = '{}{}/io/lmp-nodal/{}/'.format(
                    revmpath, iso.upper(), 'rt/rtm-yearly')
            else:
                lmpfilepath = '{}{}/io/lmp-nodal/{}/'.format(
                    revmpath, iso.upper(), market)
        else:
            lmpfilepath = lmppath

        ### Make NSRDB file list
        nsrdbfiles = list(dfin[nsrdbindex])
        for i in range(len(nsrdbfiles)):
            nsrdbfiles[i] = str(nsrdbfiles[i]) + '-{}{}'.format(yearsun, nsrdbtype)

        ### Make lmp file list
        lmpfiles = list(dfin[pnodeid])
        for i in range(len(lmpfiles)):
            lmpfiles[i] = str(lmpfiles[i]) + '-{}{}'.format(yearlmp, lmptype)

        ### Make isonode list
        isonodes = list(iso.upper() + ':' + dfin[pnodeid].astype(str))

        ### Determine runlength
        runbegin_iso = min(runbegin, len(dfin))
        runend_iso = min(runend, len(dfin))

        ##################################################################
        ##################### CALCULATE VALUE ############################
        ##################################################################
        
        ### Loop over nodes
        for i in trange(runbegin_iso, runend_iso):

            ### Calculate value with standard assumptions
            (capacity_factor, price_average, value_average, 
             revenue_yearly, value_factor) = solarvalue_compute(
                nsrdbpath=nsrdbpath, nsrdbfile=nsrdbfiles[i],
                yearsun=yearsun, resolutionsun=resolutionsun,
                lmpfilepath=lmpfilepath, lmpfile=lmpfiles[i],
                yearlmp=yearlmp, tzlmp=tzlmp, resolutionlmp=resolutionlmp,
                pricecutoff=pricecutoff,
                systemtype=systemtype, 
                axis_tilt=default_axis_tilt, axis_azimuth=default_axis_azimuth,
                dcac=dcac, 
                loss_system=loss_system, loss_inverter=loss_inverter,
                max_angle=max_angle, backtrack=backtrack, gcr=gcr, 
                n_ar=n_ar, n_glass=n_glass,
                tempcoeff=tempcoeff, temp_model=temp_model,
                albedo=albedo, diffuse_model=diffuse_model,
                et_method=et_method, model_perez=model_perez,
                nsrdbtype=nsrdbtype, clip=clip)

            ##### Prepare variables for optimization
            ### Load NSRDB file
            dfsun, info, tznode, elevation = pvvm.io.getNSRDBfile(
                filepath=nsrdbpath, 
                filename=nsrdbfiles[i], 
                year=yearsun, resolution=resolutionsun, 
                forcemidnight=False)

            ### Pull latitude and longitude from NSRDB file
            latitude = float(info['Latitude'])
            longitude = float(info['Longitude'])

            ### Determine solar position
            solpos = pvlib.solarposition.get_solarposition(
                dfsun.index, latitude, longitude)

            ### Set extra parameters for diffuse sky models
            if diffuse_model in ['haydavies', 'reindl', 'perez']:
                dni_et = pvlib.irradiance.extraradiation(
                    dfsun.index, method=et_method, epoch_year=yearsun)
                airmass = pvlib.atmosphere.relativeairmass(
                    solpos['apparent_zenith'])
            else:
                dni_et = None
                airmass = None

            ### Get LMP data
            dflmp = pvvm.io.getLMPfile(lmpfilepath, lmpfiles[i], tzlmp)['lmp']
            #####

            ### Optimize orientation for capacity factor
            optcf_orient, optcf_cf = pv_optimize_orientation(
                objective='cf', 
                dfsun=dfsun, info=info, tznode=tznode, elevation=elevation,
                solpos=solpos, dni_et=dni_et, airmass=airmass, 
                systemtype=systemtype, yearsun=yearsun, resolutionsun=resolutionsun,
                dflmp=dflmp, yearlmp=yearlmp, resolutionlmp=resolutionlmp, tzlmp=tzlmp,
                pricecutoff=pricecutoff,
                max_angle=max_angle, backtrack=backtrack, gcr=gcr, dcac=dcac,
                loss_system=loss_system, loss_inverter=loss_inverter,
                n_ar=n_ar, n_glass=n_glass, tempcoeff=tempcoeff,
                temp_model=temp_model, albedo=albedo, diffuse_model=diffuse_model,
                et_method=et_method, model_perez=model_perez,
                ranges=ranges, full_output=False,
                optimize=optimize, 
                axis_tilt_constant=axis_tilt_constant, 
                axis_azimuth_constant=axis_azimuth_constant, clip=clip,
            )

            ### Optimize orientation for revenue
            optrev_orient, optrev_rev = pv_optimize_orientation(
                objective='revenue', 
                dfsun=dfsun, info=info, tznode=tznode, elevation=elevation,
                solpos=solpos, dni_et=dni_et, airmass=airmass, 
                systemtype=systemtype, yearsun=yearsun, resolutionsun=resolutionsun,
                dflmp=dflmp, yearlmp=yearlmp, resolutionlmp=resolutionlmp, tzlmp=tzlmp,
                pricecutoff=pricecutoff,
                max_angle=max_angle, backtrack=backtrack, gcr=gcr, dcac=dcac,
                loss_system=loss_system, loss_inverter=loss_inverter,
                n_ar=n_ar, n_glass=n_glass, tempcoeff=tempcoeff,
                temp_model=temp_model, albedo=albedo, diffuse_model=diffuse_model,
                et_method=et_method, model_perez=model_perez,
                ranges=ranges, full_output=False,
                optimize=optimize, 
                axis_tilt_constant=axis_tilt_constant, 
                axis_azimuth_constant=axis_azimuth_constant, clip=clip,
            )

            ###### Calculate cross terms
            ### Pick orientations to use for cross terms
            if optimize in ['both', 'default', None, 'orient', 'orientation']:
                optcf_axis_tilt,  optcf_axis_azimuth  = optcf_orient[0],  optcf_orient[1]
                optrev_axis_tilt, optrev_axis_azimuth = optrev_orient[0], optrev_orient[1]

                optcf_rev_orientinput = (optcf_axis_tilt,  optcf_axis_azimuth)
                optrev_cf_orientinput = (optrev_axis_tilt, optrev_axis_azimuth)
                
                # optcf_rev_orientinput = (optcf_orient[0],  optcf_orient[1])
                # optrev_cf_orientinput = (optrev_orient[0], optrev_orient[1])

            elif optimize in ['azimuth', 'axis_azimuth']:
                optcf_axis_tilt,  optcf_axis_azimuth  = np.nan, optcf_orient[0]
                optrev_axis_tilt, optrev_axis_azimuth = np.nan, optrev_orient[0]

                # optcf_rev_orientinput = axis_tilt_constant
                # optrev_cf_orientinput = axis_tilt_constant
                
                optcf_rev_orientinput = (axis_tilt_constant, optcf_axis_azimuth)
                optrev_cf_orientinput = (axis_tilt_constant, optrev_axis_azimuth)

            elif optimize in ['tilt', 'axis_tilt']:
                optcf_axis_tilt,  optcf_axis_azimuth  = optcf_orient[0],  np.nan
                optrev_axis_tilt, optrev_axis_azimuth = optrev_orient[0], np.nan

                # optcf_rev_orientinput = axis_azimuth_constant
                # optrev_cf_orientinput = axis_azimuth_constant
                
                optcf_rev_orientinput = (optcf_axis_tilt,  axis_azimuth_constant)
                optrev_cf_orientinput = (optrev_axis_tilt, axis_azimuth_constant)

            ### Calculate
            optcf_rev = -pv_optimize_orientation_objective(
                axis_tilt_and_azimuth=optcf_rev_orientinput,
                objective='revenue',
                dfsun=dfsun, info=info, tznode=tznode, elevation=elevation,
                solpos=solpos, dni_et=dni_et, airmass=airmass, 
                systemtype=systemtype, yearsun=yearsun, resolutionsun=resolutionsun,
                dflmp=dflmp, yearlmp=yearlmp, resolutionlmp=resolutionlmp, tzlmp=tzlmp,
                pricecutoff=pricecutoff,
                max_angle=max_angle, backtrack=backtrack, gcr=gcr, dcac=dcac,
                loss_system=loss_system, loss_inverter=loss_inverter,
                n_ar=n_ar, n_glass=n_glass, tempcoeff=tempcoeff,
                temp_model=temp_model, albedo=albedo, diffuse_model=diffuse_model,
                et_method=et_method, model_perez=model_perez,
                axis_tilt_constant=None, axis_azimuth_constant=None, clip=clip,
            )

            optrev_cf = -pv_optimize_orientation_objective(
                axis_tilt_and_azimuth=optrev_cf_orientinput,
                objective='cf',
                dfsun=dfsun, info=info, tznode=tznode, elevation=elevation,
                solpos=solpos, dni_et=dni_et, airmass=airmass, 
                systemtype=systemtype, yearsun=yearsun, resolutionsun=resolutionsun,
                dflmp=dflmp, yearlmp=yearlmp, resolutionlmp=resolutionlmp, tzlmp=tzlmp,
                pricecutoff=pricecutoff,
                max_angle=max_angle, backtrack=backtrack, gcr=gcr, dcac=dcac,
                loss_system=loss_system, loss_inverter=loss_inverter,
                n_ar=n_ar, n_glass=n_glass, tempcoeff=tempcoeff,
                temp_model=temp_model, albedo=albedo, diffuse_model=diffuse_model,
                et_method=et_method, model_perez=model_perez,
                axis_tilt_constant=None, axis_azimuth_constant=None, clip=clip,
            )

            ### Record results
            results.append([
                iso.upper(), isonodes[i], dfin[pnodeid][i], dfin[pnodename][i],
                dfin['latitude'][i], dfin['longitude'][i], 
                dfin[latlonindex][i], tznode,
                capacity_factor, price_average, 
                value_average, revenue_yearly, value_factor,
                optcf_axis_tilt, optcf_axis_azimuth, optcf_cf, optcf_rev,
                optrev_axis_tilt, optrev_axis_azimuth, optrev_cf, optrev_rev
            ])

    ##########
    ### OUTPUT

    results_columns = [
        'ISO', 'ISO:Node', 'PNodeID', 'PNodeName', 
        'Latitude', 'Longitude', 'LatLonIndex', 'TimezoneNode',
        'Default_CF', 'Default_Price',
        'Default_Value', 'Default_Revenue', 'Default_VF',
        'OptCF_Tilt', 'OptCF_Azimuth', 'OptCF_CF', 'OptCF_Rev',
        'OptRev_Tilt', 'OptRev_Azimuth', 'OptRev_CF', 'OptRev_Rev',
    ]

    dfout = pd.DataFrame(data=results, columns=results_columns)

    ### Write the extra columns
    dfout['optimize'] = optimize
    dfout['axis_tilt_constant'] = axis_tilt_constant
    dfout['axis_azimuth_constant'] = axis_azimuth_constant
    
    ### Return if no write
    if write == False:
        return dfout

    ### Avoid empty results for runbegin/runend pairs that don't contain data
    if len(dfout) > 0:
        dfout.to_csv(savename, index=False, compression=compression)

        with open(describename, 'w') as f:
            f.write('datetime of run = {}\n'.format(pvvm.toolbox.nowtime()))
            f.write('savename of run = {}\n'.format(savename))
            f.write('script =          {}\n'.format(os.path.basename(__file__)))
            f.write('\n')
            f.write('isos =            {}\n'.format(abbrev))
            f.write('yearlmp =         {}\n'.format(yearlmp))
            f.write('yearsun =         {}\n'.format(yearsun))
            f.write('market =          {}\n'.format(market))
            f.write('resolutionlmp =   {}\n'.format(set_resolutionlmp))
            f.write('resolutionsun =   {}\n'.format(resolutionsun))
            f.write('\n')
            f.write('axis_tilt_constant =    {}\n'.format(axis_tilt_constant))
            f.write('axis_azimuth_constant = {}\n'.format(axis_azimuth_constant))
            f.write('default_axis_tilt =    {}\n'.format(default_axis_tilt))
            f.write('default_axis_azimuth = {}\n'.format(default_axis_azimuth))
            f.write('optimize =        {}\n'.format(optimize))
            f.write('ranges =          {}\n'.format(ranges))
            f.write('\n')
            f.write('pricecutoff =     {}\n'.format(pricecutoff))
            f.write('clip =            {}\n'.format(clip))
            f.write('\n')
            f.write('systemtype =      {}\n'.format(systemtype))
            f.write('dcac =            {}\n'.format(dcac))
            f.write('loss_system =     {}\n'.format(loss_system))
            f.write('loss_inverter =   {}\n'.format(loss_inverter))
            f.write('n_ar =            {}\n'.format(n_ar))
            f.write('n_glass =         {}\n'.format(n_glass))
            f.write('tempcoeff =       {}\n'.format(tempcoeff))
            f.write('temp_model =      {}\n'.format(temp_model))
            f.write('\n')
            f.write('max_angle =       {}\n'.format(max_angle))
            f.write('backtrack =       {}\n'.format(backtrack))
            f.write('gcr =             {}\n'.format(gcr))
            f.write('\n')
            f.write('albedo =          {}\n'.format(albedo))    
            f.write('diffuse_model =   {}\n'.format(diffuse_model))
            f.write('et_method =       {}\n'.format(et_method))
            f.write('model_perez =     {}\n'.format(model_perez))
            f.write('\n')
            f.write('revmpath =        {}\n'.format(revmpath))
            f.write('write =           {}\n'.format(write))
            f.write('runbegin =        {}\n'.format(runbegin))
            f.write('runend =          {}\n'.format(runend))
            f.write('nsrdbpath =       {}\n'.format(nsrdbpath))
            f.write('nsrdbtype =       {}\n'.format(nsrdbtype))
            f.write('lmppath =         {}\n'.format(lmppath))
            f.write('lmptype =         {}\n'.format(lmptype))
            f.write('\n')
            f.write("results_units =   'PNodeID': int\n")
            f.write("results_units =   'PNodeName': str\n")
            f.write("results_units =   'Latitude': float\n")
            f.write("results_units =   'Longitude': float\n")
            f.write("results_units =   'LatLonIndex': int\n")
            f.write("results_units =   'Timezone': str (Etc/GMT+{})\n")
            f.write("results_units =   'CapacityFactor': fraction\n")
            f.write("results_units =   'PriceAverage': $/MWh\n")
            f.write("results_units =   'ValueAverage': $/MWh\n")
            f.write("results_units =   'RevenueYearly': $/kWac\n")
            f.write("results_units =   'ValueFactor': fraction\n")