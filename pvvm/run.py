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
import pvvm.model


#######################
### GENERAL CLASSES ###
#######################

class makeUSA(object):
    """
    """
    def __init__(self, ferc=True, capacity=True):
        ### Load node-egrid region lineup
        dfnodes = pd.read_csv(
            # revmpath+'USA/io/usa-nodes-eGRID-region.csv')
            revmpath+'io/CEMPNI-nodes-eGRID-region.csv')
        self.dfnodes = dfnodes
        isonode2egrid = dict(zip(dfnodes['ISO:Node'],dfnodes['eGRID']))
        self.isonode2egrid = isonode2egrid

        ## Load all emissions data for faster function operation
        dfemissions = pvvm.model.getemissions(returndfemissions=True)
        self.dfemissions = dfemissions
        
        ### FERC load
        if ferc is not False:
            self.dfferc = pvvm.io.getferc()
        
        ### Capacity prices
        if capacity is not False:
            capvalpath = (
                revmpath+'io/capacityprice-CEMPNI-2010_2017-nominal-dict.p')
            with open(capvalpath, 'rb') as p:
                self.dictcapacityprice = pickle.load(p)
        
        ### Carbon markets
        with open(revmpath+'io/isonode2carbonmarket.p', 'rb') as p:
            self.isonode2carbonmarket = pickle.load(p)
            
        ###### Carbon prices
        ### CARB
        carbpath = datapath + 'California/carbon-pricing/carb-auction-settlements.xlsx'
        dfcarbin = pd.read_excel(carbpath)
        dfcarb = dfcarbin.loc[dfcarbin.notes=='yearly_mean'].copy()
        co2price_carb_nominal = dict(zip(dfcarb.auction_year, dfcarb.settlement_price_nominal))
        self.co2price_carb_nominal = {**{2010:0, 2011:0}, **co2price_carb_nominal}
        
        ### RGGI
        rggipath = datapath + 'RGGI/AllowancePricesAndVolumes_average.csv'
        dfrggiin = pd.read_csv(rggipath)
        dfrggi = dfrggiin.loc[dfrggiin.notes=='yearly_mean'].copy()
        self.co2price_rggi_nominal = dict(zip(dfrggi.year, dfrggi.clearing_price_nominal))

class makeISO(object):
    """
    """
    def __init__(self, iso, USA=None):
        ### Load USA if None
        if USA is None:
            self.USA = makeUSA()
        else:
            self.USA = USA
        ### Load all nodes
        self.iso = iso.upper()
        self.dfnodes = pvvm.io.get_iso_nodes(self.iso)
        self.nodes = self.dfnodes.node.values
        self.isonodes = (self.iso+':'+self.dfnodes.node.astype(str)).values
        
        ### Create dictionary of node-to-NSRDBlatlonindex
        if 'latlonindex' in self.dfnodes.columns:
            self.node2latlonindex = dict(self.dfnodes[['node','latlonindex']].values)
        else:
            self.node2latlonindex = dict(self.dfnodes[['node','node']].values)
        
        ### Create dictionary of node-to-latlon
        self.node2latlon = dict(zip(
            self.dfnodes.node.values, self.dfnodes[['latitude','longitude']].values.tolist()
        ))
    
    def __str__(self):
        return self.iso
    
    def __repr__(self):
        return 'makeISO({})'.format(self.iso)
    
    def get_fulltimenodes(self, year, market='da', geo=True, output='node'):
        if geo is True:
            fulltimenodes = pvvm.io.get_iso_nodes(
                iso=self.iso, market=market, yearlmp=year,
                merge=True)['node']
        elif geo is False:
            fulltimenodes = pvvm.io.get_iso_nodes(
                iso=self.iso, market=market, yearlmp=year,
                fulltimeonly=True)['node']
        
        if output in ['node','nodes']:
            pass
        elif output in ['isonode','isonodes']:
            fulltimenodes = self.iso + ':' + fulltimenodes.astype(str)

        return fulltimenodes.values
    
    def get_load(
        self, year, source='ferc', error='ignore', clean=True,
        units='MW', division='region', filepathin=None,):
        """
        """
        unitconverter = {'W': 1E-6,'kW':1E-3,'MW':1,'GW':1000,'TW':1000000}
        if source in ['ferc', 'FERC', None]:
            dfload = pvvm.io.getload_ferc(
                iso=self.iso, year=year, dfferc=self.USA.dfferc, 
                error=error) / unitconverter[units]
        elif source in ['ISO', 'iso']:
            dfload = pvvm.io.getload(
                iso=self.iso, year=year, clean=clean, units=units,
                division=division, filepathin=filepathin)
        return dfload
    
    def get_system_wind():
        pass
    
    def get_system_solar():
        pass
    
    def get_netload(self, year, net='vre', resolution=60, units='MW'):
        """
        """
        df = pvvm.io.get_netload(
            iso=self.iso, year=year, net=net, resolution=resolution, units=units)
        return df
    
    def get_capacitynodes(self, year):
        """
        Return nodes that have a capacity price in year
        """
        nodelist = [node for node in self.nodes 
                    if ('{}:{}'.format(self.iso, node), year, 1) 
                    in self.USA.dictcapacityprice.keys()]
        return nodelist
        
    def get_critical_hours(
        self, year, percent='iso', numhours=None, net=None, resolution=60,
        dfload=None, dropleap=False, ):
        """
        Notes
        -----
        Default percent for CAISO and ERCOT is 7.04 percent
        """
        ### Input check to avoid ambiguity
        if numhours is not None: assert percent is None

        ### Carry on
        if (percent in ['iso','ISO','ISO-specified',None]) and (numhours is None):
            if self.iso in ['MISO','PJM','NYISO','ISONE']:
                mask = pvvm.model.get_iso_critical_hours(
                    iso=self.iso, year=year, 
                    resolution={60:'H',30:'30T'}[resolution])
            else:
                print(self.iso)
                raise Exception('No ISO-defined critical hours for CAISO, ERCOT')
        else:
            if dfload is None:
                dfload = self.get_netload(
                    year=year, net=net, resolution=resolution,
                )
            else:
                pass
            
            if numhours is not None:
                mask = pvvm.model.getcriticalhours(
                    dsload=dfload, percent=None, year=year, numhours=numhours,
                    dropleap=dropleap, resolution=resolution,)
            else:
                mask = pvvm.model.getcriticalhours(
                    dsload=dfload, percent=percent, year=year, numhours=None,
                    dropleap=dropleap, resolution=resolution,)
        
        return mask

class makeNode(object):
    """
    """
    def __init__(self, isonode=None, ISO=None, node=None):
        if isonode is not None:
            self.isonode = isonode
        else:
            self.isonode = '{}:{}'.format(ISO.iso, node)
            
        if ISO is not None:
            self.ISO = ISO
            self.iso = self.ISO.iso.upper()
        else:
            self.iso = self.isonode.split(':')[0].upper()
            self.ISO = makeISO(self.iso)
        
        if node is not None:
            self.node = node
        else:
            self.node = isonode.split(':')[1]
            
        ### Get timezone
        self.tz = pvvm.toolbox.tz_iso[self.iso]
        self.timezone = pvvm.toolbox.timezone_iso[self.iso]
        
        ### Get NSRDB latlonindex
        try:
            self.latlonindex = self.ISO.node2latlonindex[self.node]
        except KeyError:
            self.latlonindex = self.ISO.node2latlonindex[int(self.node)]
        
        ### Get eGRID region
        self.egridregion = self.ISO.USA.isonode2egrid[self.isonode]
        self.egrid = self.ISO.USA.isonode2egrid[self.isonode]
        
        ### Get location
        try:
            self.latitude, self.longitude = self.ISO.node2latlon[self.node]
            self.latlon = tuple(self.ISO.node2latlon[self.node])
        except KeyError:
            self.latitude, self.longitude = self.ISO.node2latlon[int(self.node)]
            self.latlon = tuple(self.ISO.node2latlon[int(self.node)])
    
    def __str__(self):
        return self.isonode
    
    def __repr__(self):
        return 'makeNode({})'.format(self.isonode)
    
    def get_price(self, year, product='lmp', market='da'):
        """
        """
        filepath = revmpath+'{}/io/{}-nodal/{}/'.format(
            self.iso, product, market)
        dflmp = pvvm.io.getLMPfile(
            filepath, '{}-{}.gz'.format(self.node, year), 
            self.tz, squeeze=True, product=product)
        return dflmp
    
    def get_nsrdb_filepath(self, year):
        resolution = 60 if year == 'tmy' else 30
        filepath = revmpath+'{}/in/NSRDB/{}/{}min/'.format(
            self.iso, year, resolution)
        filename='{}-{}.gz'.format(self.latlonindex, year)
        return filepath+filename
    
    def get_nsrdb_file(self, year, returnall=True):
        resolution = 60 if year == 'tmy' else 30
        dfsun, info, tz, elevation = pvvm.io.getNSRDBfile(
            filepath=revmpath+'{}/in/NSRDB/{}/{}min/'.format(
                self.iso, year, resolution,
            ),
            filename='{}-{}.gz'.format(self.latlonindex, year),
            year=year, resolution=resolution, forcemidnight=False
        )
        if returnall is True:
            return dfsun, info, tz, elevation
        else:
            return dfsun
        
    def get_emissions(
        self, year, pollutant='co2', emissions='marginal',
        measurement='emissions', source='easiur',
        tz=None, dollaryear=2017, 
        dfemissions=None):
        """
        """
        if tz is None:
            tz = self.tz
        if dfemissions is None:
            dfemissions = self.ISO.USA.dfemissions
        return pvvm.model.getemissions(
            year=year, region=self.egrid, pollutant=pollutant,
            emissions=emissions, measurement=measurement,
            source=source, tz=tz, dollaryear=dollaryear,
            dfemissions=dfemissions)
    
    def get_capacity_price(
        self, year, mask=None, dictcapacityprice=None, 
        units='$/kWh', **kwargs):
        """
        Notes
        -----
        * **kwargs are passed to ISO.get_critical_hours(
            self, year, percent='iso', numhours=None, net=None, 
            resolution=60, dfload=None, dropleap=False)
        """
        if units in ['kW', 'kw', 'KW', '$/kWh', 'kWh']:
            unitifier = 1
        elif units in ['MW', 'mw', '$/MWh', 'MWh']:
            unitifier = 1000

        if dictcapacityprice is None:
            dictcapacityprice = self.ISO.USA.dictcapacityprice
        if mask is None:
            if self.iso in ['CAISO', 'ERCOT']:
                assert (('percent' in kwargs) or ('numhours' in kwargs))
            mask = self.ISO.get_critical_hours(
                year=year, **kwargs)
    
        dfcapacityprice = pvvm.model.get_capacity_price(
            isonode=self.isonode, year=year, mask=mask, 
            dictcapacityprice=dictcapacityprice) * unitifier
        
        return dfcapacityprice
    
    def get_co2_market_price(self, year, dollaryear='nominal'):
        """
        """
        carbonmarket = self.ISO.USA.isonode2carbonmarket[self.isonode]
        if carbonmarket == 'CARB':
            co2price_nominal = self.ISO.USA.co2price_carb_nominal[year]
        elif carbonmarket == 'RGGI':
            co2price_nominal = self.ISO.USA.co2price_rggi_nominal[year]
        else:
            co2price_nominal = 0.
        if dollaryear == 'nominal':
            return co2price_nominal
        else:
            co2price_inflated = (
                co2price_nominal * pvvm.model.inflate(yearin=year, yearout=dollaryear))
            return co2price_inflated
