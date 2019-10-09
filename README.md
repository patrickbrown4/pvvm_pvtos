This repository includes python scripts and input/output data associated with the following publication:

[1] Brown, P.R.; O'Sullivan, F. "Shaping photovoltaic array output to align with changing wholesale electricity price profiles." Applied Energy 2019. https://doi.org/10.1016/j.apenergy.2019.113734

Please cite reference [1] for full documentation if the contents of this repository are used for subsequent work.

Some of the scripts and data are also used in the following working paper:

[2] Brown, P.R.; O'Sullivan, F. "Spatial and temporal variation in the value of solar power across United States Electricity Markets". Working Paper, MIT Center for Energy and Environmental Policy Research. 2019. http://ceepr.mit.edu/publications/working-papers/705

All code is in python 3 and relies on a number of dependencies that can be installed using pip or conda.

Contents
--------
* pvvm/\*.py : Python module with functions for modeling PV generation, calculating PV revenues and capacity factors, and optimizing PV orientation.
* notebooks/\*.ipynb : Jupyter notebooks, including:
    * pvvm-pvtos-data.ipynb: Example scripts used to download and clean input LMP data, determine LMP node locations, and reproduce some figures in reference [1]
    * pvvm-pvtos-analysis.ipynb: Example scripts used to perform the calculations and reproduce some figures in reference [1]
    * pvvm-pvtos-plots.ipynb: Scripts used to produce additional figures in reference [1]
    * pvvm-example-generation.ipynb: Example scripts demonstrating the usage of the PV generation model and orientation optimization
* html/\*.html : Static images of the above Jupyter notebooks for viewing without a python kernel
* data/lmp/\*.gz : Day-ahead and real-time nodal locational marginal prices (LMPs) for CAISO, ERCOT, MISO, NYISO, and ISONE.
    * At the time of publication of this repository, permission had not been received from PJM to republish their LMP data. If permission is received in the future, a new version of this repository will linked here with the complete dataset.
* results/\*.csv.gz : Simulation results associated with reference [1] above, including modeled revenue, capacity factor, and optimized orientations for PV systems at all LMP nodes

Data notes
----------
* ISO LMP data are used with permission from the different ISOs. Adapting the MIT License (https://opensource.org/licenses/MIT), "The data are provided 'as is', without warranty of any kind, express or implied, including but not limited to the warranties of merchantibility, fitness for a particular purpose and noninfringement. In no event shall the authors or sources be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the data or other dealings with the data." Copyright and usage permissions for the LMP data are available on the ISO websites, linked below.
* ISO-specific notes:
    * CAISO data from http://oasis.caiso.com/mrioasis/logon.do are used pursuant to the terms at http://www.caiso.com/Pages/PrivacyPolicy.aspx#TermsOfUse.
    * ERCOT data are from http://www.ercot.com/mktinfo/prices.
    * MISO data are from https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/ and https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/market-report-archives/.
    * PJM data were originally downloaded from https://www.pjm.com/markets-and-operations/energy/day-ahead/lmpda.aspx and https://www.pjm.com/markets-and-operations/energy/real-time/lmp.aspx. At the time of this writing these data are currently hosted at https://dataminer2.pjm.com/feed/da_hrl_lmps and https://dataminer2.pjm.com/feed/rt_hrl_lmps.
    * NYISO data from http://mis.nyiso.com/public/ are used subject to the disclaimer at https://www.nyiso.com/legal-notice.
    * ISONE data are from https://www.iso-ne.com/isoexpress/web/reports/pricing/-/tree/lmps-da-hourly and https://www.iso-ne.com/isoexpress/web/reports/pricing/-/tree/lmps-rt-hourly-final. The Material is provided on an "as is" basis. ISO New England Inc., to the fullest extent permitted by law, disclaims all warranties, either express or implied, statutory or otherwise, including but not limited to the implied warranties of merchantability, non-infringement of third parties' rights, and fitness for particular purpose. Without limiting the foregoing, ISO New England Inc. makes no representations or warranties about the accuracy, reliability, completeness, date, or timeliness of the Material. ISO New England Inc. shall have no liability to you, your employer or any other third party based on your use of or reliance on the Material.
* Data workup: LMP data were downloaded directly from the ISOs using scripts similar to the pvvm.data.download_lmps() function (see below for caveats), then repackaged into single-node single-year files using the pvvm.data.nodalize() function. These single-node single-year files were then combined into the dataframes included in this repository, using the procedure shown in the pvvm-pvtos-data.ipynb notebook for MISO. We provide these yearly dataframes, rather than the long-form data, to minimize file size and number. These dataframes can be unpacked into the single-node files used in the analysis using the pvvm.data.copylmps() function.

Usage notes
-----------
* To use the NSRDB download functions, you will need to modify the "settings.py" file to insert a valid NSRDB API key, which can be requested from https://developer.nrel.gov/signup/. Locations can be specified by passing latitude, longitude floats to pvvm.data.downloadNSRDBfile(), or by passing a string googlemaps query to pvvm.io.queryNSRDBfile(). To use the googlemaps functionality, you will need to request a googlemaps API key (https://developers.google.com/maps/documentation/javascript/get-api-key) and insert it in the "settings.py" file.
* Note that many of the ISO websites have changed in the time since the functions in the pvvm.data module were written and the LMP data used in the above papers were downloaded. As such, the pvvm.data.download_caiso_lmp_allnodes() and pvvm.data.download_lmps() functions no longer work for all ISOs and years. We provide these functions to illustrate the general procedure used, and do not intend to maintain them or keep them up to date with the changing ISO websites. For up-to-date functions for accessing ISO data, the following repository (no connection to the present work) may be helpful: https://github.com/catalyst-cooperative/pudl.
