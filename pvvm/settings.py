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
import os
# ### Use these if project folder is different from module folder
# revmpath = os.path.expanduser('~/path/to/project/folder/')
# datapath = os.path.expanduser('~/path/to/project/folder/Data/')
# ### Use these if project folder contains module folder
# revmpath = os.path.dirname(os.path.abspath(__file__)).replace('pvvm','')
# datapath = os.path.dirname(os.path.abspath(__file__)).replace('pvvm', 'Data/')
########### CUSTOM ###########
revmpath = os.path.expanduser('~/Desktop/pvvmtest1/')
datapath = os.path.expanduser('~/Desktop/pvvmtest1/Data/')
##############################

apikeys = {
    ### Get a googlemaps API key at 
    ### https://developers.google.com/maps/documentation/geocoding/get-api-key
    'googlemaps': 'yourAPIkey',
    ### Get an NSRDB key at https://developer.nrel.gov/signup/
    'nsrdb': 'yourAPIkey',
}
nsrdbparams = {
    ### Use '+' for spaces
    'full_name': 'your+name',
    'email': 'your@email.com',
    'affiliation': 'your+affiliation',
    'reason': 'your+reason',
    'mailing_list': 'true',
}
