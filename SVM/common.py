# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:00:27 2018

@author: shine
"""

#! /usr/bin/env python
import logging
import logging.handlers
import os
import yaml
LOG_FILENAME = 'logging.out'

# Set up a specific logger with our desired output level
logger = logging.getLogger('MyLogger')
logger.setLevel(logging.DEBUG)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(
              LOG_FILENAME, maxBytes=10*1024*1024, backupCount=2)
formatter = logging.Formatter("[%(asctime)s] [%(filename)s][line:%(lineno)d][func: %(funcName)s] - [%(levelname)s] : %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

logger.warning('logger is work')

#configdir = os.getcwd()+'/config/httpconfig.yml'
#with open(configdir,"r") as f:
#    message = yaml.load(f)

#
#header = {"Content-type": "application/json", "Accept-Encoding": "utf-8"}
#url1 = "http://10.20.250.103:5000/addressSegment/"
#url2 = "http://10.20.250.103:8410/address_jiexi/"
#url3 = "http://10.20.250.103:8881/similarityes"
#urls = [url1,url2,url3]
#
#aproject = {'header': header,
#            'urls': urls
#            }
