# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:00:37 2019

@author: Scarlett
"""

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--flag_int',type = float,default = 0.01,help = 'flag_int.')

flags, unparsed = parser.parse_known_args()
print('flags : {}'.format(flags))
print('unparsed : {}'.format(unparsed))

#未定义的参数都在 unparsed 中 程序不报错
#D:\myGit\Algorithm\tensorflow_example>python test_argparse.py --flag_int 1 --flag_bool True --flag_string "haha" --flag_float 0.2
