#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:26:25 2016

@author: abhiram
"""

import csv


with open('cs.csv', 'w', newline=' ') as csvfile:
    spam_writer = csv.writer(csvfile, dlimiter = '',
                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])