import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILENAME = '../data/iperf3.csv'
if FILENAME[-3:]=="csv":
    df = pd.read_csv(FILENAME)

print(df.head())

# class PLOT:
#     def __init__(self):
#         self.

#     def rtt(self): # By algorithm type {cubic, bbrv2} - plot mean, min, max
#         return NotImplemented

#     def bps(self): # By algorithm type {cubic, bbrv2}
#         return NotImplemented