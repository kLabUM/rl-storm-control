# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:52:46 2016
@author: abhiram
"""
import numpy as np
import matplotlib.pyplot as plt


class pond:
    def __init__(self,area,max_height,inital_height=0.0):
        self.area = area
        self.initial_height = inital_height
        self.volume = area*inital_height
        self.max_height = max_height
        self.max_volume = area*max_height
        self.overflow = 0
        self.height = float(self.volume)/float(self.area)

    def timestep(self,dt):
        self.timestep=dt

    def qout(self,percent_opening):
        if percent_opening >=0 and percent_opening <= 1:
            return np.sqrt(2*9.81*self.volume/self.area)*percent_opening
        else:
            raise ValueError('A very specific bad thing happened with the gate opening, its not in 1 to 0')

    def volume_update(self,h):
        self.volume = self.volume + h*self.area
        self.height = float(self.volume)/float(self.area)
        if self.volume < 0:
            self.volume = 0
        if self.volume > self.max_volume:
            self.overflow = 1
            self.volume = self.max_volume
        else:
            self.overflow = 0

    def dhdt(self,qin,qout):
        h = float(qin-qout) * 1/float(self.area) * float(self.timestep)
        self.volume_update(h)

if __name__ == '__main__':
    test_pond=pond(100,2)
    test_pond.timestep(1.0)
    x=[]
    for i in range(0,100):
        qin=np.random.randint(0,10)
        qout=test_pond.qout(0.5)
        test_pond.dhdt(qin,qout)
        x.append(test_pond.volume)

    plt.plot(x)
    plt.show()
