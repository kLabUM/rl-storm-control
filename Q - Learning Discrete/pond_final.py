import numpy as np
import scipy as spy
from scipy import integrate
import matplotlib.pyplot as plt

class pond:
    def __init__(self,area,inital_height):
        self.area = area
        self.initial_height = inital_height
        self.volume = area*inital_height

    def qout(self,percent_opening):
        if self.volume < 0:
            v = 0
        else:
            v = self.volume
        return np.sqrt(2*9.81*v/self.area)*percent_opening

    def volume_update(self,h):
        self.volume = self.volume + h*self.area

def change(t,h,qin,qout):
    return (qin-qout)/100

def reward(h,valve_position):
    area = 1
    cd = 1
    qout = np.sqrt(2*9.81*h) * valve_position * area * cd
    if h >= 0.8 and h <=0.9:
        if qout >= 0 and qout <= 2:
            return 100
        else:
            return 0
    else:
        return 0

def state2reality(volume):
    height = volume/100
    if height <= 0.9:
        h=int(10*np.round_(height,1))
    else:
        h=9
    return h

Q=np.zeros(shape=(10,10))
for i in range(0,10000):
    action = np.random.randint(0,10)
    state = np.random.randint(0,10)
    Q[state,action] = reward(state/10.0,action/10.0) +  0.6* np.amax(Q[state,])

Q = Q/np.amax(Q)

print Q

r = spy.integrate.ode(change)
r.set_integrator('dopri5')
t0=0;h0=0
r.set_initial_value(h0,t0)
dt=1;y=[];k=0

p = pond(100,0)
qo=[]
while k < 1000:
    k=k+1
    qin=0.1
    state = state2reality(p.volume)
    percent_opening = np.argmax(Q[int(state),])
    print percent_opening
    qout=p.qout(percent_opening/10)
    qo.append(qout)
    r.set_f_params(qin,qout)
    r.integrate(r.t + dt)
    x=r.y
    p.volume_update(x)
    y.append(p.volume/p.area)

plt.plot(y)
plt.show()
