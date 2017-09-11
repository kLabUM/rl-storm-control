import visdom
import numpy as np

def load(i):
    return np.load("outflow_mean/outflow_lastprof_"+str(i)+".h5.npy")

vis = visdom.Visdom()
a = np.load("outflow_mean/outflow_lastprof_0.h5.npy")

j=0
for i in range(0,10):
    z = i * 10 + 1
    if z != 1:
        win=vis.line(np.load("outflow_mean/outflow_lastprof_"+str(z)+".h5.npy"),opts=dict(legend=[str(z)]))
    else:
        win=vis.line(np.load("outflow_mean/outflow_lastprof_"+str(0)+".h5.npy"),opts=dict(legend=[str(0)]))

    for ii in range(2,10):
        vis.updateTrace(X=np.linspace(0,len(a),len(a)),Y=np.load("outflow_mean/outflow_lastprof_"+str(z+ii)+".h5.npy"),win=win,
                        name=str(ii+z))
