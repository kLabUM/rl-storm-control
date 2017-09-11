import random as r
import swmm

def select_random_event(duration, base_name, end_name):
    temp1 = r.sample(duration, 1)
    return base_name+temp1[0]+end_name

duration = ['18','12','1','24','2','3','6']
base_name = 'aa_orifices_v3_scs_10yr_'
times = {}
for i in duration:
    inp = base_name+i+'hr.inp'
    swmm.initialize(inp)
    t = 0
    while t<60540:
        swmm.run_step()
        t=t + 1
    times[inp] = t

for i in times.keys():
    print i
    print times[i]
