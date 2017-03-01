import swmm

inp = 's.inp'
swmm.initialize(inp)

height1 = []
height2 = []
qout = []
t = 0
while not(swmm.is_over()):
    swmm.run_step()
    t = t + 1
swmm.close()
print t
