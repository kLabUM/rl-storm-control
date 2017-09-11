import swmm
inp = 'aa_orifices_v3_scs_0005min_001yr.inp'
swmm.initialize(inp)
time = 0.0
while not(swmm.is_over()):
    swmm.run_step()
    time += 1
print time
