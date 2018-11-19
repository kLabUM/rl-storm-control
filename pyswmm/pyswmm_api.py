from pyswmm import Simulation

class Env:
    def __init__(self, input_file):
        self.input_file = input_file
        self.sim = Simulation(self.input_file)
        self.sim.start()
        self.time = 1.0


    def step(self):
        if self.time > 0:
            done = False
            self.time = self.sim._model.swmm_step()
        else:
            done = True
            print("Simulation Ended")
        return done
        
    def terminate(self):
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()

    def depthN(self, node_id):
        return self.sim._model.getNodeResult(node_id, 5)

    def depthL(self, link_id):
        return self.sim._model.getLinkResult(link_id, 1)

    def flow(self, link_id):
        return self.sim._model.getLinkResult(link_id, 0)

    def get_precip(self, subcatchment_id):
        pass

    def set_precip(self, subcatchment_id):
        pass

    def get_gate(self, orifice):
        return self.sim._model.getLinkResult(orifice, 6)

    def set_gate(self, orifice, gate):
        self.sim._model.setLinkSetting(orifice, gate)

    def get_pollutant_node(self, node):
        return self.sim._model.getNodeResult(node, 100)

    def get_pollutant_link(self, link):
        return self.sim._model.getLinkResult(link, 100)

    def reset(self):
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()
        # Start the next simulation
        self.sim._model.swmm_open()
        self.sim._model.swmm_start()
        self.time = 1.0
