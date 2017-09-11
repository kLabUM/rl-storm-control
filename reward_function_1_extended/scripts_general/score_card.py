import swmm

class score_board:
    def __init__(self, flow_limit):
        self.overflow = 0.0
        self.system_flow = 0.0
        self.scores = 0.0
        self.flow_bound = flow_limit

    def update(self, list_of_nodes, final_node):
        for i in list_of_nodes:
            if swmm.get(i, swmm.FLOODING, swmm.SI) > 0:
                self.overflow += 1.0
        temp = swmm.get(final_node, swmm.INFLOW, swmm.SI)
        if temp > self.flow_bound:
            self.system_flow += 1.0

    def score_is(self):
        self.scores = self.overflow #+ self.system_flow
        return self.scores
