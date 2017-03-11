from core_network import stacker, replay_stacker

class pond_tracker:
    def __init__(self,
                 pond_id,
                 orifice_id,
                 states,
                 replay_window):

        self.pond_id = pond_id
        self.orifice_id = orifice_id
        self.states = states
        self.replay_window = replay_window

        # Initialize replay memory
        self.replay_memory = {'states': replay_stacker(self.states,
                                                       self.replay_window),
                              'states_new': replay_stacker(self.states,
                                                           self.replay_window),
                              'rewards': replay_stacker(1, self.replay_window),
                              'actions': replay_stacker(1, self.replay_window),
                              'terminal': replay_stacker(1, self.replay_window)}

        # Initialize tracking memory for each simulation
        self.tracker_pond = {'depth': stacker(1),
                             'inflow': stacker(1),
                             'outflow': stacker(1),
                             'flooding': stacker(1),
                             'gate_position': stacker(1),
                             'rewards': stacker(1)}

        # Bookkeeping terms
        self.bookkeeping = {'mean_rewards': stacker(1),
                            'mean_depth': stacker(1),
                            'mean_outflow': stacker(1),
                            'mean_flooding': stacker(1)}

    def replay_memory_update(self,
                             states,
                             states_new,
                             rewards,
                             actions,
                             terminal):

        self.replay_memory['rewards'].update(rewards)
        self.replay_memory['states'].update(states)
        self.replay_memory['states_new'].update(states_new)
        self.replay_memory['actions'].update(actions)
        self.replay_memory['terminal'].update(terminal)

    def tracker_update(self, N):
        depth,inflow,outflow,flooding,gate_position = N
        self.tracker_pond['depth'].update(depth)
        self.tracker_pond['inflow'].update(inflow)
        self.tracker_pond['outflow'].update(outflow)
        self.tracker_pond['flooding'].update(flooding)
        self.tracker_pond['gate_position'].update(gate_position)

    def record_mean(self):
        self.bookkeeping['mean_rewards'] = np.mean(self.tracker_pond['rewards'])
        self.bookkeeping['mean_depth'] = np.mean(self.tracker_pond['depth'])
        self.bookkeeping['mean_outflow'] = np.mean(self.tracker_pond['outflow'])
        self.bookkeeping['mean_flooding'] = np.mean(self.tracker_pond['flooding'])

    def forget_past(self):
        self.tracker_pond = {'depth': stacker(1),
                             'inflow': stacker(1),
                             'outflow': stacker(1),
                             'flooding': stacker(1),
                             'gate_position': stacker(1),
                             'rewards': stacker(1)}
