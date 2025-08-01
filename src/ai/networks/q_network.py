from tf_agents.networks import q_network


def create_q_network(observation_spec, action_spec):
    return q_network.QNetwork(observation_spec, action_spec, fc_layer_params=(24, 24))
