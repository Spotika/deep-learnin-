import Network
import numpy as np
params = {
    'IN_DIM' : 4,
    'HIDDEN_L' : [5],
    'OUT_DIM' : 3,
}

Test = Network.Network(params)

x = np.random.randn(params['IN_DIM'])

print(Test.get_weights())
Test.save_weights()