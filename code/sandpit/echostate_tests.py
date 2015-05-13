
### Tests for the echo state network code.

import numpy as np
from echostate import EchoStateNetwork

def test_hidden_weights():
    """
    Test the initialization for the hidden units
    :return:
    """
    ## dummy data
    x = np.arange(10)
    y = np.ones_like(x)

    ## choose value for r
    r = 0.1
    ## choose value for b, should be different from r
    ## for demonstration purposes
    b = 0.5

    ## pick 10 hidden units
    N = 10

    ## pick a weight for the input units
    a = 0.75

    #### go through the topologies

    ## SCR first
    esn = EchoStateNetwork(x,y,N,a,r,b,topology="scr")
    print("SCR topology: \n "
          "Elements should be on lower sub-diagonal only (U_(i+1,i)), plus one"
          "element on position in upper right corner: (0,N) \n" + str(esn.uu) + "\n")

    ## DLR second
    esn = EchoStateNetwork(x,y,N,a,r,b,topology="dlr")
    print("DLR topology: \n "
          "Elements should be on lower sub-diagonal only (U_(i+1,i)) \n" + str(esn.uu) + "\n")

    ## DLRB third
    esn = EchoStateNetwork(x,y,N,a,r,b,topology="dlrb")
    print("DLRB topology: \n "
          "Elements should be on lower sub-diagonal (U_(i+1,i)), "
          "on upper subdiagonal (U_(i,i+1)) \n" + str(esn.uu) + "\n")

    return


def test_input_weights():
    """
    Test the initialization of the weights of connections between input units and hidden units.
    Testing for both a single time series and multivariate data.
    :return:
    """
    ## dummy data
    x = np.arange(10)

    ## single data stream
    y = np.ones_like(x)

    ## double data stream
    yd = np.ones((len(x), 2))
    print(yd.shape)

    ## choose value for r
    r = 0.1

    ## pick a weight for the input weights
    a = 0.75

    ## pick 10 hidden units
    N = 10

    ## single data stream
    esn = EchoStateNetwork(x,y,N,a,r,topology="scr")
    print("Input weights for single data stream: \n " + str(esn.vv))

    ## double data stream
    esn = EchoStateNetwork(x,yd,N,a,r,topology="scr")
    print("Input weights for single data stream (should be randomly -0.75 and 0.75: \n " + str(esn.vv))
    print("shape of data points: " + str(esn.y.shape))
    print("shape of input weights (should be the same as shape of data): " + str(esn.vv.shape))

    return

### run all tests
test_hidden_weights()
test_input_weights()
