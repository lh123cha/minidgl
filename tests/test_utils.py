# import sys
# sys.path.append('./python')
# sys.path.append('./minidgl/python')
# import numpy as np
# import pytest
# import torch
# import itertools

# import needle as ndl
# import needle.nn as nn

# import utils as util


import sys
sys.path.append('./python')
import numpy as np
import pytest
import torch
import itertools
import minidgl.python.indexutils as util

import needle as ndl
import needle.nn as nn

NUM=[10]

@pytest.mark.parametrize("l",NUM)
def test_index_from_numpy(l):
    ans = np.ones((l,),dtype=np.int64)*10
    data = np.ones((l,),dtype=np.int64)*10
    idx = util.toindex(data)
    y1 = idx.tonumpy()
    print(y1)
    assert np.allclose(ans, y1)

@pytest.mark.parametrize("l",NUM)
def test_index_from_list(l):
    data = [10]*10
    idx = util.toindex(data)
    y1 = idx.tonumpy()
    y2 = idx.todgltensor().numpy()
    assert np.allclose(y1,y2)