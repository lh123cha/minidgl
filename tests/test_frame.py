# import sys
# sys.path.append('./python')
# import itertools
# import torch
# import minidgl.python.frame as F

# import needle as ndl
# from needle import backend_ndarray as nd

import sys
sys.path.append('./python')
import numpy as np
import pytest
import torch
import itertools
import minidgl.python.frame as F
import minidgl.python.indexutils as util

import needle as ndl
import needle.nn as nn


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


FRAME_PARAMETER = [("test_f1"),("test_f2"),("test_f3")]



@pytest.mark.parametrize("column_name",FRAME_PARAMETER)
@pytest.mark.parametrize("device",_DEVICES,ids=['cpu','cuda'])
def test_frame_add_column(column_name,device):
    data = {
        "fa":ndl.Tensor([[1,2,1],[2,5,2]]),
        "hb":ndl.Tensor([[1,2,2],[2,5,6]])
    }

    frame=F.Frame(data,2)
    frame.add_column(name=column_name,scheme=frame.schemes["fa"],ctx=device)
    frame.show_column(column_name)

NUM_ROWS_PARAMETERS = [2,3,4]
@pytest.mark.parametrize("num_rows",NUM_ROWS_PARAMETERS)
@pytest.mark.parametrize("device",_DEVICES,ids=['cpu','cuda'])
def test_frame_add_rows(num_rows,device):
    data = {
        "fa":ndl.Tensor([[1,2,1],[2,5,2]]),
        "hb":ndl.Tensor([[1,2,2],[2,5,6]])
    }
    frame = F.Frame(data,2)
    frame.add_rows(2)
    assert(frame._num_rows==4)
    print(frame._columns["fa"].data)

COLUMN_NAME_PARAMETER = ["fa","fb","fc"]
UPDATE_DATA_PARAMETER = [ndl.Tensor([[7,8,9],[4,5,6]])]

@pytest.mark.parametrize("column_name",COLUMN_NAME_PARAMETER)
@pytest.mark.parametrize("update_data",UPDATE_DATA_PARAMETER)
@pytest.mark.parametrize("device",_DEVICES,ids=['cpu','cuda'])
def test_update_frame(column_name,update_data,device):
    data = {
        "fa":ndl.Tensor([[1,2,1],[2,5,2]]),
        "fb":ndl.Tensor([[1,2,2],[2,5,6]]),
        "fc":ndl.Tensor([[1,1,1],[2,2,2]])
    }
    frame = F.Frame(data,2)
    frame.update_column(column_name,update_data)
    print(frame._columns[column_name].data)
    assert(frame._columns[column_name].data == update_data)

@pytest.mark.parametrize("column_name",COLUMN_NAME_PARAMETER)
def test_column_update(column_name):
    tensor_data = ndl.Tensor([[1,2,1],[2,5,2],[55,5,7],[9,8,7]])
    print(tensor_data.numpy()[0])
    c1 = F.Column(tensor_data,F.infer_scheme(tensor_data))
    c1.update(util.Index([0,2]),ndl.Tensor([[2,5,10],[0,0,0]]))
    print(type(c1.data))
    assert c1.data.numpy().sum() == 17
