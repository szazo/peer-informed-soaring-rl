import numpy as np
from tianshou.data import Batch


def test_batch_sandbox():

    data = dict(a=[11, 22], obs=dict(a0=[1, 2, 4, 5], a1=[3, 4, 5]))

    batch = Batch(data)

    print(batch)
    print('batch.obs.a0.shape', batch.obs.a0.shape)
    print('batch.obs.a1.shape', batch.obs.a1.shape)

    concat = Batch([data] * 3)
    print(concat)
    print('concat.obs.shape', concat.obs.a0.shape)
    print('concat.obs.shape', concat.obs.a1.shape)

    print('concat[0]', concat[0])
    print('concat[[0,3]]', concat[[0, 1]])
    print('concat[-2:]', concat[-2:])

    # change

    tmp = concat[0]
    tmp.f = 42
    tmp.a[0] = 42
    print('CHANGED', tmp)

    print('ORIGINAL', concat)

    # del concat.obs.a0
    # print('deleted', concat)


def test_aggregation():

    pass

    b1 = Batch(a=[dict(b=np.float64(1.0), d=dict(e=np.array(3.0)))])
    b2 = Batch(a=[dict(b=np.float64(4.0), d=dict(e=np.array(6.0)))])

    print('b1', b1)
    print('b2', b2)

    b12_cat_out = Batch.cat([b1, b2])
    print('b12_cat_out', b12_cat_out)

    b3 = Batch(a=np.zeros((3, 2)), b=np.ones((2, 3)), c=Batch(d=[[1], [2]]))
    b4 = Batch(a=np.ones((3, 2)), b=np.ones((2, 3)), c=Batch(d=[[0], [3]]))
    print('b3', b3, b3.c.d.shape)
    print('b4', b4, b4.c.d.shape, b4.b.shape)
    b34_stack = Batch.stack((b3, b4), axis=1)
    print('b34_stack', b34_stack, b34_stack.c.d.shape, b34_stack.b.shape)

    print("========================================")
    print(type(b34_stack.split(1)))
    print(list(b34_stack.split(1, shuffle=True)))
