"""MP2RAGE tests."""

from blochsim import mp2rage_sim


def test_scalar_forward():
    sig = mp2rage_sim(
        nshots=100,
        TI=(200.0, 1500.0),
        flip=5.0,
        TRspgr=10.0,
        TRmp2rage=5000.0,
        T1=1000.0,
    )
    assert sig.shape == (2,)


def test_multiple_forward():
    sig = mp2rage_sim(
        nshots=100,
        TI=(200.0, 1500.0),
        flip=5.0,
        TRspgr=10.0,
        TRmp2rage=5000.0,
        T1=(200, 500, 1000.0),
    )
    assert sig.shape == (3, 2)


def test_scalar_derivative():
    _, dsig = mp2rage_sim(
        nshots=100,
        TI=(200.0, 1500.0),
        flip=5.0,
        TRspgr=10.0,
        TRmp2rage=5000.0,
        T1=1000.0,
        diff="T1",
    )
    assert dsig.shape == (2,)


def test_multiple_derivative():
    _, dsig = mp2rage_sim(
        nshots=100,
        TI=(200.0, 1500.0),
        flip=5.0,
        TRspgr=10.0,
        TRmp2rage=5000.0,
        T1=(200, 500, 1000.0),
        diff="T1",
    )

    assert dsig.shape == (3, 2)
