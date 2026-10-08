"""Headless test of the demo's teaching logic (no window is opened).

Drives LakeDemo through the presentation script: raise a past the fold,
treat the lake chemically, lower a back into the bistable window, and so on,
checking that the ball ends where the model says it should.
"""

import importlib.util

import matplotlib

matplotlib.use("Agg", force=True)  # draw off-screen

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from lake.config import REPO_ROOT, load_params  # noqa: E402
from lake.equilibria import equilibria, folds  # noqa: E402

spec = importlib.util.spec_from_file_location("demo", REPO_ROOT / "demo" / "demo.py")
demo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_module)

PARAMS = load_params()
NUM = PARAMS["numerics"]
ROOT_ARGS = (NUM["n_grid"], NUM["xtol"], NUM["tangency_tol"])
Q = PARAMS["model"]["q"]
B0 = PARAMS["model"]["b"]
UPPER, LOWER = folds(B0, Q, NUM["n_grid"], NUM["xtol"])
EQ_TOL = 1e-6


def _run_until_rest(demo):
    while demo.frames:
        demo.tick()


def _clear_eq(a):
    return equilibria(a, B0, Q, *ROOT_ARGS)[0].x


def _turbid_eq(a):
    return equilibria(a, B0, Q, *ROOT_ARGS)[-1].x


def _make_demo():
    """LakeDemo without screen redraws: these tests check the logic only, and
    off-screen drawing of every frame would make them slow."""
    d = demo_module.LakeDemo(PARAMS, interactive=False)
    d.fig.canvas.draw_idle = lambda: None
    return d


@pytest.fixture
def demo():
    d = _make_demo()
    yield d
    plt.close(d.fig)


def test_starts_and_settles_clear(demo):
    _run_until_rest(demo)
    assert demo.x == pytest.approx(_clear_eq(demo.a), abs=EQ_TOL)


def test_hysteresis_and_chemical_treatment(demo):
    _run_until_rest(demo)

    # Push a past the upper fold: the clear basin vanishes, lake goes turbid.
    a_high = UPPER.a + 0.01
    demo.slider_a.set_val(a_high)
    _run_until_rest(demo)
    assert demo.x == pytest.approx(_turbid_eq(a_high), abs=EQ_TOL)

    # Treating the lake while a is still high does not help: it rolls back.
    demo.treat()
    _run_until_rest(demo)
    assert demo.x == pytest.approx(_turbid_eq(a_high), abs=EQ_TOL)

    # Lower a back into the bistable window: the lake stays turbid (hysteresis).
    a_mid = 0.5 * (UPPER.a + LOWER.a)
    demo.slider_a.set_val(a_mid)
    _run_until_rest(demo)
    assert demo.x == pytest.approx(_turbid_eq(a_mid), abs=EQ_TOL)

    # Now treatment works: moving the state into the clear basin sticks.
    demo.treat()
    _run_until_rest(demo)
    assert demo.x == pytest.approx(_clear_eq(a_mid), abs=EQ_TOL)


def test_reset_restores_baseline(demo):
    demo.slider_a.set_val(UPPER.a + 0.01)
    demo.slider_b.set_val(0.7)
    _run_until_rest(demo)
    demo.reset()
    assert demo.a == PARAMS["model"]["a"] and demo.b == B0
    _run_until_rest(demo)
    assert demo.x == pytest.approx(_clear_eq(demo.a), abs=EQ_TOL)


def test_noise_is_reproducible_with_seed():
    """Same seed, same noisy path."""
    paths = []
    for _ in range(2):
        d = _make_demo()
        d.chk_noise.set_active(0)
        assert d.noise_on
        for _ in range(20):
            d.tick()
        paths.append(d.x)
        plt.close(d.fig)
    assert paths[0] == paths[1]
