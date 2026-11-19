import pytest
from datamate import set_root_context
from omegaconf import OmegaConf

from flyvis.solver import MultiTaskSolver
from flyvis.utils.config_utils import get_default_config
from flyvis_cli.training.train_single import prepare_config


@pytest.fixture(scope="module")
def solver(mock_sintel_data, tmp_path_factory) -> MultiTaskSolver:
    config = get_default_config(
        path="../../flyvis/config/solver.yaml",
        overrides=[
            "task_name=flow",
            "ensemble_and_network_id=0",
            "task.n_iters=50",
            f"+task.dataset.sintel_path={str(mock_sintel_data)}",
            "task.original_split=false",
            "task.dataset.boxfilter.extent=1",
            "task.dataset.n_frames=4",
            "task.dataset.dt=0.041",
            "task.batch_size=2",
            "network.connectome.extent=1",
        ],
    )
    with set_root_context(str(tmp_path_factory.mktemp("tmp"))):
        return MultiTaskSolver("test", config)


def test_solver_config():
    config = get_default_config(
        path="../../flyvis/config/solver.yaml",
        overrides=[
            "task_name=flow",
            "ensemble_and_network_id=0",
        ],
    )
    assert config.backend == "scatter"
    assert config.task_name == "flow"
    assert config.ensemble_and_network_id == 0


@pytest.mark.parametrize("backend", ["csr", None])
def test_solver_network_backend(tmp_path, backend):
    overrides = [
        "task_name=flow",
        "ensemble_and_network_id=0",
        "network.connectome.extent=1",
    ]
    if backend:
        overrides.append(f"backend={backend}")
    args = get_default_config(
        path="../../flyvis/config/solver.yaml", overrides=overrides, as_namespace=False
    )
    config = prepare_config(OmegaConf.create(args))
    if backend is None:
        assert "backend" not in config  # Compatible with saved scatter runs.
    with set_root_context(str(tmp_path)):
        solver = MultiTaskSolver(
            "test",
            config,
            init_decoder=False,
            init_task=False,
            init_optim=False,
            init_penalties=False,
            init_scheduler=False,
        )
    assert solver.network.backend == (backend or "scatter")
    assert "backend" not in solver.config.network
    assert "backend" not in solver.network.config


@pytest.mark.slow
def test_solver_init(solver):
    assert isinstance(solver, MultiTaskSolver)
    assert solver.dir.path.exists()
    assert solver.network
    assert solver.task
    assert solver.decoder
    assert solver.optimizer
    assert solver.penalty
    assert solver.scheduler


def test_solver_overfit(solver):
    solver.train(overfit=True)
    loss = solver.dir.loss[:]
    assert loss[-1] < loss[0]
