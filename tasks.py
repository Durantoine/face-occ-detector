from invoke import task


@task
def clean(c):
    c.run("find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true", warn=True)
    c.run("rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage", warn=True)


@task
def install(c):
    c.run("uv sync --all-extras")


@task
def format(c):
    c.run("uv run ruff format src tests tasks.py", pty=True)


@task
def lint(c, fix=False):
    flag = "--fix" if fix else ""
    c.run(f"uv run ruff check src tests {flag}", pty=True)


@task
def test(c):
    c.run("uv run pytest", pty=True)


@task
def typecheck(c):
    c.run("uv run mypy src tests", pty=True)


@task
def check(c):
    format(c)
    lint(c)
    typecheck(c)
    test(c)


@task
def pretrain(c):
    c.run("uv run python src/pretrain_ibot.py", pty=True)


@task
def train(c):
    c.run("uv run python src/train.py", pty=True)


@task
def optimize(c):
    c.run("uv run python src/optimize.py", pty=True)


@task
def ensemble(c):
    c.run("uv run python src/ensemble_train.py", pty=True)


@task
def predict(c):
    c.run("uv run python src/predict.py", pty=True)


@task
def evaluate(c):
    c.run("uv run python src/evaluate_model.py", pty=True)


@task
def mlflow_ui(c, port=5000, backend="sqlite:///mlflow.db"):
    env = "OMP_NUM_THREADS=1 TMPDIR=$HOME/tmp MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false MLFLOW_DEPLOYMENTS_TARGET=''"
    c.run(
        f"mkdir -p $HOME/tmp && {env} uvx --python 3.12 --from 'mlflow<3' mlflow ui --backend-store-uri {backend} --host 0.0.0.0 --port {port} --workers 1",
        pty=True,
    )


@task
def mlflow_stop(c):
    c.run("pkill -f 'mlflow ui'", warn=True)


@task
def optuna_dashboard(c, port=8080, storage="sqlite:///optuna.db"):
    c.run(f"OMP_NUM_THREADS=1 uvx --python 3.12 --from optuna-dashboard optuna-dashboard {storage} --host 0.0.0.0 --port {port}", pty=True)


@task
def optuna_stop(c):
    c.run("pkill -f 'optuna-dashboard'", warn=True)
