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
        f"mkdir -p $HOME/tmp && {env} uvx --python 3.12 --from mlflow mlflow server --backend-store-uri {backend} --host 0.0.0.0 --port {port} --workers 1",
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


_IMAGE_BASE_DIR = (
    "/home/matt/Programmation/704_IADATA_ML_avance/datachallenge"
    "/DataChallenge2026/occlusion_datasets/raw/Crop_224_5fp_100K"
)


@task
def zero_shot(
    c,
    n_samples=500,
    seed=42,
    csv="data/train.csv",
    image_base_dir=_IMAGE_BASE_DIR,
    output_dir="results/zero_shot",
    sam2_model="facebook/sam2.1-hiera-large",
    qwen_model="Qwen/Qwen2.5-VL-7B-Instruct",
):
    """Zero-shot evaluation: SAM2 + Qwen2.5-VL on a stratified sample of the dataset.

    Examples:
        inv zero-shot                        # 500 stratified samples
        inv zero-shot --n-samples=100        # quick sanity check
        inv zero-shot --n-samples=2000 --seed=7
    """
    c.run(
        f"uv run python src/zero_shot_eval.py "
        f"--csv {csv} "
        f"--image-base-dir {image_base_dir} "
        f"--n-samples {n_samples} "
        f"--seed {seed} "
        f"--output-dir {output_dir} "
        f"--sam2-model {sam2_model} "
        f"--qwen-model {qwen_model}",
        pty=True,
    )


@task
def visualize_sam2(
    c,
    predictions_csv,
    output_dir="results/sam2_failures",
    max_per_cat=10,
    sam2_model="facebook/sam2.1-hiera-large",
    sam2_device="cuda",
):
    """Visualize SAM2 failure modes from a predictions CSV.

    Example:
        inv visualize-sam2 --predictions-csv results/zero_shot/predictions_XYZ.csv
        inv visualize-sam2 --predictions-csv results/zero_shot/predictions_XYZ.csv --max-per-cat=5
    """
    c.run(
        f"uv run python src/visualize_sam2_failures.py "
        f"--predictions-csv {predictions_csv} "
        f"--output-dir {output_dir} "
        f"--max-per-cat {max_per_cat} "
        f"--sam2-model {sam2_model} "
        f"--sam2-device {sam2_device}",
        pty=True,
    )
