from src.pipelines.conditional_2d import run as run_conditional
from src.pipelines.ot_flow import run as run_ot
from src.pipelines.mnist_flow import run as run_mnist
from src.pipelines.unconditional_2d import run as run_unconditional

PIPELINES = {
    "unconditional_2d": run_unconditional,
    "conditional_2d": run_conditional,
    "ot_flow": run_ot,
    "mnist_flow": run_mnist,
}


def run_pipeline(name, config):
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}. Available: {', '.join(PIPELINES)}")
    return PIPELINES[name](config)
