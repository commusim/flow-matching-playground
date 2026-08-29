from src.pipelines.conditional_2d import run as run_conditional
from src.pipelines.ot_flow import run as run_ot
from src.pipelines.mnist_flow import run as run_mnist
from src.pipelines.mnist_conditional_flow import run as run_mnist_conditional
from src.pipelines.unconditional_2d import run as run_unconditional

PIPELINES = {
    "unconditional_2d": run_unconditional,
    "conditional_2d": run_conditional,
    "ot_flow": run_ot,
    "mnist_flow": run_mnist,
    "mnist_conditional_flow": run_mnist_conditional,
    "mnist_conditional_additive": run_mnist_conditional,
    "mnist_conditional_adagn": run_mnist_conditional,
}


def run_pipeline(name, config):
    if name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {name}. Available: {', '.join(PIPELINES)}")
    return PIPELINES[name](config)
