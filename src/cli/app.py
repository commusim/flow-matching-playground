import argparse
from src.pipelines.registry import PIPELINES, run_pipeline
from src.utils.common import load_config


def build_parser():
    parser = argparse.ArgumentParser(description="Flow Matching learning project")
    parser.add_argument("--pipeline", choices=sorted(PIPELINES))
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-root")
    parser.add_argument("--ode-steps", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--checkpoint-path")
    parser.add_argument(
        "--model-variant",
        choices=["auto", "unconditional", "conditional_additive", "conditional_adagn"],
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    for key in (
        "steps",
        "particles",
        "batch_size",
        "seed",
        "device",
        "output_root",
        "ode_steps",
        "hidden",
        "lr",
        "subset_size",
        "samples_per_class",
        "checkpoint_path",
        "model_variant",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.no_animation:
        config["animation"] = False
    pipeline = args.pipeline or config.get("pipeline", "unconditional_2d")
    config["pipeline"] = pipeline
    print("Resolved config:", config)
    run_pipeline(pipeline, config)
