import argparse
from src.pipelines.registry import run_unconditional, run_conditional, run_ot


def main():
    p = argparse.ArgumentParser(description="Flow Matching learning project")
    p.add_argument(
        "--pipeline",
        default="unconditional_2d",
        choices=["unconditional_2d", "conditional_2d", "ot_flow"],
    )
    p.add_argument("--steps", type=int)
    p.add_argument("--particles", type=int)
    p.add_argument("--no-animation", action="store_true")
    a = p.parse_args()
    argv = []
    if a.steps is not None:
        argv += ["--steps", str(a.steps)]
    if a.particles is not None:
        argv += ["--particles", str(a.particles)]
    if a.no_animation:
        argv += ["--no-animation"]
    {
        "unconditional_2d": run_unconditional,
        "conditional_2d": run_conditional,
        "ot_flow": run_ot,
    }[a.pipeline](argv)


if __name__ == "__main__":
    main()
