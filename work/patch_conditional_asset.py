from pathlib import Path
path=Path(r'C:\Code\SIQA\ImageTest\scripts\build_tutorial_assets.py')
text=path.read_text(encoding='utf-8')
if 'from src.modules.velocity import ConditionalVelocityMLP' not in text:
    text=text.replace('from src.modules.trajectory_model_loader import (','from src.modules.velocity import ConditionalVelocityMLP\nfrom src.modules.trajectory_model_loader import (')
old='''    conditional = LegacyConditional()
    conditional.load_state_dict(
        torch.load(
            ROOT / "outputs/conditional_2d/conditional_checkpoint.pt",
            map_location="cpu",
        )
    )'''
new='''    conditional = ConditionalVelocityMLP(hidden=128)
    conditional.load_state_dict(
        torch.load(
            ROOT / "outputs/conditional_2d/20260830_135438_seed42/checkpoint.pt",
            map_location="cpu",
        )
    )'''
if old not in text:
    raise SystemExit('old block missing')
path.write_text(text.replace(old,new),encoding='utf-8')
