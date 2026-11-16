"""Model registry; importing this module never imports a model framework."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Backend:
    """Describe one model and the independent environment that runs it."""

    id: str
    label: str
    environment: str
    distribution: str
    default_model: str

    @property
    def manifest(self):
        return Path(__file__).parent / "environments" / self.environment / "pixi.toml"

    @property
    def python(self):
        override = os.environ.get(f"INSPIRED_{self.environment.upper()}_PYTHON")
        if override:
            return Path(override).expanduser().resolve()
        return self.manifest.parent / ".pixi" / "envs" / "default" / "bin" / "python"

    @property
    def available(self):
        return self.python.is_file() and os.access(self.python, os.X_OK)

    @property
    def setup_hint(self):
        return f"pixi install --manifest-path {self.manifest}"


BACKENDS = (
    Backend("mattersim", "MatterSim", "mattersim", "mattersim", "MatterSim-v1.0.0-5M.pth"),
    Backend("sevennet", "SevenNet", "sevennet", "sevenn", "7net-mf-ompa"),
    Backend("orb", "ORB v3", "orb", "orb-models", "orb_v3_conservative_inf_omat"),
    Backend("mace", "MACE", "mace", "mace-torch", "medium-mpa-0"),
    Backend("mace-off", "MACE-OFF", "mace", "mace-torch", "medium"),
)


def get_backend(identifier):
    """Resolve a stable model identifier instead of depending on UI indices."""
    for backend in BACKENDS:
        if backend.id == identifier:
            return backend
    raise ValueError(f"Unknown MLFF backend: {identifier}")


def create_calculator(identifier, model, device):
    """Load only the selected calculator, inside the worker process."""
    backend = get_backend(identifier)
    selected_model = model or backend.default_model
    if identifier == "mattersim":
        from mattersim.forcefield.potential import MatterSimCalculator

        return MatterSimCalculator(load_path=selected_model, device=device)
    if identifier == "sevennet":
        from sevenn.calculator import SevenNetCalculator

        options = {} if model else {"modal": "mpa"}
        return SevenNetCalculator(model=selected_model, device=device, **options)
    if identifier == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        if model:
            raise ValueError("Custom checkpoints for ORB are not supported by this adapter yet.")
        potential = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high", compile=False)
        return ORBCalculator(potential, device=device)
    if identifier == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model=selected_model, device=device, default_dtype="float64")
    if identifier == "mace-off":
        from mace.calculators import mace_off

        return mace_off(model=selected_model, device=device, default_dtype="float64")
    raise ValueError(f"No calculator adapter for {identifier}")
