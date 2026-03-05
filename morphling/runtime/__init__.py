try:
    from .model_emulator import EmulationEngine, InitEmptyModel
except Exception:
    EmulationEngine = None  # type: ignore
    InitEmptyModel = None  # type: ignore
