import numbers
from typing import Any, Dict, Optional


_wandb = None
_run = None
_enabled = False


def _load_wandb():
    global _wandb
    if _wandb is not None:
        return _wandb
    try:
        import wandb
    except Exception as exc:
        print(f"[wandb] disabled: failed to import wandb ({exc})")
        return None
    _wandb = wandb
    return _wandb


def _clean_config_value(value: Any):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, (list, tuple)):
        cleaned = []
        for item in value:
            item_clean = _clean_config_value(item)
            cleaned.append(str(item) if item_clean is None else item_clean)
        return cleaned
    if isinstance(value, dict):
        return {
            str(k): _clean_config_value(v)
            for k, v in value.items()
            if _clean_config_value(v) is not None
        }
    return None


def _build_config(args) -> Dict[str, Any]:
    skip_keys = {"model", "model_factory", "head"}
    config = {}
    for key, value in vars(args).items():
        if key in skip_keys:
            continue
        clean_value = _clean_config_value(value)
        if clean_value is not None:
            config[key] = clean_value
    return config


def _default_run_name(args, run_index: int) -> str:
    model_name = getattr(args, "model_name", getattr(args, "model", "model"))
    lr = getattr(args, "local_learning_rate", "lr")
    seed = getattr(args, "random_seed", "seed")
    return (
        f"{getattr(args, 'algorithm', 'algo')}_"
        f"{getattr(args, 'dataset', 'data')}_"
        f"{model_name}_lr{lr}_seed{seed}_run{run_index}"
    )


def init_wandb(args, run_index: int = 0):
    global _run, _enabled
    if not bool(getattr(args, "use_wandb", False)):
        _enabled = False
        _run = None
        return None

    wandb = _load_wandb()
    if wandb is None:
        _enabled = False
        _run = None
        return None

    tags = [
        tag.strip()
        for tag in str(getattr(args, "wandb_tags", "")).split(",")
        if tag.strip()
    ]
    entity = getattr(args, "wandb_entity", "")
    group = getattr(args, "wandb_group", "")
    name = getattr(args, "wandb_name", "") or _default_run_name(args, run_index)

    try:
        _run = wandb.init(
            project=getattr(args, "wandb_project", "FedSTAR"),
            entity=entity or None,
            group=group or None,
            name=name,
            mode=getattr(args, "wandb_mode", "online"),
            tags=tags,
            config=_build_config(args),
            reinit=True,
        )
        _enabled = True
        run_url = getattr(_run, "url", None)
        if run_url:
            print(f"[wandb] logging to: {run_url}")
        else:
            print("[wandb] logging enabled")
        return _run
    except Exception as exc:
        print(f"[wandb] disabled: wandb.init failed ({exc})")
        _enabled = False
        _run = None
        return None


def wandb_log(metrics: Dict[str, Any], step: Optional[int] = None):
    if not _enabled or _run is None:
        return
    clean_metrics = {}
    for key, value in metrics.items():
        clean_value = _clean_config_value(value)
        if clean_value is not None:
            clean_metrics[key] = clean_value
    if not clean_metrics:
        return
    try:
        _run.log(clean_metrics, step=step)
    except Exception as exc:
        print(f"[wandb] warning: log failed ({exc})")


def finish_wandb():
    global _run, _enabled
    if not _enabled or _run is None:
        return
    try:
        _run.finish()
    except Exception as exc:
        print(f"[wandb] warning: finish failed ({exc})")
    finally:
        _run = None
        _enabled = False
