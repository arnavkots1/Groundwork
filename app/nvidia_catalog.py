from __future__ import annotations

import json
from pathlib import Path

from rate_limiter import is_model_usable, model_status


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "nvidia_model_catalog.json"
AVAILABLE_PATH = ROOT / "config" / "nvidia_available_models.json"


ROLE_HINTS: dict[str, list[str]] = {
    "CEO": ["nemotron", "glm", "kimi", "minimax", "llama-4", "gpt-oss-120b", "mistral-large", "jamba"],
    "CTO": ["nemotron", "deepseek", "qwen", "codestral", "codegemma", "codellama", "granite", "starcoder"],
    "Product Lead": ["nemotron", "gemma", "mistral", "minimax", "glm", "llama", "jamba"],
    "Lead Designer": ["vision", "multimodal", "vl", "fuyu", "kosmos", "vila", "neva", "phi-4-multimodal"],
    "Lead Engineer": ["deepseek", "qwen", "codestral", "codegemma", "codellama", "granite", "starcoder", "nemotron"],
    "Research Lead": ["kimi", "nemotron", "mistral", "llama", "gemma", "jamba", "yi-large", "dbrx"],
    "Growth/Sales": ["mistral", "gemma", "kimi", "llama", "palmyra", "jamba", "solar"],
    "Critic/Risk": ["guard", "safety", "nemotron", "kimi", "llama", "glm", "gpt-oss"],
    "QA/Release": ["deepseek", "qwen", "guard", "safety", "nemotron", "llama"],
    "Data/Ops": ["embed", "rerank", "granite", "phi", "gemma", "nemotron", "gpt-oss-20b"],
    "Customer Success": ["gemma", "mistral", "llama", "palmyra", "solar", "kimi", "gpt-oss-20b"],
}

NON_CHAT_HINTS = [
    "bge",
    "embed",
    "e5",
    "nv-embed",
    "nemo-retriever",
    "rerank",
    "parse",
    "clip",
    "pii",
    "deplot",
    "fuyu",
    "kosmos",
    "vision",
    "multimodal",
    "-vl",
    "vl-",
    "vila",
    "neva",
    "translate",
    "synthetic-video-detector",
    "calibration",
]

SMART_CHAT_HINTS: list[tuple[str, int]] = [
    ("nemotron-3-ultra", 35),
    ("nemotron-3-super", 32),
    ("glm-5.1", 31),
    ("glm5.1", 31),
    ("kimi-k2-thinking", 30),
    ("deepseek-v4-pro", 29),
    ("minimax-m3", 28),
    ("gpt-oss-120b", 27),
    ("llama-4-maverick", 25),
    ("deepseek-v4-flash", 24),
    ("qwen3", 23),
    ("qwen2.5", 18),
    ("llama-3.3-70b", 18),
    ("llama-3.1-70b", 16),
    ("codellama-70b", 14),
    ("jamba-1.5-large", 13),
    ("dbrx", 12),
    ("gemma-3-27b", 12),
]


SPECIALIZED_TYPE_HINTS: list[tuple[str, str]] = [
    ("embedding", "embed"),
    ("rerank", "rerank"),
    ("document_parse", "parse"),
    ("privacy", "pii"),
    ("vision_language", "vision"),
    ("vision_language", "multimodal"),
    ("vision_language", "-vl"),
    ("vision_language", "vl-"),
    ("vision_language", "fuyu"),
    ("vision_language", "kosmos"),
    ("vision_language", "vila"),
    ("vision_language", "neva"),
    ("vision_language", "deplot"),
    ("safety", "guard"),
    ("safety", "safety"),
    ("safety", "jailbreak"),
    ("translation", "translate"),
    ("video_safety", "synthetic-video-detector"),
    ("specialized", "calibration"),
    ("retrieval", "clip"),
]


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def load_available_model_ids() -> set[str]:
    if not AVAILABLE_PATH.exists():
        return set()
    data = json.loads(AVAILABLE_PATH.read_text(encoding="utf-8"))
    return set(data.get("models", []))


def all_available_models() -> list[dict]:
    catalog_by_id = {}
    for model in all_models():
        catalog_by_id[model["id"]] = model
        catalog_by_id[_normalized_model_id(model["id"])] = model
    rows = []
    for model_id in sorted(load_available_model_ids()):
        catalog_model = catalog_by_id.get(model_id) or catalog_by_id.get(_normalized_model_id(model_id))
        if catalog_model:
            rows.append({**catalog_model, "id": model_id, "availability": "synced"})
        else:
            endpoint_type = classify_model_id(model_id)
            rows.append(
                {
                    "id": model_id,
                    "name": model_id,
                    "endpoint_type": endpoint_type,
                    "roles": inferred_roles_for_model(model_id),
                    "strengths": ["inferred from synced NVIDIA model list"],
                    "priority": 65,
                    "source": "config/nvidia_available_models.json",
                    "availability": "synced",
                }
            )
    return rows


def all_models() -> list[dict]:
    return load_catalog().get("models", [])


def models_for_role(role: str, endpoint_types: set[str] | None = None) -> list[dict]:
    candidates = []
    for model in all_models():
        if role in model.get("roles", []):
            if endpoint_types is None or model.get("endpoint_type") in endpoint_types:
                candidates.append(model)
    return sorted(candidates, key=lambda item: item.get("priority", 0), reverse=True)


def chat_models_for_role(role: str) -> list[dict]:
    return models_for_role(role, {"chat"})


def reviewer_fleet(role: str, limit: int = 4) -> list[dict]:
    seen = set()
    fleet = []
    for model in available_models_for_role(role):
        endpoint_type = model.get("endpoint_type")
        if endpoint_type != "chat":
            continue
        family = model["id"].split("/")[0]
        key = (family, endpoint_type)
        if key in seen:
            continue
        seen.add(key)
        fleet.append(model)
        if len(fleet) >= limit:
            break
    return fleet


def smartest_chat_models(role: str | None = None, limit: int = 10) -> list[dict]:
    """Return a ranked cloud chat fleet for multi-helper council work.

    The score intentionally favors strong general reasoning/coding models and
    skips retrieval, embedding, vision, and guard-only routes unless a role is
    explicitly risk/release oriented.
    """
    role = role or "CEO"
    candidates = []
    for model in all_available_models():
        model_id = model["id"]
        if model.get("endpoint_type") != "chat":
            continue
        if not is_model_usable("NVIDIA NIM", model_id):
            continue
        if _is_specialized_non_chat_name(model_id):
            continue
        if _is_guard_model(model_id) and role not in {"Critic/Risk", "QA/Release"}:
            continue
        candidates.append({**model, "smart_score": _smart_chat_score(model, role)})

    candidates.sort(key=lambda item: (item["smart_score"], item.get("priority", 0), item["id"]), reverse=True)
    fleet = []
    seen_families = set()
    for model in candidates:
        family = _model_family(model["id"])
        if family in seen_families and len(fleet) < max(4, limit // 2):
            continue
        seen_families.add(family)
        fleet.append(model)
        if len(fleet) >= limit:
            return fleet

    seen_ids = {model["id"] for model in fleet}
    for model in candidates:
        if model["id"] in seen_ids:
            continue
        fleet.append(model)
        if len(fleet) >= limit:
            break
    return fleet


def available_models_for_role(role: str) -> list[dict]:
    available_ids = load_available_model_ids()
    catalog_models = models_for_role(role)
    if available_ids:
        catalog_models = [model for model in catalog_models if model["id"] in available_ids]

    seen = {model["id"] for model in catalog_models}
    inferred = []
    for model_id in sorted(available_ids):
        if model_id in seen or not _looks_role_relevant(role, model_id):
            continue
        endpoint_type = "chat" if _looks_chat_model(model_id) else "specialized"
        inferred.append(
            {
                "id": model_id,
                "name": model_id,
                "endpoint_type": endpoint_type,
                "roles": [role],
                "strengths": ["inferred from synced NVIDIA model list"],
                "priority": _inferred_priority(role, model_id),
                "source": "config/nvidia_available_models.json",
            }
        )

    candidates = catalog_models + inferred
    candidates = [model for model in candidates if is_model_usable("NVIDIA NIM", model["id"])]
    return sorted(candidates, key=lambda item: item.get("priority", 0), reverse=True)


def fleet_status(role: str, limit: int = 20) -> list[dict]:
    rows = []
    for model in available_models_for_role(role)[:limit]:
        status = model_status("NVIDIA NIM", model["id"])
        rows.append({**model, "health": status.get("status", "unknown"), "cooldown": status.get("cooldown_remaining_seconds", 0)})
    return rows


def classify_model_id(model_id: str) -> str:
    lower = model_id.lower()
    for endpoint_type, hint in SPECIALIZED_TYPE_HINTS:
        if hint in lower:
            return endpoint_type
    return "chat"


def inferred_roles_for_model(model_id: str) -> list[str]:
    roles = []
    for role in ROLE_HINTS:
        if _looks_role_relevant(role, model_id):
            roles.append(role)
    return roles


def _looks_chat_model(model_id: str) -> bool:
    return classify_model_id(model_id) == "chat"


def _looks_role_relevant(role: str, model_id: str) -> bool:
    lower = model_id.lower()
    if any(hint in lower for hint in ["guard", "safety", "jailbreak"]) and role not in {"Critic/Risk", "QA/Release"}:
        return False
    hints = ROLE_HINTS.get(role, [])
    return any(hint in lower for hint in hints)


def _inferred_priority(role: str, model_id: str) -> int:
    lower = model_id.lower()
    if role == "Lead Engineer" and any(hint in lower for hint in ["deepseek", "qwen", "codestral"]):
        return 82
    if role == "CEO" and any(hint in lower for hint in ["nemotron", "glm", "kimi", "minimax"]):
        return 82
    if "guard" in lower or "safety" in lower:
        return 80
    if "70b" in lower or "120b" in lower or "253b" in lower or "550b" in lower:
        return 78
    return 65


def _smart_chat_score(model: dict, role: str) -> int:
    model_id = model["id"].lower()
    score = int(model.get("priority", 0))
    for hint, bonus in SMART_CHAT_HINTS:
        if hint in model_id:
            score += bonus
    if role in model.get("roles", []):
        score += 12
    for hint in ROLE_HINTS.get(role, []):
        if hint in model_id:
            score += 4
            break
    if any(size in model_id for size in ["550b", "253b", "120b", "70b", "128e"]):
        score += 8
    if any(size in model_id for size in ["2b", "3b", "7b", "8b"]):
        score -= 8
    if _is_guard_model(model_id) and role in {"Critic/Risk", "QA/Release"}:
        score += 10
    return score


def _normalized_model_id(model_id: str) -> str:
    lower = model_id.lower()
    lower = lower.replace("glm5.", "glm-5.")
    lower = lower.replace("glm4.", "glm-4.")
    return lower


def _is_specialized_non_chat_name(model_id: str) -> bool:
    lower = model_id.lower()
    return any(hint in lower for hint in NON_CHAT_HINTS)


def _is_guard_model(model_id: str) -> bool:
    lower = model_id.lower()
    return any(hint in lower for hint in ["guard", "safety", "jailbreak"])


def _model_family(model_id: str) -> str:
    provider, _, name = model_id.partition("/")
    parts = name.split("-")
    if len(parts) >= 2:
        return f"{provider}/{parts[0]}-{parts[1]}"
    return provider or model_id
