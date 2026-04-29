from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re


@dataclass
class TargetRef:
    """Resolved target object in the USD scene."""
    class_name: str
    prim_path: str
    source: str
    score: float = 1.0
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["metadata"] is None:
            d["metadata"] = {}
        return d


DEFAULT_ALIASES: Dict[str, List[str]] = {
    "cup": [
        "cup",
        "mug",
        "teacup",
        "tea_cup",
        "coffee_cup",
        "coffee_mug",
        "ceramic_cup",
        "drinkware",
    ],
    "bottle": [
        "bottle",
        "water_bottle",
        "plastic_bottle",
        "drink_bottle",
    ],
    "bowl": [
        "bowl",
        "dish",
        "small_bowl",
    ],
}


def norm_text(s: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(s).lower().replace("-", "_")).strip("_")


def load_json(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Scene registry json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _extract_paths_from_node(node: Any) -> List[str]:
    """Accept multiple registry styles:
    - "cup": "/World/.../cup"
    - "cup": ["/World/.../cup", "/World/.../mug"]
    - "cup": {"path": "..."}
    - "cup": {"paths": ["...", "..."]}
    - [{"class": "cup", "path": "..."}]
    """
    paths: List[str] = []

    if isinstance(node, str):
        paths.append(node)
    elif isinstance(node, list):
        for item in node:
            paths.extend(_extract_paths_from_node(item))
    elif isinstance(node, dict):
        for key in ("path", "prim_path", "usd_path", "target_path"):
            if isinstance(node.get(key), str):
                paths.append(node[key])
        for key in ("paths", "prim_paths", "usd_paths", "candidates"):
            if key in node:
                paths.extend(_extract_paths_from_node(node[key]))
    return [p for p in paths if isinstance(p, str) and p.startswith("/")]


def registry_candidate_paths(registry: Dict[str, Any], target_class: str) -> List[Tuple[str, str]]:
    """Return [(path, source_detail), ...] from a flexible scene registry."""
    target_norm = norm_text(target_class)
    candidates: List[Tuple[str, str]] = []

    # Direct top-level: {"cup": "..."} or {"cup": ["..."]}
    for key, value in registry.items():
        if norm_text(key) == target_norm:
            for p in _extract_paths_from_node(value):
                candidates.append((p, f"registry.top_level.{key}"))

    # Common nested sections.
    for section_name in ("objects", "targets", "classes", "semantic_objects"):
        section = registry.get(section_name)
        if isinstance(section, dict):
            for key, value in section.items():
                if norm_text(key) == target_norm:
                    for p in _extract_paths_from_node(value):
                        candidates.append((p, f"registry.{section_name}.{key}"))
        elif isinstance(section, list):
            for i, item in enumerate(section):
                if not isinstance(item, dict):
                    continue
                labels = []
                for k in ("class", "class_name", "category", "semantic_class", "label", "name", "id"):
                    if k in item:
                        labels.append(item[k])
                if any(norm_text(v) == target_norm for v in labels):
                    for p in _extract_paths_from_node(item):
                        candidates.append((p, f"registry.{section_name}[{i}]"))

    # Also scan lists at root level.
    if isinstance(registry.get("object_list"), list):
        for i, item in enumerate(registry["object_list"]):
            if not isinstance(item, dict):
                continue
            labels = [item.get(k) for k in ("class", "class_name", "category", "semantic_class", "label", "name", "id")]
            if any(norm_text(v) == target_norm for v in labels if v is not None):
                for p in _extract_paths_from_node(item):
                    candidates.append((p, f"registry.object_list[{i}]"))

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for p, src in candidates:
        if p not in seen:
            deduped.append((p, src))
            seen.add(p)
    return deduped


def registry_aliases(registry: Dict[str, Any], target_class: str) -> List[str]:
    aliases = list(DEFAULT_ALIASES.get(norm_text(target_class), []))
    aliases.append(target_class)

    alias_section = registry.get("aliases", {})
    if isinstance(alias_section, dict):
        for key, value in alias_section.items():
            if norm_text(key) == norm_text(target_class):
                aliases.extend(str(x) for x in _as_list(value))

    # Some registry files use {"cup": {"aliases": [...]}}
    for section_name in ("objects", "targets", "classes"):
        section = registry.get(section_name)
        if isinstance(section, dict):
            node = section.get(target_class) or section.get(norm_text(target_class))
            if isinstance(node, dict):
                aliases.extend(str(x) for x in _as_list(node.get("aliases")))

    return sorted({norm_text(a) for a in aliases if a is not None})


def prim_has_mesh_descendant(prim) -> bool:
    from pxr import UsdGeom

    if prim and prim.IsValid() and prim.IsA(UsdGeom.Mesh):
        return True
    if not prim or not prim.IsValid():
        return False
    for child in prim.GetChildren():
        if prim_has_mesh_descendant(child):
            return True
    return False


def _semantic_blob(prim) -> str:
    values: List[str] = []
    try:
        values.append(prim.GetName())
        values.append(str(prim.GetPath()))
    except Exception:
        pass

    try:
        custom = prim.GetCustomData()
        for k, v in custom.items():
            if any(x in k.lower() for x in ("semantic", "class", "category", "label", "name")):
                values.append(str(v))
    except Exception:
        pass

    try:
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if any(x in name.lower() for x in ("semantic", "class", "category", "label", "name")):
                val = attr.Get()
                if val is not None:
                    values.append(str(val))
    except Exception:
        pass

    return norm_text(" ".join(values))


def _score_prim(prim, target_class: str, aliases: Sequence[str]) -> float:
    if not prim or not prim.IsValid():
        return 0.0
    if not prim_has_mesh_descendant(prim):
        return 0.0

    target = norm_text(target_class)
    name = norm_text(prim.GetName())
    path = norm_text(str(prim.GetPath()))
    blob = _semantic_blob(prim)

    score = 0.0
    if name == target:
        score += 100.0
    if target in blob:
        score += 80.0
    if target in path:
        score += 40.0

    for alias in aliases:
        if name == alias:
            score += 90.0
        if alias in blob:
            score += 70.0
        if alias in path:
            score += 35.0

    # Prefer object-level prims over deeply nested mesh leaves when scores tie.
    score -= 0.01 * str(prim.GetPath()).count("/")
    return score


def resolve_target(
    stage,
    *,
    target_class: str,
    scene_registry_json: str | Path | None = None,
    explicit_path: str | None = None,
    allow_name_scan: bool = True,
) -> TargetRef:
    """Resolve an object path for target_class.

    Resolution order:
      1. explicit --target-path
      2. scene registry JSON
      3. semantic/name/alias scan over USD prims
    """
    registry = load_json(scene_registry_json)

    if explicit_path:
        prim = stage.GetPrimAtPath(explicit_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Explicit target path does not exist: {explicit_path}")
        if not prim_has_mesh_descendant(prim):
            raise RuntimeError(f"Explicit target path has no mesh descendant: {explicit_path}")
        return TargetRef(target_class, explicit_path, "explicit_path", 1.0, {"registry": str(scene_registry_json or "")})

    for path, source_detail in registry_candidate_paths(registry, target_class):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim_has_mesh_descendant(prim):
            return TargetRef(target_class, path, source_detail, 1.0, {"registry": str(scene_registry_json or "")})

    if not allow_name_scan:
        raise RuntimeError(
            f"Cannot resolve target_class={target_class!r} from registry={scene_registry_json}. "
            "Provide --target-path or fix the registry mapping."
        )

    aliases = registry_aliases(registry, target_class)
    scored: List[Tuple[float, str]] = []
    for prim in stage.Traverse():
        score = _score_prim(prim, target_class, aliases)
        if score > 0:
            scored.append((score, str(prim.GetPath())))

    scored.sort(key=lambda x: (-x[0], len(x[1]), x[1]))

    if not scored:
        raise RuntimeError(
            f"Cannot resolve target_class={target_class!r}. "
            f"Registry={scene_registry_json} did not contain a valid path, and name scan found nothing."
        )

    top_score, top_path = scored[0]
    return TargetRef(
        class_name=target_class,
        prim_path=top_path,
        source="usd_name_or_semantic_scan",
        score=float(top_score),
        metadata={
            "aliases": aliases,
            "top_candidates": [{"score": float(s), "path": p} for s, p in scored[:10]],
            "registry": str(scene_registry_json or ""),
        },
    )
