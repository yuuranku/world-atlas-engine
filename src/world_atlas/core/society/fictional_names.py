"""Seeded country-name lexicon independent of the existing atlas setting."""

from __future__ import annotations

import hashlib
from typing import Collection

from .model import NameLexicon
from .naming_profiles import profile_lineage
from .onomastics import lineage_key, lineage_roots


_ROLES = (
    "regional",
    "coastal",
    "highland",
    "river",
    "steppe",
    "regional",
    "coastal",
    "river",
)


def procedural_name_lexicon(
    seed: int,
    *,
    forbidden: Collection[str] = (),
    profile_set: str = "mixed",
) -> NameLexicon:
    """Build country identities from fourteen repeatable naming languages."""

    blocked = {str(item).strip() for item in forbidden if str(item).strip()}
    family_buckets: list[tuple[str, ...]] = []
    for family_identifier in range(1, 15):
        style = (family_identifier * 5 + seed) % 12
        lineage = profile_lineage(lineage_key(family_identifier, style_index=style),
                                  seed=seed, profile_set=profile_set)
        roots = lineage_roots(lineage, count=128)
        digest = hashlib.sha256(f"{seed}:{family_identifier}".encode("utf-8")).digest()
        start = int.from_bytes(digest[:4], "big") % len(roots)
        ordered = roots[start:] + roots[:start]
        bucket = tuple(
            dict.fromkeys(
                root.strip()
                for root in ordered
                if 2 <= len(root.strip()) <= 7 and root.strip() not in blocked
            )
        )
        family_buckets.append(bucket)
    candidates: list[str] = []
    seen = set(blocked)
    for rank in range(max(map(len, family_buckets))):
        for bucket in family_buckets:
            if rank >= len(bucket):
                continue
            name = bucket[rank]
            if name in seen:
                continue
            seen.add(name)
            candidates.append(name)
    required = 19 + 160
    if len(candidates) < required:
        raise ValueError("procedural naming languages did not provide enough unique names")
    major = tuple(candidates[:19])
    minor = tuple(candidates[19:required])
    roles = tuple(_ROLES[index % len(_ROLES)] for index in range(len(major)))
    roots = tuple(candidates[required:required + 128])
    if len(roots) < 128:
        roots = tuple(candidates[-128:])
    return NameLexicon(
        major_country_names=major,
        major_country_roles=roles,
        minor_country_names=minor,
        place_roots=roots,
    )


__all__ = ["procedural_name_lexicon"]
