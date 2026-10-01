"""Select one complete naming tradition per civilization, inherited by its languages."""

import hashlib
import re

from .fantasy import PROFILES as FANTASY
from .historical import PROFILES as HISTORICAL
from .model import NamingProfile
from .realistic import PROFILES as REALISTIC


PROFILES = {profile.key: profile for profile in (*HISTORICAL, *REALISTIC, *FANTASY)}
PROFILE_SETS = {"historical": HISTORICAL, "realistic": REALISTIC, "fantasy": FANTASY}


def profile_lineage(lineage: str, *, seed: int, profile_set: str = "mixed") -> str:
    """Persist the selected profile in the lineage; local branches inherit it."""

    if profile_set not in {"mixed", *PROFILE_SETS}:
        raise ValueError("naming profile set must be mixed, historical, realistic or fantasy")
    base = re.sub(r":profile-[^:]+", "", lineage).split(":branch-", 1)[0]
    explicit = re.search(r"lineage-s(\d+)-", base)
    if explicit is None:
        raise ValueError(f"naming profile requires an explicit sound family: {lineage}")
    style = int(explicit.group(1))
    if not 0 <= style < 12:
        raise ValueError(f"naming sound family is out of range: {lineage}")
    digest = hashlib.sha256(f"naming-profiles-v3:{seed}:{base}".encode("utf-8")).digest()
    selected_set = profile_set
    if selected_set == "mixed":
        selected_set = ("historical", "realistic", "realistic", "fantasy")[digest[0] % 4]
    available = tuple(profile for profile in PROFILE_SETS[selected_set] if profile.style == style)
    selected = available[int.from_bytes(digest[1:5], "big") % len(available)]
    return f"{base}:profile-{selected.key}"


def naming_profile(lineage: str, *, style: int) -> NamingProfile:
    match = re.search(r":profile-([^:]+)", lineage)
    if match is None:
        return REALISTIC[style]
    profile = PROFILES.get(match.group(1))
    if profile is None or profile.style != style:
        raise ValueError(f"invalid naming profile for lineage: {lineage}")
    return profile


__all__ = ["NamingProfile", "PROFILE_SETS", "naming_profile", "profile_lineage"]
