"""One coherent morpheme inventory shared by a civilization's place names."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NamingProfile:
    key: str
    label: str
    style: int
    stems: tuple[str, ...]
    endings: tuple[str, ...]
    polities: tuple[str, ...] = ()
    landmarks: tuple[str, ...] = ()

