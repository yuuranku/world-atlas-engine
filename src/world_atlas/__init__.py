"""Public world generation API. No LLM service or project-directory lookup."""
from .api import generate_terrain, generate_world, reproduce_world, verify_world
from .core.procedural_planet import PlanetRecipe

__version__ = "1.1.0"
__all__ = ["PlanetRecipe", "generate_terrain", "generate_world", "reproduce_world", "verify_world"]
