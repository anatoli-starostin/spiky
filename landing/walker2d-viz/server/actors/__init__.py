"""Actor auto-discovery: scans this package for Actor subclasses and returns {name: class}.

Add a new actor by dropping `actors/my_actor.py` with `class MyActor(Actor): name = "my"; def act(...)`.
"""
import importlib
import inspect
import pkgutil

from .base import Actor

__all__ = ["Actor", "discover_actors"]


def discover_actors():
    """Return {name: Actor subclass} for every module in this package."""
    registry = {}
    for _, modname, _ in pkgutil.iter_modules(__path__):
        if modname == "base":
            continue
        module = importlib.import_module(f"{__name__}.{modname}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Actor) and obj is not Actor and obj.__module__ == module.__name__:
                registry[obj.name] = obj
    return registry
