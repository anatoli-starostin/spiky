"""Actor interface. Drop a new file in this folder defining an Actor subclass with a unique `name`
and it will be auto-discovered by the server (see actors/__init__.py)."""
from __future__ import annotations


class Actor:
    """Minimal control-policy interface. Subclass and implement act()."""
    name: str = "base"

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        """obs: np.ndarray observation -> action np.ndarray in action_space."""
        raise NotImplementedError
