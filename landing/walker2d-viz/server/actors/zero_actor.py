"""ZeroActor — outputs zero torque on every joint (stand / go limp)."""
import numpy as np

from .base import Actor


class ZeroActor(Actor):
    name = "zero"

    def act(self, obs):
        return np.zeros(self.action_space.shape, dtype=np.float32)
