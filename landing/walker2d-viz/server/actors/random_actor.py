"""RandomActor — samples the action space uniformly (a flailing baseline)."""
from .base import Actor


class RandomActor(Actor):
    name = "random"

    def act(self, obs):
        return self.action_space.sample()
