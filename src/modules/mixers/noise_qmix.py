import torch as th

from .qmix import QMixer

class NoiseQMixer(QMixer):

    def forward(self, agent_qs, states, noise):
        states = th.cat([states, noise], dim=-1)
        return super(NoiseQMixer, self).forward(agent_qs, states)
