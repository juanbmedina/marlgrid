import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override


class EnergyRLModule(TorchRLModule):
    """A simple VPG (vanilla policy gradient)-style RLModule for testing purposes.

    Use this as a minimum, bare-bones example implementation of a custom TorchRLModule.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        model_config,
        **kwargs,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        hidden = model_config.get("fcnet_hiddens", [256, 256])

        # === Policy network ===
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h

        self.policy_net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(last, act_dim)

        # Log std (learned, PPO-style)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # === Value network ===
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], 1),
        )


    @override(TorchRLModule)
    def _forward_train(self, batch):
        obs = batch[Columns.OBS]

        features = self.policy_net(obs)
        mean = self.mean_layer(features)
        log_std = self.log_std.expand_as(mean)

        values = self.value_net(obs).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: torch.cat([mean, log_std], dim=-1),
            Columns.VF_PREDS: values,
        }
    
    @override(TorchRLModule)
    def _forward_inference(self, batch):
        with torch.no_grad():
            obs = batch[Columns.OBS]
            features = self.policy_net(obs)

            mean = self.mean_layer(features)
            log_std = self.log_std.expand_as(mean)

            return {
                Columns.ACTION_DIST_INPUTS: torch.cat([mean, log_std], dim=-1)
            }
        
    @override(TorchRLModule)
    def forward_exploration(self, batch):
        with torch.no_grad():
            return self._forward_inference(batch)