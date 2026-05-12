# training/seed_callbacks.py
#
# Deterministic seeding for RLlib new-API-stack PPO. Two pieces:
#
# 1) SeedEverythingCallback: seeds torch / numpy / python.random on:
#      * the driver process    (on_algorithm_init)
#      * each env_runner actor (on_environment_created)
#    This makes stochastic action sampling from the Normal policy
#    distribution reproducible across runs.
#
# 2) DeterministicPPOTorchRLModule: subclass of RLlib's default PPO torch
#    RLModule that seeds torch INSIDE setup(), BEFORE super().setup() builds
#    any nn.Linear. That guarantees identical initial NN weights across runs.
#    Seed is read from `model_config["_deterministic_seed"]` (train_ppo.py
#    injects one distinct seed per policy, deterministic function of master).

import os
import random
import numpy as np

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, _ = try_import_torch()


# =============================================================================
# 1) Per-process RNG seeding callback
# =============================================================================

def _seed_everything(seed: int, label: str = "") -> None:
    """Seed torch, numpy, and python random on the CURRENT process."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print(f"[SeedCallback {label} pid={os.getpid()}] "
          f"seeded torch/numpy/random with seed={seed}", flush=True)


class SeedEverythingCallback(RLlibCallback):
    """Seed every Ray actor that runs user code, at every reasonable opportunity.

    Hooks used:
        on_algorithm_init       : runs on driver process at Algorithm setup.
        on_environment_created  : runs on each env_runner actor right after
                                  the env is built but before the first
                                  sample() call.
        on_episode_start        : runs before each episode. Reseeds the env_runner
                                  process's torch RNG deterministically. This is
                                  critical because action sampling (Normal.sample)
                                  uses torch's global RNG, and between episodes
                                  the state drifts in ways that vary across runs
                                  (e.g. timing of Ray messages, memory layout).
                                  By reseeding with a deterministic function of
                                  (base_seed, worker_idx, episode_id), every
                                  episode samples actions from a fresh, pinned
                                  RNG state.
    """

    def on_algorithm_init(self, *, algorithm, **kwargs):
        seed = algorithm.config.seed
        if seed is None:
            return
        _seed_everything(seed, label="driver")

    def on_environment_created(self, *, env_runner, env, env_context, **kwargs):
        base = env_runner.config.seed
        if base is None:
            return
        worker_idx = getattr(env_runner, "worker_index", 0) or 0
        # Different workers get different seeds (so parallel runners don't
        # collect identical rollouts), but each worker's seed is deterministic
        # given the master seed.
        _seed_everything(base + worker_idx, label=f"env_runner w={worker_idx}")
        # Initialize per-env-runner episode counter used by on_episode_start.
        # Stored on env_runner so it persists across on_episode_start calls.
        env_runner._det_episode_counter = 0

    def on_episode_start(self, *, episode, env_runner=None, **kwargs):
        """Reseed torch RNG before every episode, deterministically.

        This closes the last RNG leak: between episodes, torch's RNG state on
        the env_runner can drift non-deterministically (Ray IPC, memory layout
        in a fresh Docker container, etc). By reseeding at the start of every
        episode with a deterministic function of (base_seed, worker_idx, ep_id),
        the first action sampled in every episode is bit-identical across runs.
        """
        if env_runner is None:
            return
        base = env_runner.config.seed
        if base is None:
            return
        worker_idx = getattr(env_runner, "worker_index", 0) or 0
        ep_idx = getattr(env_runner, "_det_episode_counter", 0)
        env_runner._det_episode_counter = ep_idx + 1

        # Episode-specific torch seed. Offset by 1_000_000 so it doesn't collide
        # with per-policy seeds (SEED*1000 + pol_idx ~ 42_000..42_999) or with
        # the process-global seed (SEED + 500_000 ~ 500_042).
        ep_seed = 1_000_000 + base * 10_000 + worker_idx * 1_000 + ep_idx
        if torch is not None:
            torch.manual_seed(ep_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(ep_seed)
        # Only print first few to avoid log spam.
        if ep_idx < 3:
            print(f"[SeedCallback on_episode_start pid={os.getpid()} w={worker_idx} "
                  f"ep={ep_idx}] torch seed={ep_seed}", flush=True)


# =============================================================================
# 2) Deterministic-init PPO RLModule
# =============================================================================
# Since your train_ppo.py uses the default PPO module (no `module_class=`
# argument to `RLModuleSpec`), we wrap that default module here and inject a
# seed call at the top of setup(). The default module builds its network
# inside setup() too (new API stack convention), so seeding here runs BEFORE
# any nn.Linear samples weights.

try:
    # Ray >= 2.40 path
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
        DefaultPPOTorchRLModule,
    )
except ImportError:
    # Older path
    from ray.rllib.algorithms.ppo.ppo_torch_rl_module import (
        PPOTorchRLModule as DefaultPPOTorchRLModule,
    )


class DeterministicPPOTorchRLModule(DefaultPPOTorchRLModule):
    """Same as DefaultPPOTorchRLModule, but seeds torch before weight init.

    Reads seed from `model_config["_deterministic_seed"]`. Falls back to 0
    if unset (still reproducible, but all policies would share init -- prefer
    setting a distinct seed per policy in train_ppo.py).

    Also performs a ONE-TIME seeding of the process-global torch RNG. This
    is crucial for the Learner actor: RLlib's callback API only exposes hooks
    on the Algorithm (driver) and EnvRunner actors, NOT the Learner. So the
    only code that runs on the Learner actor which we can reliably hook is
    the RLModule's setup() method. We use that to seed the Learner's global
    RNG exactly once, deterministically, regardless of which policy's setup()
    runs first (the order is non-deterministic across runs).
    """

    # Class-level flag, per process. Guarantees we seed the global RNG only
    # once per actor process, regardless of how many RLModules are built in
    # that process or in which order.
    _process_global_rng_seeded = False

    @override(DefaultPPOTorchRLModule)
    def setup(self):
        # Get our custom key from the model config. On the new API stack,
        # RLlib merges the user's model_config dict into DefaultModelConfig's
        # defaults, so self.model_config is a dict containing our key.
        seed = 0
        mc = self.model_config
        if isinstance(mc, dict):
            seed = int(mc.get("_deterministic_seed", 0))
        elif hasattr(mc, "_deterministic_seed"):
            seed = int(getattr(mc, "_deterministic_seed", 0))

        # ONE-TIME per-process seed of the GLOBAL torch RNG.
        # This seeds the Learner actor (and any other process that builds an
        # RLModule for the first time). We derive the global seed from the
        # FLOOR of the per-policy seed to a "master" seed, so all policies
        # agree on the same global seed regardless of which one runs first:
        #   per_policy_seed = SEED*1000 + pol_idx  (from train_ppo.py)
        #   master_seed     = per_policy_seed // 1000 = SEED
        # We then use `master_seed + 500000` for the process-global RNG so it
        # doesn't collide with any per-policy seed.
        if torch is not None and not DeterministicPPOTorchRLModule._process_global_rng_seeded:
            master_seed = seed // 1000  # recover the master SEED
            process_global_seed = master_seed + 500_000
            torch.manual_seed(process_global_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(process_global_seed)
            np.random.seed(process_global_seed)
            random.seed(process_global_seed)
            DeterministicPPOTorchRLModule._process_global_rng_seeded = True
            print(f"[DetPPO pid={os.getpid()}] ONE-TIME process-global RNG "
                  f"seed={process_global_seed} (from master={master_seed})",
                  flush=True)

        # CRITICAL: build the network inside `torch.random.fork_rng` so that
        # weight-init consumes a LOCAL, isolated RNG stream. Without this, the
        # 6 RLModules built in the same process would each consume some of the
        # process-global torch RNG state, and their call order (which is not
        # deterministic when RLlib builds them in parallel) would leave the
        # RNG in a run-dependent state. When the first rollout then samples
        # an action, that sample depends on the post-setup state -> rollouts
        # diverge across runs.
        #
        # fork_rng() saves the current global RNG, lets us reseed + build the
        # layers from a pinned state, and then restores the global state to
        # what it was before setup(). This makes setup() order-independent
        # w.r.t. the global RNG.
        if torch is not None:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                print(f"[DetPPO pid={os.getpid()}] setup() with weight-init "
                      f"seed={seed} (RNG forked)", flush=True)
                super().setup()
        else:
            super().setup()