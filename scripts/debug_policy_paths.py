"""逐层对比 SB3 官方路径与纯 torch 直读路径，定位差异来源。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402


def main() -> None:
    import torch

    paths.ensure_msgym_on_path()
    paths.ensure_msgym_model_link()
    import gymnasium as gym
    import msgym  # noqa: F401,F811

    from hemirl import policy as pol
    from hemirl.wrappers import official_muscle_norm_wrapper

    ckpt = paths.checkpoint_dir("LocomotionFull")
    env = gym.make(
        "msgym/LocomotionFullEnv-v1",
        skip_frames=10, reset_noise_scale=0.001, qpos_diff_th=0.06, gait_cycles=3, random_init=True,
        reward_dict={"w_qpos": 50, "w_xpos": 50, "w_pelvis": 100, "w_energy": 0.1, "w_healthy": 100},
    )
    env = official_muscle_norm_wrapper()(env)

    stack = pol.load_sb3_stack(ckpt, env, deterministic=True)
    actor_direct = pol.direct_actor_from_checkpoint(ckpt)
    policy = stack.model.policy

    obs, _ = env.reset(seed=3)
    norm = np.asarray(stack.normalize_obs(obs), dtype=np.float32)
    print("norm stats: min", norm.min(), "max", norm.max(), "mean", norm.mean())

    t = torch.as_tensor(norm[None, :], dtype=torch.float32)
    print("policy.features_extractor:", type(policy.features_extractor))
    print("actor.features_extractor:", type(policy.actor.features_extractor))
    with torch.no_grad():
        # --- SB3 路径逐步拆解 ---
        features_sb3 = policy.actor.extract_features(t, policy.actor.features_extractor)
        latent_sb3 = policy.actor.latent_pi(features_sb3)
        mean_sb3 = policy.actor.mu(latent_sb3)
        log_std_sb3 = policy.actor.log_std(latent_sb3)
        act_sb3 = policy.actor(t, deterministic=True)
        act_sb3_stoch = policy.actor(t, deterministic=False)

        # --- 直读路径 ---
        h = actor_direct.latent_pi(t)
        mean_direct = actor_direct.mu(h)
        log_std_direct = actor_direct.log_std(h)
        act_direct = actor_direct(norm, deterministic=True)

    print("\nfeatures equal:", torch.allclose(features_sb3, t))
    print("latent_pi 末端是否有 ReLU（SB3 侧 latent >= 0 处处成立）:", bool((latent_sb3 >= 0).all()))
    print("latent max diff:", float((latent_sb3 - h).abs().max()))
    print("mean  max diff:", float((mean_sb3 - mean_direct).abs().max()))
    print("logstd max diff:", float((log_std_sb3 - log_std_direct).abs().max()))
    print("logstd range:", float(log_std_sb3.min()), float(log_std_sb3.max()))
    print("actor(det) shape:", tuple(act_sb3.shape))
    print("actor(det) vs actor(stoch) max diff:", float((act_sb3 - act_sb3_stoch).abs().max()))
    print("sb3(det) vs direct max diff:", float(np.max(np.abs(act_sb3.numpy() - act_direct))))

    # group 展开是否一致
    groups = pol.load_policy_kwargs(ckpt)["policy_kwargs"]["dynsyn"]
    exp = pol.build_group_expansion(groups)
    gof = exp["group_of_muscle"]
    manual = np.clip(torch.tanh(mean_sb3).numpy()[0][gof], -1, 1)
    print("manual expand vs direct:", float(np.max(np.abs(manual - act_direct))))

    # SB3 内部 DynSynLayer 展开是否等价于 group_of_muscle 索引
    with torch.no_grad():
        expanded = policy.actor.dynsyn_layer.repeat_replace_x(torch.tanh(mean_sb3))
    print("repeat_replace_x vs manual:", float(np.max(np.abs(expanded.numpy()[0] - manual))))

    # 多次调用是否确定性
    with torch.no_grad():
        a1 = policy.actor(t, deterministic=True)
        a2 = policy.actor(t, deterministic=True)
    print("deterministic 重复调用一致:", float((a1 - a2).abs().max()))

    # model.predict 与 policy.actor 是否一致
    pred = stack.model.predict(norm, deterministic=True)[0]
    print("model.predict vs actor(det) max diff:", float(np.max(np.abs(pred - act_sb3.numpy()[0]))))
    pred2 = stack.model.predict(norm, deterministic=True)[0]
    print("model.predict 重复一致:", float(np.max(np.abs(pred - pred2))))

    # model.predict 是否等价于 sampling
    pred_s, _ = stack.model.predict(norm, deterministic=False)
    print("model.predict(det=True) vs actor(stoch) diff:", float(np.max(np.abs(pred - act_sb3_stoch.numpy()[0]))))


if __name__ == "__main__":
    main()
