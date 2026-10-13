# -*- coding: utf-8 -*-
"""SmolVLA 推理验证脚本(知识库实操路线第①步)

用模拟观测(随机图像 + 随机状态 + 语言指令)验证 lerobot/smolvla_base
预训练策略能否在本机 GPU 上加载并输出合法动作块。
不依赖实体机械臂。
"""

import time

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

REPO = "lerobot/smolvla_base"
DEVICE = "cuda"
TASK = "pick up the yellow cube and place it into the bin"


def build_dummy_obs(device, seed=0):
    """构造与 SO-100/101 相机配置一致的模拟观测(3 相机 + 6 维状态)。"""
    g = torch.Generator().manual_seed(seed)
    return {
        "observation.state": (torch.rand(1, 6, generator=g) * 2 - 1).to(device),
        "observation.images.camera1": torch.rand(1, 3, 256, 256, generator=g).to(device),
        "observation.images.camera2": torch.rand(1, 3, 256, 256, generator=g).to(device),
        "observation.images.camera3": torch.rand(1, 3, 256, 256, generator=g).to(device),
        "task": TASK,
        "robot_type": "",
    }


def gpu_mem_mb():
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "peak": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    assert torch.cuda.is_available(), "CUDA 不可用"
    torch.cuda.reset_peak_memory_stats()

    # 1. 加载预训练策略
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(REPO)
    policy.to(DEVICE).eval()
    load_s = time.time() - t0
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"[1] 模型加载完成:{load_s:.1f}s | 参数量 {n_params:.1f}M | VLM 骨干 = {policy.config.vlm_model_name}")
    mem = gpu_mem_mb()
    print(f"    显存:已分配 {mem['allocated']:.0f}MB / 保留 {mem['reserved']:.0f}MB")

    # 2. 加载 hub 上保存的 pre/post processor(含归一化统计)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=REPO)
    print("[2] pre/post processor 加载完成(含数据集归一化统计)")

    # 3. 单步动作推理(select_action:队列语义,每次取 1 步)
    obs = build_dummy_obs(DEVICE)
    policy.reset()
    with torch.inference_mode():
        batch = preprocessor(deepcopy_obs(obs))
        t1 = time.time()
        action = policy.select_action(batch)
        first_infer_s = time.time() - t1
        action = postprocessor(action)
    print(f"[3] select_action 输出 shape = {tuple(action.shape)}(首次含完整 chunk 生成 {first_infer_s:.2f}s)")
    print(f"    动作示例(前 6 维):{[round(v, 4) for v in action[0].tolist()]}")
    assert torch.isfinite(action).all(), "输出动作包含 NaN/Inf!"

    # 4. 完整动作块推理(predict_action_chunk:一次输出整个 chunk)
    policy.reset()
    with torch.inference_mode():
        batch = preprocessor(deepcopy_obs(obs))
        chunk = policy.predict_action_chunk(batch)
    print(f"[4] predict_action_chunk 输出 shape = {tuple(chunk.shape)}(chunk_size={policy.config.chunk_size})")
    assert torch.isfinite(chunk).all(), "动作块包含 NaN/Inf!"

    # 5. 速度测试:连续生成 3 个 chunk 取平均
    latencies = []
    for i in range(3):
        policy.reset()
        with torch.inference_mode():
            batch = preprocessor(build_dummy_obs(DEVICE, seed=i))
            torch.cuda.synchronize()
            t2 = time.time()
            policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            latencies.append(time.time() - t2)
    avg = sum(latencies) / len(latencies)
    mem = gpu_mem_mb()
    print(f"[5] 动作块推理延迟:{[f'{x*1000:.0f}ms' for x in latencies]} | 平均 {avg*1000:.0f}ms/50 步动作块")
    print(f"    推理吞吐 ≈ {policy.config.chunk_size / avg:.0f} 步/秒 | 显存峰值 {mem['peak']:.0f}MB")
    print("\n✅ SmolVLA 推理验证通过:模型加载、语言条件输入、动作块输出均正常")


def deepcopy_obs(obs):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}


if __name__ == "__main__":
    main()
