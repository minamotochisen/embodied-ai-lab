# embodied-ai-lab · SmolVLA 端侧全链路验证

> 在单张 RTX 4070 Ti SUPER(16GB)+ Windows 10 上验证一件事:**VLA 模型不需要数据中心**——推理、SFT 微调、仿真闭环三步全部跑通,本文所有数字均为本机实测,非论文转述。

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10_cu126-EE4C2C?logo=pytorch&logoColor=white)
![LeRobot](https://img.shields.io/badge/LeRobot-0.4.4-F59E0B)
![GPU](https://img.shields.io/badge/GPU-RTX_4070_Ti_SUPER_16GB-76B900?logo=nvidia&logoColor=white)

## 结果速览

| 环节 | 模型 / 数据 | 关键实测 |
|---|---|---|
| ① 推理验证 | SmolVLA base(450M,SmolVLM2 骨干) | 50 步动作块生成 **176ms**(≈284 步/s);显存常驻 883MB,**峰值 931MB** |
| ② SFT 微调 | `svla_so101_pickplace`(50 episodes / 11939 帧) | 150 步 loss **0.146 → 0.060**,墙钟 **55 秒**(2.72 step/s);可训练参数 100M / 450M |
| ③ 仿真闭环 | `diffusion_pusht` @ PushT-v0 | 3 episodes 平均累计奖励 **124.2**,平均最大奖励 0.857,逐条回放见 [`docs/eval_videos/`](docs/eval_videos/) |

微调 loss 曲线(batch 8,每 10 步记录,原始日志见 [`docs/train_log.txt`](docs/train_log.txt)):

| step | 10 | 30 | 50 | 70 | 100 | 120 | 150 |
|---|---|---|---|---|---|---|---|
| loss | 0.146 | 0.106 | 0.092 | 0.069 | 0.055 | 0.064 | **0.060** |

## 这套验证说明了什么

- **消费级显卡足以承载 VLA 推理与微调**:450M 策略推理峰值不足 1GB 显存,150 步微调不到 1 分钟——"端侧 VLA"不是营销话术;
- **异步推理有数据支撑**:单个 action chunk 仅 176ms,环境可以每 50 步才调用一次模型("低频推理 + 高频执行"),这正是 SmolVLA 论文异步部署方案的基础;
- **动作专家小而快**:默认训练预设冻结视觉编码器(`freeze_vision_encoder=True`),450M 总参数中仅 100M 参与训练;
- **闭环 ≠ 单次推理**:PushT rollout 补齐"观测 → 策略 → 动作 → 环境反馈"完整循环,顺带解决旧版 checkpoint 迁移、离线环境补丁等真实工程问题。

## 快速开始

### 环境搭建(Windows 正确顺序)

```bash
conda create -n embodied python=3.10 -y && conda activate embodied

# ① 先装 CUDA 版 torch(Windows 下 pip 默认装 CPU 版!)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ② 装 lerobot —— 注意:它会无视已装的 cu126 torch,降级覆盖为 CPU 版!
pip install lerobot
#    → 必须再重装一遍 CUDA 版(版本需落在 lerobot 约束内,torch<2.11):
pip install "torch==2.10.0+cu126" "torchvision==0.25.0+cu126" --index-url https://download.pytorch.org/whl/cu126

# ③ SmolVLA 额外依赖
pip install "transformers<5" peft num2words gym-pusht "pymunk<7"

# ④ HuggingFace 镜像 + 缓存目录(可选)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=D:/具身智能-lab/hf_cache
```

### 三步复现

```bash
# ① 推理验证:模拟观测(3×256×256 图像 + 6 维状态 + 语言指令)→ 合法动作块,无真机依赖
python scripts/01_smolvla_inference_test.py

# ② SFT 微调:社区 SO-101 拾放数据集,55 秒跑完 150 步
lerobot-train \
  --policy.path="D:/具身智能-lab/models/smolvla_base" \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --rename_map='{"observation.images.side": "observation.images.camera1", "observation.images.up": "observation.images.camera2"}' \
  --output_dir=D:/具身智能-lab/outputs/smolvla_ft2 \
  --batch_size=8 --steps=150 --save_freq=75 --num_workers=0 --log_freq=10 --wandb.enable=false

# ③ PushT 仿真闭环:预训练 Diffusion Policy 回放验证
python scripts/02_migrate_pusht.py   # 旧版 checkpoint → processor 格式(含离线补丁)
lerobot-eval --env.type=pusht --env.task=PushT-v0 \
  --policy.path="D:/具身智能-lab/models/diffusion_pusht_migrated" \
  --eval.n_episodes=3 --eval.batch_size=1 --policy.push_to_hub=false
```

> 模型权重与训练产出(`models/`、`hf_cache/`、`outputs/`)体积大,不入库;按上述命令重新下载/训练即可,微调 checkpoint 约 1.8GB/个。

## 仓库结构

```
├── scripts/
│   ├── 01_smolvla_inference_test.py   # 推理验证:加载→processor→select_action→chunk→延迟/显存实测(104 行)
│   └── 02_migrate_pusht.py            # 旧版 checkpoint 迁移 + ModelCard 联网校验的离线绕过
├── docs/
│   ├── eval_videos/                   # PushT 闭环 rollout 回放(3 episodes,逐条成败可查)
│   ├── eval_info.json                 # 逐 episode 奖励/成功率原始评测数据
│   └── train_log.txt                  # 微调完整控制台日志(150 步 / 55s / 2.72 step/s)
├── models/  hf_cache/  outputs/       # 大文件,.gitignore 已排除,按命令重建
└── README.md
```

## Windows 踩坑速查(全踩过,省你半天)

| 坑 | 现象 | 解法 |
|---|---|---|
| lerobot 依赖解析降级 torch | `torch.cuda.is_available()` 变 False | 装完 lerobot 后重装 cu126 版 torch |
| `--policy.path=lerobot/smolvla_base` | "Repo id 非法"(Windows 反斜杠路径) | 先 `snapshot_download` 到本地目录,再用本地路径加载 |
| 数据集相机名不匹配 | `side`/`up` vs 基础模型 `camera1/2/3` | `--rename_map` 重命名,缺失相机自动置空不阻塞 |
| gym-pusht 装不上/报错 | `pymunk` 7.x 删除了 `add_collision_handler` | 锁 `pymunk<7`;环境注册名是 `PushT-v0` |
| 旧 checkpoint 无法加载 | ModelCard 校验硬编码 huggingface.co,离线卡死 | 迁移脚本置空 `ModelCard.validate`(见 `02_migrate_pusht.py`) |

## 路线图

- [ ] SO-100/SO-101 真机闭环:遥操作采集自己的数据 → 同款命令微调 → 部署回真机(待硬件,千元级预算)
- [ ] LIBERO / RoboTwin 上对比复现 OpenVLA-OFT(SFT)与 SimpleVLA-RL(RL 后训练),需 WSL2/Linux

## 相关项目

- [aigc-ecommerce-pipeline](https://github.com/minamotochisen/aigc-ecommerce-pipeline) —— 同一张 4070 Ti SUPER 上的另一个完整生产管线(AIGC 电商商拍)
- 配套中文知识库:18 篇 VLA / 世界模型前沿论文深度解读(私有仓库)
