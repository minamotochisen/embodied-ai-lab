# -*- coding: utf-8 -*-
"""迁移 diffusion_pusht 旧版 checkpoint 到 processor 格式(绕过 ModelCard 联网校验)"""
import sys

import huggingface_hub.repocard as rc

rc.ModelCard.validate = lambda self, *a, **k: None  # 硬编码 huggingface.co,离线环境置空

sys.argv = [
    "migrate",
    "--pretrained-path", "D:/具身智能-lab/models/diffusion_pusht",
]
from lerobot.processor.migrate_policy_normalization import main
main()
