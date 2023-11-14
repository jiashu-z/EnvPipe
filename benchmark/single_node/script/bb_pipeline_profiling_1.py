import json
import os
import itertools
import time

NCCL_BUFFSIZE = 33554432
NCCL_IB_DISABLE = 1
NCCL_P2P_DISABLE = 1

batch_size_list = [2, 4]
global_batch_size_list = [8, 16, 32]

dp_size_list = [1, 2]
pp_size_list = [2, 4]

SEQ_LEN = 1024

gpt_config_list = [
    dict(
        name="small",
        MODEL_SIZE=0.125,
        NUM_LAYERS=12,
        HIDDEN_SIZE=768,
        NUM_ATTN_HEADS=12,
        LR=6.0e-4,
        MIN_LR=6.0e-5,
    ),
    dict(
        name="medium",
        MODEL_SIZE=0.35,
        NUM_LAYERS=24,
        HIDDEN_SIZE=1024,
        NUM_ATTN_HEADS=16,
        LR=3.0e-4,
        MIN_LR=3.0e-5,
    ),
    dict(
        name="large",
        MODEL_SIZE=0.76,
        NUM_LAYERS=24,
        HIDDEN_SIZE=1536,
        NUM_ATTN_HEADS=16,
        LR=2.5e-4,
        MIN_LR=2.5e-5,
    ),
    dict(
        name="xlarge",
        MODEL_SIZE=1.3,
        NUM_LAYERS=24,
        HIDDEN_SIZE=2048,
        NUM_ATTN_HEADS=16,
        LR=2.0e-4,
        MIN_LR=2.0e-5,
    ),
]

NUM_GPUS = 4
STEPS = 32
PARTITIONS = "-"

deepspeed_config_template_path = "config/bb_template.json"

os.environ["NCCL_BUFFSIZE"] = "33554432"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

for config, dp, pp, batch_size, global_batch_size in itertools.product(
    gpt_config_list, dp_size_list, pp_size_list, batch_size_list, global_batch_size_list
):
    if NUM_GPUS % (dp * pp) != 0:
        continue
    config_name = config["name"]
    if config_name in ("large", "xlarge") and batch_size > 2:
        continue
    name = f"gpt{config_name}_dp{dp}_pp{pp}_batch{batch_size}_globalbatch{global_batch_size}"
    print(f'\n\n\n\n\nRUNNING EXPERIMENT')
    num_layers = config["NUM_LAYERS"]
    hidden_size = config["HIDDEN_SIZE"]
    num_attn_heads = config["NUM_ATTN_HEADS"]
    train_options = f" \
        --steps ${STEPS} \
        --backend nccl \
        --dp ${dp} \
        --pp ${pp} \
        -N ${num_layers} \
        -dm ${hidden_size} \
        -H ${num_attn_heads} \
        --seq ${SEQ_LEN} \
        --parts ${PARTITIONS} \
        --log ${name}"
    with open(deepspeed_config_template_path) as f:
        j = json.load(f)
        j["train_micro_batch_size_per_gpu"] = global_batch_size
        j["train_batch_size"] = batch_size
        with open(f"config/{name}.json", "w") as fout:
            json.dump(j, fout, indent=True)
            fout.flush()
    os.system(
        f"deepspeed ../model/gpt.py ${train_options} --deepspeed_config config/${name}.json"
    )
    print(f'FINISHING EXPERIMENT\n\n')
    time.sleep(60)
