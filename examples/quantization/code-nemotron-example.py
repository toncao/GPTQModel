from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

import os 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "nvidia/OpenCodeReasoning-Nemotron-1.1-32B"
quant_path = "./OpenCodeReasoning-Nemotron-1.1-32B-GPTQ-4bit"

calibration_dataset = load_dataset(
    "nvidia/OpenCodeReasoning",
    "split_0",
    split="split_0"
  ).shuffle(seed=42)["output"][:1024]
#calibration_dataset = [conversation[-1]["content"] for conversation in calibration_dataset]

quant_config = QuantizeConfig(bits=4, group_size=32)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=4)

model.save(quant_path)