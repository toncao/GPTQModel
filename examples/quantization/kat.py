from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "Kwaipilot/KAT-V1-40B"
quant_path = "/mnt/LinuxDrive/huggingface/hub/KAT-V1-40B-GPTQ-2bit"

calibration_dataset = load_dataset("sqres/tulu3_qwen32b", split="train").shuffle(seed=42).select(range(1024))["conversations"]
calibration_dataset = [content[0]["content"] for content in calibration_dataset]

quant_config = QuantizeConfig(bits=2, group_size=128, v2=True, desc_act=False, hyb_act=True)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1, auto_gc = False, buffered_fwd = True)

model.save(quant_path)
