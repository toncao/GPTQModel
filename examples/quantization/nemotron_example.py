from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
quant_path = "./Llama-3_3-Nemotron-Super-49B-v1_5-GPTQ-4bit"

dataset_id = "nvidia/Llama-Nemotron-Post-Training-Dataset"
subset = "SFT"

calibration_dataset = load_dataset(
    dataset_id,
    subset
  ).shuffle(seed=42).select(range(1024))["output"]

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
calibration_dataset = [tokenizer(output) for output in calibration_dataset]

quant_config = QuantizeConfig(bits=4, group_size=128, v2=True, hyb_act=True, desc_act=False)


model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=4, auto_gc = False, buffered_fwd = True)

model.save(quant_path)