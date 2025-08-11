import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/DialoGPT-small"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torchscript=True)
model.eval()

print(model)

example_input = torch.randint(0, 100, (1,128), dtype=torch.int32)
traced_model = torch.jit.trace(model, example_input)



