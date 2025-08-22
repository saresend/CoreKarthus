import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import coremltools as ct
import numpy as np

MODEL_ID = "apple/OpenELM-270M"

print("🔄 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torchscript=True).eval()


# Test with raw HuggingFace model
print("🧪 Testing raw HuggingFace model...")

test_prompt = "hi there whats your name"
test_input_ids = tokenizer.encode(test_prompt, return_tensors="pt")

test_outputs = model.generate(
        test_input_ids,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

test_generated_text = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
print(f"Raw HF model output: {test_generated_text}")
print()

