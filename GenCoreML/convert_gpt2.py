import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
modelInfo = AutoConfig.from_pretrained(MODEL_ID)
# print(modelInfo)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torchscript=True)
model.eval()

example_input = torch.zeros((1,128), dtype=torch.int32)



## Try Running the Raw Pytorch model 
prompt = "Hello there, who are you?"
tokenized_prompt = tokenizer(prompt)



inputs = tokenizer.apply_chat_template(
   [{ "role": "user", "content": prompt }],
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

traced_model = torch.jit.trace(model, example_input)

# Export to CoreML
import coremltools as ct

# Convert the model to CoreML
mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(name="input", shape=example_input.shape)])

# Save the model
mlmodel.save("gpt2_coreml.mlpackage")

#print(inputs)
#outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
