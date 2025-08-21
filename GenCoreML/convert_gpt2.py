import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import coremltools as ct
import numpy as np

MODEL_ID = "microsoft/DialoGPT-small"

print("🔄 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torchscript=True).eval()

sample_input = torch.zeros((1,128), dtype=torch.int32)
sequence_length=torch.export.Dim(name="sequence_length", min=1, max=128)
dynamic_shapes = { "input_ids": {1: sequence_length}}

print("🚀 Exporting to torch program...")
exported_program = torch.jit.trace(model, sample_input)

print("🍎 Converting to CoreML...")
mlmodel = ct.convert(exported_program,
        inputs=[ct.TensorType(
                name="input_ids",
                shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=100))),
                dtype=np.int32
            )],
                  )

prompt = "Once upon a time there was"
tokenized_prompt = torch.tensor(tokenizer(prompt)["input_ids"])
# Since model takes input ids in batch,
# create a dummy batch dimension (i.e. size 1) for tokenized prompt
tokenized_prompt = tokenized_prompt.unsqueeze(0)

input_ids = np.int32(tokenized_prompt.detach().numpy())
# extend sentence (sequence) word-by-word (token-by-token)
# until reach max sequence length
for i in range(0, 50):
    logits = list(mlmodel.predict({"input_ids": input_ids}).values())[0]
    # determine the next token by greedily choosing the one with highest logit (probability)
    output_id = np.argmax(logits, -1)[:, -1 :]
    # append the next token to sequence
    input_ids = np.concat((input_ids, output_id), dtype=np.int32, axis=-1)
# decode tokens back to text
output_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
print("Output text from the converted Core ML model:")
print(output_text)


















