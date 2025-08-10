
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Test run of the model
text = "hi there gpt2, how are you today?"
encode = tokenizer(text, return_tensors='pt')
out = model(**encode)

print(out)
