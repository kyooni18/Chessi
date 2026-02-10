from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "kyooni18/chessi-0.1"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

prompt = "e4 e5 Nf3 Nc6 Bb5"
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=1)
new_ids = out[:, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_ids[0], skip_special_tokens=True))


