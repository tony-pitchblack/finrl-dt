from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Ensure compatibility and correct device placement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load the GPT-2 model and tokenizer with explicit configurations
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', force_download=True)
print("Tokenizer loaded.")

print("Loading GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained('gpt2', force_download=True)
model.config.pad_token_id = model.config.eos_token_id  # Set pad_token_id
model.to(device)
print("Model loaded.")

prompt = "What is 1+1?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Generating text...")
outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", generated_text)