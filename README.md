
# Finetuned Gemma LLM

A finetuned LLM specifically trained on datasets of python codes to learn python and help developers in debugging.

[Model Link](https://huggingface.co/shahdishank/gemma-2b-it-finetune-python-codes)
## Run Model on Google Colab CPU

- Create read access token on Hugging Face [[Here]](https://huggingface.co/settings/tokens)

Install transformers library
```bash
pip install transformers
```

Use LLM on Google Colab to Generate Code
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "shahdishank/gemma-2b-it-finetune-python-codes"
HUGGING_FACE_TOKEN = "YOUR_TOKEN"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="HUGGING_FACE_TOKEN")
model = AutoModelForCausalLM.from_pretrained(model_name, token="HUGGING_FACE_TOKEN")

prompt_template = """\
  user:\n{query} \n\n assistant:\n
  """
prompt = prompt_template.format(query="write a simple python function") # write your query here

input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
outputs = model.generate(**input_ids, max_new_tokens=2000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
## Screenshots
![llm_output](https://github.com/ShahDishank/gemma-finetuned/assets/109618750/01541065-76fd-4ac4-9151-ab2f4efc3329)

## Features

- Code generation
- Debugging
- Learn and understand various python coding styles
## Tech Stack

**Language:** Python

**Library:** transformers, PEFT

**LLM:** Gemma-2b-it

**IDE:** Google Colab
## Resources Used

 - Base Model:
    - [Gemma Model Docs](https://ai.google.dev/gemma/docs)
    - [Gemma Model from Hugging Face](https://huggingface.co/google/gemma-2b-it)
 - Dataset:
    - [python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k)
## Feedback

If you have any feedback, please reach out to me at shahdishank24@gmail.com


## Author

- [@shahdishank](https://www.github.com/ShahDishank)

