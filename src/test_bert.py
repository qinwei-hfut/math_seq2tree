import torch
from transformers import BertTokenizer, BertModel
import pdb

pretrained = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained)
model = BertModel.from_pretrained(pretrained)

input_text = "第一天挖了316方，连续挖了NUM天，一周共挖土多少方？"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
pooler_output  = outputs.pooler_output
pdb.set_trace()
# last_hidden_state   = outputs.last_hidden_state
print()