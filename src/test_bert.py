import torch
from transformers import BertTokenizer, BertModel
import pdb

pretrained = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained)
model = BertModel.from_pretrained(pretrained)

# outputs.last_hidden_state
while True:
    input_text = "第一天挖了316方，连续挖了NUM天，一周共挖土，多少方？"
    input_text = "第一天挖了316方，"
    pdb.set_trace()
    inputs = tokenizer(input_text, return_tensors="pt")
    # inputs.input_ids
    print(len(input_text))
    print(inputs.input_ids.shape)
    outputs = model(**inputs)
    pooler_output  = outputs.pooler_output
    last_hidden_state   = outputs.last_hidden_state
    # pdb.set_trace()
    print()