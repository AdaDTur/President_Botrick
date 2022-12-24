# -*- coding: utf-8 -*-


!pip3 install transformers==4.16.2 #installing transformers library

from google.colab import drive #connecting Google Drive
drive.mount('/content/drive', force_remount=True)

#install libraries
from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd
import transformers
import torch
from torch.nn import Dropout
from keras.preprocessing.sequence import pad_sequences
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2TokenizerFast, AdamW
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
import random

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium') 
model = torch.load('/content/drive/MyDrive/President_Botrick/reverse_model.pt')
#initializing tokenizer and loading model from Drive

#generating text from prior code, but looping to get full sentence
#using last five words of previous sentence as context for next sentence to assist flow
maxlen = 200
input_context = 'White bears'
for i in range(3):
  input_ids = tokenizer.encode(input_context, return_tensors='pt')
  outputs = model.generate(input_ids=input_ids, max_length=maxlen, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
  decoded = tokenizer.decode(outputs[0])
  decoded = decoded.replace("<|endoftext|>","")
  print(decoded)
  print("\n")
  decoded_array = decoded.split()
  decoded_array = decoded_array[-5:]
  input_context = ' '.join(decoded_array)
