# -*- coding: utf-8 -*-
"""President_Botrick_2
This portion of the code consists of the generative task, writing out new, novel speeches based 
on the understanding of persuasiveness created in President_Botrick, using GPT-2
"""

!pip3 install transformers==4.16.2 #installing transformers library

from google.colab import drive
drive.mount('/content/drive', force_remount=True) #connecting to Google Drive

#importing libraries
from transformers import AutoTokenizer, AutoModelWithLMHead #4.20.1
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

import re
import os
datarray = []
linearray = []
foldername = '/content/drive/MyDrive/President_Botrick/CORPS_II/'
for filename in os.listdir(foldername):             #reading through each speech in a loop and removing extraneous information
    print(filename)
    filename = foldername + filename
    data1 = open(filename, 'r')
    lines = data1.readlines()
    for line in lines:
        line = re.sub('\{event.*?event\}', '', line)
        line = re.sub('\{date.*?date\}', '', line)
        line = re.sub('\{speaker.*?speaker\}', '', line)
        line = re.sub('\{source.*?source\}', '', line)
        line = re.sub('\{description.*?description\}', '', line)
        line = re.sub('\{title.*?title\}', '', line)
        line = re.sub('\{AUDIENCE.*?AUDIENCE\}', '', line)
        line = re.sub('\{COMMENT=.*?"\}', '', line)
        line = line.replace('{LAUGHTER}', '')
        line = line.replace('{BOOING}', '')
        line = line.replace('{speech}', '')
        line = line.replace('{/speech}', '')
        if line == '\n':
            continue

        if '{APPLAUSE}' not in line:
            linearray.append(line)    #only considering charismatic lines
        line = line.lower()
        line = line.replace('\n', '')

maxlen = 100
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium') #initialize tokenizer
model = AutoModelWithLMHead.from_pretrained('gpt2-medium')
input_context = 'i am ' #stem words
input_ids = tokenizer.encode(input_context, return_tensors='pt')
print(input_ids)
outputs = model.generate(input_ids=input_ids, max_length=maxlen, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=3)
print(tokenizer.decode(outputs[0]))
print(tokenizer.decode(outputs[1]))
print(tokenizer.decode(outputs[2]))

tokens = [tokenizer.encode(line) for line in linearray] #tokenizer converts sentences to numerical values
tokens = [token[0:maxlen] for token in tokens]
input_ids = pad_sequences(tokens, maxlen=maxlen, dtype="long", value=50256, truncating="post", padding="post")
masks = [[float(i != 50256) for i in ii] for ii in input_ids]
data = TensorDataset(torch.tensor(input_ids), torch.tensor(masks))

#split data into training, evaluation, and testing
print(len(input_ids))
test_input = input_ids[0:500]
dev_input = input_ids[500:1000]
train_input = input_ids[41646:]

test_masks = masks[0:500]
dev_masks = masks[500:1000]
train_masks = masks[41646:]

train_data = TensorDataset(torch.tensor(train_input), torch.tensor(train_masks))
dev_data = TensorDataset(torch.tensor(dev_input), torch.tensor(dev_masks))
test_data = TensorDataset(torch.tensor(test_input), torch.tensor(test_masks))

train_sampler = RandomSampler(train_data)
dev_sampler = SequentialSampler(dev_data)
test_sampler = SequentialSampler(test_data)

bs = 16
train_data_loader = DataLoader(train_data, sampler = train_sampler, batch_size = bs)
dev_data_loader = DataLoader(dev_data, sampler = dev_sampler, batch_size = bs)
test_data_loader = DataLoader(test_data, sampler = test_sampler, batch_size = bs)

#function to generate text
def generate():
    model.eval()
    with torch.no_grad():
      input_context = '<|endoftext|>'
      input_ids = tokenizer.encode(input_context, return_tensors='pt') # encode input context
      model.to(torch.device('cuda'))
      outputs = model.generate(input_ids=input_ids, max_length=maxlen, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=5)
      print(tokenizer.decode(outputs[0]))
      print(tokenizer.decode(outputs[1]))
      print(tokenizer.decode(outputs[2]))

#generate()

#training data, fine-tuning GPT-2 for 10 epochs
device = torch.device('cuda')

optimizer = AdamW(
    model.parameters(),
    lr=5e-4,
    eps=1e-8
)

total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
  itercnt = 0
  for batch in train_data_loader:
    itercnt += 1
    model.train()
    model.to(device)
    batch = tuple(t.to(device) for t in batch)
    input, mask = batch
    model.zero_grad()
    output = model(input_ids = input,
                  attention_mask = mask,
                  token_type_ids = None,
                  labels = input)
    loss = output[0]
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    if itercnt%100==0:
      print(itercnt)
      #generate()
  print(epoch,"\n")
  #generate()
  filename = "/content/drive/MyDrive/President_Botrick/jul18nonapp"+str(epoch)+".pt"
  torch.save(model, filename)
  print("Total Loss: ",total_loss)
  total_loss = 0
