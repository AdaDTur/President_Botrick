# -*- coding: utf-8 -*-
"""Lines_Classification

This file is for classifying generated lines as influential or not. This takes the output of the generation portion of the
experiment and plugs it back into the first portion.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True) #connecting to Google Drive

!pip install transformers==4.16.2 #installing the transformers library, which allows access to our models
from transformers import BertTokenizerFast, BertTokenizer #importing libraries for deep learning functions and models
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
from transformers import BertModel

#defining model and its architecture, which consists of BERT layer, followed by a linear layer for classification
class Model(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_labels: int):
        super(Model, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)
        self.dropout = Dropout(dropout)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, num_labels)

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                label: torch.tensor = None
                ):
        output = self.bert_model(input_ids=input_ids,
                                                            attention_mask=attention_mask,
                                                            token_type_ids=token_type_ids)
        logits = self.classifier(self.dropout(output.pooler_output))

        loss_fct = CrossEntropyLoss()
        # Compute losses if labels provided
        if label is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), label.type(torch.long))
        else:
            loss = torch.tensor(0)

        return logits, loss

maxlen = 200
bertdevice = torch.device("cuda")
bertmodel = Model('bert-base-uncased', 0.1, 2)
bertmodel.to(bertdevice)
bertmodel.load_state_dict(torch.load('/content/drive/MyDrive/President_Botrick/model0315.pt'))
berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, model_max_length=maxlen) #intialize tokenizer

gpttokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium') 
gptmodel = torch.load('/content/drive/MyDrive/President_Botrick/jul17app9.pt')
gptdevice = torch.device("cuda")
gptmodel.to(gptdevice)

outputfile = open('foo.txt', 'w')

def generate(input_context):
    input_ids = gpttokenizer.encode(input_context, return_tensors='pt').to(gptdevice)
    outputs = gptmodel.generate(input_ids=input_ids, max_length=maxlen, do_sample=True, top_k=50, top_p=0.9, num_return_sequences=1)
    decoded = gpttokenizer.decode(outputs[0])
    decoded = decoded.replace("<|endoftext|>","")
    decoded = decoded.replace('{APPLAUSE}', '')
    print(decoded)
    outputfile.write(decoded + '\n')
    return(decoded)

def classify(decoded):
  tokens = [berttokenizer.encode(decoded.lower())]
  tokens = [token[0:maxlen] for token in tokens]
  test_input_ids = pad_sequences(tokens, maxlen=maxlen, dtype="long", value=0.0, truncating="post", padding="post")
  print(test_input_ids.shape)
  test_masks = [[float(i != 0.0) for i in ii] for ii in test_input_ids]
  test_data = TensorDataset(torch.tensor(test_input_ids), torch.tensor(test_masks))
  test_sampler = SequentialSampler(test_data)
  test_data_loader = DataLoader(test_data, sampler = test_sampler, batch_size = 1)
  with torch.no_grad():
    for batch in test_data_loader:
      batch = tuple(t.to(bertdevice) for t in batch)
      i, m = batch
      logits, _ = bertmodel(i, m, None, None)
      predictions = torch.argmax(logits, dim=1)
      logits = logits.cpu().numpy()
      print(logits)
      props = 1 / (1 + np.exp(-1 * logits))
      print(props)
      print(str(props[0][1] * 100) + '% to be applauded')
  return(predictions)

count = 0
tot = 0
num = 100
#contexts = ['White bears are', 'Artificial intelligence is', 'Global warming is', 'Water pollution is', 'Gokhan is', 'Quantum physics is', 'Psychedelic drugs are', 'Penguin migrations are', 'Climbing mountains is', 'The music industry is', 'A clothing shortage is', 'Conflicts in Serbia are', 'Extremist organizations are', 'Smeltering heat waves are', 'Deforestation is', 'Our promise to Alaskans is', 'Public transportation is', 'Electricians are', 'Polarization is', 'The city of Ankara, Turkey is', 'Norwegian influence is', 'Water bills are', 'Roses are', 'Homeless populations are', 'Coyotes are', 'Assisting wildlife is', 'Natural disasters are', 'Rock music is', 'High-speed internet is', 'Machine learning is', 'National parks are', 'California\'s redwood forests are', 'The tallest buildings are', 'The meaning of life is', 'College admissions are', 'World hunger is', 'Consumerism is', 'Movie theaters are', 'Political expression is', 'Transgender communities are', 'Bankruptcy is', 'Animal shelters are', 'Nuclear energy is', 'Ghosts are', 'Insurance for health is', 'Our favorite movies are', 'Affordable housing is', 'The World Cup is', 'The largest neighborhoods are', 'Music production is','Christmas is', 'Traditional', 'Thanksgiving is', 'Colorful', 'Our neighbors are', 'Native territories are', 'Foreign invasion is', 'Breast cancer is', 'Marriage between', 'Divorce in families is', 'Adoption is', 'Fostering is', 'Urban planning is', 'Animal cruelty is', 'Veganism is', 'Job interviews are', 'Account balances are', 'Graduation is', 'Amazon warehouses are', 'Undeveloped countries are',' The great pyramids are', 'Silicon Valley is', 'Cat food is', 'The automobile industry is', 'Mobile phones are', 'Serial killers are', 'Cryptocurrency is', 'Chimney cleaning is', 'Capitalism is', 'Planting trees are', 'Backyards are', 'Suburban', 'Broken cities are', 'The war on drugs is', 'Gun control is', 'Columbia University is', 'Tom Cruise is', 'Jeff Bezos is', 'Television production is', 'Lethal viruses are', 'Pacific islanders are', 'Celebrities are', 'Los Angeles is', 'New York is', 'American football is', 'Counterfeit currencies are', 'Military submarines are', 'Winter blizzards are', 'Farmers markets are', 'The East coast is']
#contexts = ['The', 'we want', 'this election', 'the united states', 'these', 'our children', 'the schools', 'our future', 'this party', 'the past']
contexts=['The']

for i in range(len(contexts)):
  for j in range(int(num/len(contexts))):
    decoded = generate(contexts[i].lower())
    predictions = classify(decoded).cpu().numpy()[0]
    if(predictions == 1):
      count += 1
    tot +=1
    print(str(count) + '|' + str(tot) + '|' + str(count/tot))
