# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive', force_remount=True) #connecting code to  Google Drive, in order to access and read data files

!pip install transformers==4.16.2 #installing the transformers library, which allows access to our models # try 4.16.2
from transformers import BertTokenizerFast #importing libraries for deep learning functions and models
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForTokenClassification, AdamW, BertModel
from torch.nn import Dropout
from torch import nn
from torch.nn import CrossEntropyLoss

import re
import os
datarray = []
linearray = []
foldername = '/content/drive/MyDrive/President_Botrick/CORPS_II/'
for filename in os.listdir(foldername): #reading through each speech in a loop and removing extraneous information
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

        if '{APPLAUSE}' in line: #considering lines with APPLAUSE as 'charismatic'; non-charismatic lines are added to an array as 0, charismatic as 1
            datarray.append(0)
            line = line.replace('{APPLAUSE}', '')
        else:
            datarray.append(1)
        line = line.lower()
        line = line.replace('\n', '')
        linearray.append(line)


print(datarray[10])
print(linearray[10])

print(len(datarray))
print(len(linearray))

maxlen = 200
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True, model_max_length=maxlen) #tokenizer converts sentences to numerical values
tokens = [tokenizer.encode(line) for line in linearray]
tokens = [token[0:maxlen] for token in tokens]
input_ids = pad_sequences(tokens, maxlen=maxlen, dtype="long", value=0.0, truncating="post", padding="post")
masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

#splitting data into training data, evaluation data, and testing data
test_input = input_ids[:5000]
dev_input = input_ids[5000:10000]
train_input = input_ids[10000:] 

test_label = datarray[:5000]
dev_label = datarray[5000: 10000]
train_label = datarray[10000:]

test_masks = masks[:5000]
dev_masks = masks[5000: 10000]
train_masks = masks[10000:]

train_data = TensorDataset(torch.tensor(train_input), torch.tensor(train_masks), torch.tensor(train_label))
dev_data = TensorDataset(torch.tensor(dev_input), torch.tensor(dev_masks), torch.tensor(dev_label))
test_data = TensorDataset(torch.tensor(test_input), torch.tensor(test_masks), torch.tensor(test_label))

train_sampler = RandomSampler(train_data)
dev_sampler = SequentialSampler(dev_data)
test_sampler = SequentialSampler(test_data)

bs = 64
train_data_loader = DataLoader(train_data, sampler = train_sampler, batch_size = bs)
dev_data_loader = DataLoader(dev_data, sampler = dev_sampler, batch_size = bs)
test_data_loader = DataLoader(test_data, sampler = test_sampler, batch_size = bs)

cnt=0
for k in datarray:
  if k==1:
    cnt+=1
print(cnt)

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

device = torch.device("cuda")
model = Model('bert-base-uncased', 0.1, 2)
model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    eps=1e-8
)

#for num_epochs epochs, training the model
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    itercnt = 0
    totalloss = 0
    for batch in train_data_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, mask, label = batch
        model.zero_grad()
        _, loss = model(input_ids, mask, None, label)
        totalloss += totalloss + loss.item()
        print(itercnt, loss)
        itercnt += 1
        loss.backward()
        optimizer.step()
    print("Total Loss for this Epoch:", totalloss)
    
    #computing error on evaluation and test set
    deverror = 0
    #deverror
    model.eval()
    testsen = open("/content/drive/MyDrive/President_Botrick/Test_Sentences.csv", 'w')
    for batch in dev_data_loader:
        batch = tuple(t.to(device) for t in batch)
        dev_input_ids, dev_mask, dev_label = batch
        with torch.no_grad():
            logits,_ = model(dev_input_ids, dev_mask, None, dev_label)
            predictions = torch.argmax(logits, dim = 1)
            predictions = predictions.cpu().numpy()
            dev_label = dev_label.cpu().numpy()
            for i in range(len(predictions)):
              testsen.write(str(dev_label[i]))
              testsen.write(',')
              testsen.write(str(predictions[i]))
              testsen.write(',')
              testsen.write(str(tokenizer.decode(dev_input_ids[i].tolist())))
              testsen.write('\n')
              if predictions[i] != dev_label[i]:
                  deverror += 1
    print(deverror)
    print(deverror/5000)

    #compute error on test set
    #testerror
    testerror = 0
    for batch in test_data_loader:
        batch = tuple(t.to(device) for t in batch)
        test_input_ids, test_mask, test_label = batch
        with torch.no_grad():
            logits, _ = model(test_input_ids, test_mask, None, test_label)
            predictions = torch.argmax(logits, dim=1)
            predictions = predictions.cpu().numpy()
            test_label = test_label.cpu().numpy()
            for i in range(len(predictions)):
                if predictions[i] != test_label[i]:
                    testerror += 1
    print(testerror)
    print(testerror/5000)
torch.save(model.state_dict(), "/content/drive/MyDrive/President_Botrick/reverse_model.pt")
