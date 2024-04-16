import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch import nn


print("Initializing dataset generator class")

class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data['text']
        self.targets = self.data['labels']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
print("Creating model class")

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class myModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.l1 = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.linear1 = nn.Linear(768, 768, bias = True)
        self.dropout1 = nn.Dropout(p = 0.1)
        self.linear2 = nn.Linear(768, 1024, bias = True)
        self.dropout2 = nn.Dropout(p = 0.1)
        self.linear3 = nn.Linear(1024, 768, bias = True)
        self.dropout3 = nn.Dropout(p = 0.1)
        self.linear4 = nn.Linear(768, 3, bias = True)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1.roberta(input_ids=input_ids, attention_mask=attention_mask)        
        linear1 = self.linear1(output_1[0])
        dropout1 = self.dropout1(linear1)        
        linear2 = self.linear2(dropout1)
        dropout2 = self.dropout2(linear2)
        linear3 = self.linear3(dropout2)
        dropout3 = self.dropout3(linear3)
        linear4 = self.linear4(dropout3)
        output = torch.sum(linear4, dim = 1)
        
        return output

def validation(testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    print(fin_targets, fin_outputs)
    return fin_outputs, fin_targets


def train(epoch):
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        loss.backward()
        optimizer.step()


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


if __name__ == "__main__":

    print("Setting model training parameters")

    MAX_LEN = 300
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    LEARNING_RATE = 1e-04


    train_df = pd.read_csv("train.csv")
    train_df['labels'] = train_df['labels'].map(eval)


    print("Loading Model")

    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading dataset generator")
    #  Creating the dataset and dataloader for the neural network

    train_size = 0.8
    train_data=train_df.sample(frac=train_size,random_state=200)
    validation_data=train_df.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)


    print("FULL Dataset: {}".format(train_df.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("TEST Dataset: {}".format(validation_data.shape))

    training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)
    validation_set = MultiLabelDataset(validation_data, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(validation_set, **test_params)


    print("Initializing model")

    model = myModel(model_name)
    model.to(device)


    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


    print("Running training loop")

    EPOCHS = 1

    for epoch in range(EPOCHS):
        train(epoch)
        
    torch.save(model.state_dict(), "modet.pth")
    
