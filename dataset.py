import torch
from torch.utils.data import DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CitationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item =  {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        item['point'] = self.encodings[idx]
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)

def create_dataloader(tokenizer, edges, nodes_citation, labels, batch):
    tokenized_citations = tokenizer(edges, truncation=True, padding=True, max_length=512, return_tensors='pt')
    combined_dataset = []
    tokenized_citations['input_ids'] = tokenized_citations['input_ids'].to(device)
    tokenized_citations['attention_mask'] = tokenized_citations['attention_mask'].to(device)
    tokenized_citations['token_type_ids'] = tokenized_citations['token_type_ids'].to(device)
    for i in range(len(tokenized_citations['input_ids'])):
        citation = (tokenized_citations['input_ids'][i], tokenized_citations['attention_mask'][i], tokenized_citations['token_type_ids'][i])
        titles = nodes_citation[i]
        combined_dataset.append((citation, titles))
    citation_dataset = CombinedDataset(combined_dataset,  labels)
    citation_dataloader = DataLoader(citation_dataset, batch_size=batch, shuffle=False)

    return citation_dataloader
