from torch_geometric.nn import GraphSAGE, PNA, GAT
from torch.nn import Linear, Dropout
from acl_graph import *
from transformers import AutoModel
from datasets import load_metric


class CitationMLP(torch.nn.Module):
  def __init__(self,  num_features, num_classes, hidden_channels, graph):
    super(CitationMLP, self).__init__()
    self.pred = Linear(2 * num_features, num_classes)
    self.graph = graph

  def forward(self, nodes):
      c = torch.cat((self.graph.x[nodes[0]], self.graph.x[nodes[1]]), dim=1)
      pred = self.pred(c)
      return pred

  def train_step(self, dataloader, optimizer, scheduler, criterion):
      running_loss = 0.
      self.train()
      for i, data in enumerate(dataloader):
          # Every data instance is an input + label pair

          optimizer.zero_grad()  # Clear gradients.

          # Perform a single forward pass.
          nodes = data['point'][1]
          pred = self(nodes)

          loss = criterion(pred, data['labels'])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
          optimizer.step()  # Update parame
          scheduler.step()
          # Gather data and report
          running_loss += loss.item()

      # Calculate the average loss over all of the batches.
      avg_train_loss = running_loss/len(dataloader)
      # Measure how long this epoch took.
      return avg_train_loss

  def test_step(self, dataloader, criterion):
      metric = load_metric("f1")
      self.eval()
      running_loss = 0.
      all_pred = torch.tensor([], device=device)
      all_labels = torch.tensor([], device=device)
      for i, data in enumerate(dataloader):
        nodes = data['point'][1]
        with torch.no_grad():
            pred = self(nodes)
        loss = criterion(pred, data['labels'])
        running_loss += loss.item()
        pred = pred.argmax(dim=1)  # Use the class with highest probability
        all_pred = torch.cat((all_pred,pred),0)
        all_labels = torch.cat((all_labels, data['labels']), 0)

      avg_train_loss = running_loss / len(dataloader)
      return metric.compute(predictions=all_labels, references=all_pred, average="macro")['f1'], avg_train_loss

class CitationBERT(torch.nn.Module):
  def __init__(self,  num_features, num_classes, hidden_channels, graph):
    super(CitationBERT, self).__init__()
    self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    self.dropout = Dropout(0.1)
    self.pred = Linear(num_features, num_classes)

  def forward(self, input_ids, attention_mask, token_type_ids):
    citation = self.bert(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).last_hidden_state[:, 0, :]  # batch_size, sequence_length, hidden_size; we take the CLS token embedding
    drop_c = self.dropout(citation)
    pred = self.pred(drop_c)
    return pred

  def train_step(self, dataloader, optimizer, scheduler, criterion):
      running_loss = 0.
      self.train()
      for i, data in enumerate(dataloader):
          # Every data instance is an input + label pair

          optimizer.zero_grad()  # Clear gradients.

          # Perform a single forward pass.

          input_ids = data['point'][0][0]
          attention_mask = data['point'][0][1]
          token_type_ids = data['point'][0][2]

          pred = self(input_ids,  attention_mask, token_type_ids)

          loss = criterion(pred, data['labels'])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
          optimizer.step()  # Update parame
          scheduler.step()
          # Gather data and report
          running_loss += loss.item()

          # Calculate the average loss over all of the batches.
      avg_train_loss = running_loss/len(dataloader)

      # Measure how long this epoch took.

      return avg_train_loss

  def test_step(self, dataloader, criterion):
      metric = load_metric("f1")
      self.eval()
      running_loss = 0.
      all_pred = torch.tensor([], device=device)
      all_labels = torch.tensor([], device=device)
      for i, data in enumerate(dataloader):
        input_ids = data['point'][0][0]
        attention_mask = data['point'][0][1]
        token_type_ids = data['point'][0][2]
        with torch.no_grad():
            pred = self(input_ids, attention_mask, token_type_ids)
        loss = criterion(pred, data['labels'])
        running_loss += loss.item()
        pred = pred.argmax(dim=1)  # Use the class with highest probability
        all_pred = torch.cat((all_pred,pred),0)
        all_labels = torch.cat((all_labels, data['labels']), 0)

      avg_train_loss = running_loss / len(dataloader)
      return metric.compute(predictions=all_labels, references=all_pred, average="macro")['f1'], avg_train_loss



class CitationGAT(torch.nn.Module):
  def __init__(self, num_features, num_classes, hidden_channels, graph):
      super(CitationGAT, self).__init__()
      self.graph_layer = GAT(num_features, hidden_channels, 2, edge_dim=768, add_self_loops = False)
      self.graph = graph
      #self.combined = Linear(hidden_channels, hidden_channels)
      self.pred = Linear(hidden_channels, num_classes)

  def forward(self, nodes):
      titles = self.graph_layer(self.graph.x, self.graph.adj_t, self.graph.edge_attr)
      #c = torch.cat((titles[nodes[0]], titles[nodes[1]]), dim=1)
      c = torch.mul(titles[nodes[0]], titles[nodes[1]])
      #combined = self.combined(c)
      pred = self.pred(c)
      return pred

  def train_step(self, dataloader, optimizer, scheduler, criterion):
      running_loss = 0.
      self.train()
      for i, data in enumerate(dataloader):
          # Every data instance is an input + label pair

          optimizer.zero_grad()  # Clear gradients.

          # Perform a single forward pass.
          nodes = data['point'][1]
          pred = self(nodes)

          loss = criterion(pred, data['labels'])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
          optimizer.step()  # Update parame
          scheduler.step()
          # Gather data and report
          running_loss += loss.item()

          # Calculate the average loss over all of the batches.
      avg_train_loss = running_loss / len(dataloader)

      # Measure how long this epoch took.

      return avg_train_loss

  def test_step(self, dataloader, criterion):
      metric = load_metric("f1")
      self.eval()
      running_loss = 0.
      all_pred = torch.tensor([], device=device)
      all_labels = torch.tensor([], device=device)
      for i, data in enumerate(dataloader):
          nodes = data['point'][1]
          with torch.no_grad():
              pred = self(nodes)
          loss = criterion(pred, data['labels'])
          running_loss += loss.item()
          pred = pred.argmax(dim=1)  # Use the class with highest probability
          all_pred = torch.cat((all_pred, pred), 0)
          all_labels = torch.cat((all_labels, data['labels']), 0)

      avg_train_loss = running_loss / len(dataloader)
      return metric.compute(predictions=all_labels, references=all_pred, average="macro")['f1'], avg_train_loss


class CitationBERTGAT(torch.nn.Module):
  def __init__(self, num_features, num_classes, hidden_channels, graph):
      super(CitationBERTGAT, self).__init__()
      self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

      self.dropout = Dropout(0.1)
      self.merged = Linear(hidden_channels + 768, hidden_channels)  # set up the other FC layer
      self.graph_layer = GAT(num_features, hidden_channels, 2)
      self.graph = graph
      self.pred = Linear(hidden_channels, num_classes)


  def forward(self, input_ids, attention_mask, token_type_ids, nodes):
      citation = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:, 0, :]  # batch_size, sequence_length, hidden_size; we take the CLS token embedding
      drop_c = self.dropout(citation)
      titles = self.graph_layer(self.graph.x, self.graph.adj_t)
      # now we can reshape `c` and `f` to 2D and concat them
      t = torch.mul(titles[nodes[0]], titles[nodes[1]])
      #combined = torch.cat((drop_c.view(drop_c.size(0), -1), titles[nodes[0]], titles[nodes[1]]), dim=1)
      combined = torch.cat((drop_c.view(drop_c.size(0), -1), t), dim=1)
      merged = self.merged(combined)
      pred = self.pred(merged)
      return pred

  def train_step(self, dataloader, optimizer, scheduler, criterion):
      running_loss = 0.
      self.train()
      for i, data in enumerate(dataloader):
          # Every data instance is an input + label pair

          optimizer.zero_grad()  # Clear gradients.

          # Perform a single forward pass.

          input_ids = data['point'][0][0]
          attention_mask = data['point'][0][1]
          token_type_ids = data['point'][0][2]
          nodes = data['point'][1]
          pred = self(input_ids, attention_mask, token_type_ids, nodes)

          loss = criterion(pred, data['labels'])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
          optimizer.step()  # Update parame
          scheduler.step()
          # Gather data and report
          running_loss += loss.item()

          # Calculate the average loss over all of the batches.
      avg_train_loss = running_loss / len(dataloader)

      # Measure how long this epoch took.

      return avg_train_loss

  def test_step(self, dataloader, criterion):
      metric = load_metric("f1")
      self.eval()
      running_loss = 0.
      all_pred = torch.tensor([], device=device)
      all_labels = torch.tensor([], device=device)
      for i, data in enumerate(dataloader):
          input_ids = data['point'][0][0]
          attention_mask = data['point'][0][1]
          token_type_ids = data['point'][0][2]
          nodes = data['point'][1]
          with torch.no_grad():
              pred = self(input_ids, attention_mask, token_type_ids, nodes)
          loss = criterion(pred, data['labels'])
          running_loss += loss.item()
          pred = pred.argmax(dim=1)  # Use the class with highest probability
          all_pred = torch.cat((all_pred, pred), 0)
          all_labels = torch.cat((all_labels, data['labels']), 0)

      avg_train_loss = running_loss / len(dataloader)
      return metric.compute(predictions=all_labels, references=all_pred, average="macro")['f1'], avg_train_loss
