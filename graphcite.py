from citation_models import *
from acl_graph import *
from scicite_graph import *
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataset import CombinedDataset, create_dataloader
from sentence_transformers import SentenceTransformer
import json
import os
import torch
import random
import numpy as np
import pandas as pd
import time
import datetime
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):
    """
    Everything should be reproducible
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # these are just for deterministic behaviour
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def pretty_print_stats(stats):
    pd.set_option('precision', 4)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    print(df_stats)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    params = {
    'dataset': 'acl-arc',
    'model': 'CitationBERT',
    'epochs': 5,
    'learning_rate': 2e-5,
    'batch_size': 16,
    'model_location': 'best_model',
    'seed':0,
    'add_authors': False,
    'add_venues': False
    }
    parser.add_argument("dataset", help="on which dataset to run the model")
    parser.add_argument("model", help="which model to use, from CitationBERT, CitationGAT, CitationBERTGAT, CitationBERTGraphSAGE")
    parser.add_argument("-b", "--batch_size", type=int, help="batch size, default 16")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs, default 5")
    parser.add_argument("-s", "--seed", type=int, help="seed for random generation, default 0")
    parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate, default 2e-5")
    parser.add_argument("-a", "--authors", type=bool, help="if to add authors, default False")
    parser.add_argument("-v", "--venues", type=bool, help="if to add venues, default False")
    args = parser.parse_args()
    all_models = {
        'CitationBERT': CitationBERT,
        'CitationGAT': CitationGAT,
        'CitationBERTGAT': CitationBERTGAT,
        'CitationBERTGraphSAGE': CitationBERTGraphSAGE,
        'CitationMLP': CitationMLP
    }
    classes_toadd = 0
    params['model'] = args.model
    params['dataset'] = args.dataset
    model = all_models[args.model]
    if args.batch_size:
        params['batch_size'] = args.batch_size
    if args.epochs:
        params['epochs'] = args.epochs
    if args.seed:
        params['seed'] = args.seed
    if args.learning_rate:
        params['learning_rate'] = args.learning_rate
    if args.authors:
        params['add_authors'] = args.authors
        classes_toadd +=1
    if args.venues:
        params['add_venues'] = args.venues
        classes_toadd += 1
    set_seed(args.seed)
    params['model_location'] = params['model']+params['dataset']+"B"+ str(params['batch_size'])+"E"+str(params['epochs'])+"S"+str(params['seed'])+"LR"+str(params['learning_rate'])
    if not os.path.exists(params['model_location']):
        os.mkdir(params['model_location'])
    f = open(params['model_location']+'/statistics.json', mode='w')
    f.write(json.dumps(params))
    f.write("\n")
    training_stats = []
    print(params)

    sentence_trans = SentenceTransformer('allenai/scibert_scivocab_uncased')

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    # we store in all_nodes information on the id in the first list and then in the second the title of the paper
    all_nodes = ([], [])
    all_title2nodeid = {}
    # we store the information in the edges objects as 3 lists: the first one only the connectivity information,
    # the second one the citation text, the third one the citation intent
    edges_train = ([], [], [])
    edges_dev = ([], [], [])
    edges_test = ([], [], [])
    all_edges = ([], [], [])
    num_classes = 3 + classes_toadd
    if params['dataset'] == 'acl-arc':
        read_acl("acl-arc/train.jsonl", all_nodes, edges_train, all_title2nodeid, params['add_authors'], params['add_venues'])
        read_acl("acl-arc/dev.jsonl", all_nodes, edges_dev, all_title2nodeid, params['add_authors'], params['add_venues'])
        read_acl("acl-arc/test.jsonl", all_nodes, edges_test, all_title2nodeid, params['add_authors'], params['add_venues'])
        num_classes = 6 + classes_toadd
    else:
        read_scicite("scicite/train.jsonl", all_nodes, edges_train, all_title2nodeid, params['add_authors'], params['add_venues'])
        read_scicite("scicite/dev.jsonl", all_nodes, edges_dev, all_title2nodeid, params['add_authors'], params['add_venues'])
        read_scicite("scicite/test.jsonl", all_nodes, edges_test, all_title2nodeid, params['add_authors'], params['add_venues'])

    all_edges[0].extend(edges_train[0] + edges_dev[0] + edges_test[0])
    all_edges[1].extend(edges_train[1] + edges_dev[1] + edges_test[1])
    all_edges[2].extend(edges_train[2] + edges_dev[2] + edges_test[2])
    print(len(all_edges[0]))

    print(len(all_nodes[0]))
    print(len(all_nodes[1]))
    graph = create_data_object(all_nodes, all_edges, sentence_trans)

    model = all_models[args.model](768, num_classes, 128, graph).to(device)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=0.01)  # Define optimizer.

    train_citation_dataloader = create_dataloader(tokenizer, edges_train[1], edges_train[0], edges_train[2], params['batch_size'])
    dev_citation_dataloader = create_dataloader(tokenizer, edges_dev[1], edges_dev[0], edges_dev[2], params['batch_size'])
    test_citation_dataloader = create_dataloader(tokenizer, edges_test[1], edges_test[0], edges_test[2], params['batch_size'])

    total_steps = len(train_citation_dataloader) * params['epochs']
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,  num_training_steps=total_steps)


    best_f1 = 0.
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    for epoch in range(params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1,params['epochs']))
        print('Training...')

        t0 = time.time()
        train_loss = model.train_step(train_citation_dataloader, optimizer, scheduler, criterion)
        training_time = format_time(time.time() - t0)
        f1, dev_loss = model.test_step(dev_citation_dataloader, criterion)
        if f1 > best_f1:
            torch.save(model, params['model_location']+"/best_model")
            best_f1 = f1

        print("")
        print("  Average training loss: {0:.2f}".format(train_loss))
        print("  Training epoch took: {:}".format(training_time))
        print("  Average dev loss: {0:.2f}".format(dev_loss))
        print("  F1 score of dev set: {0:.2f}".format(f1*100))
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': train_loss,
                'Dev Loss': dev_loss,
                'Dev F1': f1,
                'Training Time': training_time,
            }
        )

    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    model = torch.load(params['model_location']+"/best_model")
    f1, test_loss = model.test_step(test_citation_dataloader, criterion)
    print(f'F1 test set: {f1}')
    pretty_print_stats(training_stats)
    f.write(json.dumps(training_stats))
    f.write("\n")
    test_set = {'test_F1': f1 , 'test_loss': test_loss }
    f.write(json.dumps(test_set))
    f.write("\n")
    f.close()