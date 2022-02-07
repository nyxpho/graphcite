import json
import sys
from sentence_transformers import SentenceTransformer
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
import requests
import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

intent2intScicite = {
    "background": 0,
    "method":1,
    "result":2,
    "author":3,
    "venue":4
}

def read_scicite(path, nodes, edges, nodeid2index, authors, venues):

    papers = {}
    rb = open("scicite/small_papers_info.jsonl", "r", encoding="utf-8")
    for line in rb.readlines():
        paper = json.loads(line.strip())
        if 'paperId' not in paper:
            continue
        papers[paper["paperId"]] = {'title':paper['title'], 'authors':paper['authors'], 'venue': paper['venue']}
    rb.close()


    rb = open(path, "r", encoding="utf-8")

    for line in rb.readlines():
        citation = json.loads(line.strip())
        if citation["citingPaperId"] not in nodeid2index:
            nodeid2index[citation["citingPaperId"]] = len(nodes[0])
            nodes[0].append(nodeid2index[citation["citingPaperId"]])
            if citation["citingPaperId"] not in papers:
                nodes[1].append("")
            else:
                nodes[1].append(papers[citation["citingPaperId"]]['title'])

            if citation["citingPaperId"] in papers and authors:
                for author in papers[citation["citingPaperId"]]['authors']:
                    if 'authorId' not in author or author['authorId'] is None:
                        continue
                    if 'author' + author['authorId'] not in nodeid2index:
                        nodeid2index['author' + author['authorId']] = len(nodes[0])
                        nodes[0].append(nodeid2index['author' + author['authorId']])
                        nodes[1].append("")
                    edges[0].append([nodeid2index['author' + author['authorId']], nodeid2index[citation["citingPaperId"]]])
                    edges[1].append("")
                    edges[2].append(intent2intScicite["author"])

            if citation["citingPaperId"] in papers and venues and len(papers[citation["citingPaperId"]]['venue']):
                if papers[citation["citingPaperId"]]['venue'] not in nodeid2index:
                    nodeid2index[papers[citation["citingPaperId"]]['venue']] = len(nodes[0])
                    nodes[0].append(nodeid2index[papers[citation["citingPaperId"]]['venue']])
                    nodes[1].append("")
                edges[0].append([nodeid2index[papers[citation["citingPaperId"]]['venue']], nodeid2index[citation["citingPaperId"]]])
                edges[1].append("")
                edges[2].append(intent2intScicite["venue"])
        if citation["citedPaperId"] not in nodeid2index:
            nodeid2index[citation["citedPaperId"]] = len(nodes[0])
            nodes[0].append(nodeid2index[citation["citedPaperId"]])
            if citation["citedPaperId"] not in papers:
                nodes[1].append("")
            else:
                nodes[1].append(papers[citation["citedPaperId"]]['title'])
            if citation["citedPaperId"] in papers and authors:
                for author in papers[citation["citedPaperId"]]['authors']:
                    if 'authorId' not in author or author['authorId'] is None:
                        continue
                    if 'author' + author['authorId'] not in nodeid2index:
                        nodeid2index['author' + author['authorId']] = len(nodes[0])
                        nodes[0].append(nodeid2index['author' + author['authorId']])
                        nodes[1].append("")
                    edges[0].append([nodeid2index['author' + author['authorId']], nodeid2index[citation["citedPaperId"]]])
                    edges[1].append("")
                    edges[2].append(intent2intScicite["author"])

            if citation["citedPaperId"] in papers and venues and len(papers[citation["citedPaperId"]]['venue']):
                if papers[citation["citedPaperId"]]['venue'] not in nodeid2index:
                    nodeid2index[papers[citation["citedPaperId"]]['venue']] = len(nodes[0])
                    nodes[0].append(nodeid2index[papers[citation["citedPaperId"]]['venue']])
                    nodes[1].append("")
                edges[0].append([nodeid2index[papers[citation["citedPaperId"]]['venue']], nodeid2index[citation["citedPaperId"]]])
                edges[1].append("")
                edges[2].append(intent2intScicite["venue"])
        # To remove the lines below if we do not want authors


        edges[0].append([nodeid2index[citation["citingPaperId"]], nodeid2index[citation["citedPaperId"]]])
        edges[1].append(citation["string"])
        edges[2].append(intent2intScicite[citation["label"]])

    rb.close()

def create_data_object(nodes, edges, lang_model):
    edge_index = torch.tensor(edges[0], dtype=torch.long, device=device)
    x_array = lang_model.encode(nodes[1])
    x = torch.tensor(x_array, dtype=torch.float, device=device)

    edge_attr = lang_model.encode(edges[1])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=device)
    y = torch.tensor(edges[2], dtype=torch.long, device=device)
    print("number of nodes, x :" + str(len(x)))
    print("number of edges intents, y:" + str(len(y)))
    print("number of edge citations:" + str(len(edge_attr)))
    print("number of edges:" + str(len(edge_index)))
    data = Data(x=x, y=y, edge_attr=edge_attr, edge_index=edge_index.t().contiguous())
    transform = T.ToSparseTensor()
    data = data if transform is None else transform(data)
    return data


def query_semantic_scholar(paper_id):
    response = requests.get('https://api.semanticscholar.org/v1/paper/'+paper_id)
    return response.json()

