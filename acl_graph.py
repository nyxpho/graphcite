import json
import sys
from sentence_transformers import SentenceTransformer
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
import requests
import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

intent2intACL = {
    "Background": 0,
    "Uses":1,
    "Future":2,
    "CompareOrContrast":3,
    "Motivation": 4,
    "Extends": 5,
    "author":6,
    "venue":7
}


def read_acl(path, nodes, edges, node2nodeid, authors, venues):

    papers = {}
    rb = open("acl-arc/papers_info.jsonl", "r", encoding="utf-8")
    for line in rb.readlines():
        paper = json.loads(line.strip())
        if 'paperId' not in paper:
            continue
        papers[paper["old_id"]] = {'title':paper['title'], 'authors':paper['authors'], 'venue': paper['venue']}
    rb.close()
    print(len(papers))
    rb = open(path, "r", encoding="utf-8")

    for line in rb.readlines():
        citation = json.loads(line.strip())
        if citation["citing_paper_id"] not in node2nodeid:
            node2nodeid[citation["citing_paper_id"]] = len(nodes[0])
            nodes[0].append(node2nodeid[citation["citing_paper_id"]])
            if citation["citing_paper_id"] in papers:
                nodes[1].append(papers[citation["citing_paper_id"]]['title'])
            else:
                nodes[1].append(citation["citing_paper_title"])

            if citation["citing_paper_id"] in papers and authors:
                for author in papers[citation["citing_paper_id"]]['authors']:
                    if 'authorId' not in author or author['authorId'] is None:
                        continue
                    if 'author' + author['authorId'] not in node2nodeid:
                        node2nodeid['author' + author['authorId']] = len(nodes[0])
                        nodes[0].append(node2nodeid['author' + author['authorId']])
                        nodes[1].append("")
                    edges[0].append([node2nodeid['author' + author['authorId']], node2nodeid[citation["citing_paper_id"]]])
                    edges[1].append("")
                    edges[2].append(intent2intACL["author"])
            if citation["citing_paper_id"] in papers and venues and len(papers[citation["citing_paper_id"]]['venue']):
                if papers[citation["citing_paper_id"]]['venue'] not in node2nodeid:
                    node2nodeid[papers[citation["citing_paper_id"]]['venue']] = len(nodes[0])
                    nodes[0].append(node2nodeid[papers[citation["citing_paper_id"]]['venue']])
                    nodes[1].append("")
                edges[0].append([node2nodeid[papers[citation["citing_paper_id"]]['venue']], node2nodeid[citation["citing_paper_id"]]])
                edges[1].append("")
                edges[2].append(intent2intACL["venue"])


        if citation["cited_paper_id"] not in node2nodeid:
            node2nodeid[citation["cited_paper_id"]] = len(nodes[0])
            nodes[0].append(node2nodeid[citation["cited_paper_id"]])
            if citation["cited_paper_id"] in papers:
                nodes[1].append(papers[citation["cited_paper_id"]]['title'])
            else:
                nodes[1].append(citation["cited_paper_title"])

            if citation["cited_paper_id"] in papers and authors:
                for author in papers[citation["cited_paper_id"]]['authors']:
                    if 'authorId' not in author or author['authorId'] is None:
                        continue
                    if 'author' + author['authorId'] not in node2nodeid:
                        node2nodeid['author' + author['authorId']] = len(nodes[0])
                        nodes[0].append(node2nodeid['author' + author['authorId']])
                        nodes[1].append("")
                    edges[0].append([node2nodeid['author' + author['authorId']], node2nodeid[citation["cited_paper_id"]]])
                    edges[1].append("")
                    edges[2].append(intent2intACL["author"])

            if citation["cited_paper_id"] in papers and venues and len(papers[citation["cited_paper_id"]]['venue']):
                if papers[citation["cited_paper_id"]]['venue'] not in node2nodeid:
                    node2nodeid[papers[citation["cited_paper_id"]]['venue']] = len(nodes[0])
                    nodes[0].append(node2nodeid[papers[citation["cited_paper_id"]]['venue']])
                    nodes[1].append("")
                edges[0].append([node2nodeid[papers[citation["cited_paper_id"]]['venue']], node2nodeid[citation["cited_paper_id"]]])
                edges[1].append("")
                edges[2].append(intent2intACL["venue"])
        edges[0].append([node2nodeid[citation["citing_paper_id"]], node2nodeid[citation["cited_paper_id"]]])
        edges[1].append(citation["text"])
        edges[2].append(intent2intACL[citation["intent"]])

    rb.close()


def create_data_object(nodes, edges, lang_model):
    edge_index = torch.tensor(edges[0], dtype=torch.long, device=device)
    x_array = lang_model.encode(nodes[1])
    x = torch.tensor(x_array, dtype=torch.float, device=device)

    edge_attr = lang_model.encode(edges[1])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=device)
    y = torch.tensor(edges[2], dtype=torch.long, device=device)

    data = Data(x=x, y=y, edge_attr=edge_attr, edge_index=edge_index.t().contiguous())
    transform = T.ToSparseTensor()
    data = data if transform is None else transform(data)
    return data

def query_semantic_scholar(paper_id):
    response = requests.get('https://api.semanticscholar.org/graph/v1/paper/ACL:'+paper_id+'?fields=title,abstract,authors,venue,fieldsOfStudy')
    return response.json()

