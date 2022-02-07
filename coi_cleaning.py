import sys
from flair.data import Sentence
from flair.models import SequenceTagger


# load the NER tagger
tagger = SequenceTagger.load('ner')

rb = open(sys.argv[1], 'r')
wb = open(sys.argv[2], "w")
corpus = []
corpus_info = dict()
set_tags = dict()
for line in rb.readlines():
    splits = line.strip().rsplit("\t",1)
    if len(splits[0]) ==0 or len(splits[1]) ==1:
        continue
    freq = int(splits[1][0:len(splits[1])-1])
    coi = Sentence(splits[0])
    tagger.predict(coi)
    if len(coi.get_spans('ner')):
       continue
    wb.write(splits[0] + "\t" + str(freq)+"\n")

wb.close()
rb.close()
