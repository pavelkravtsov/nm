from pymystem3 import Mystem
import logging
import re

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.StreamHandler()])

logging.info("Loading mystem")
m = Mystem()
logging.info("Loaded mystem")


def parse_gr(gr):
    options = re.search('\(([^\)]*)\)', gr, re.IGNORECASE)
    if options:
        title = options.group(1)
        for stuff in title.split('|'):
            yield gr.replace("(" + title + ")", stuff)
    else:
        yield gr

writer = open("pairs.tsv", "w+")

with open("test.txt", "r") as input_file:
    logging.info("file opened")

    for line in input_file:
        for w in m.analyze(line):
            if 'analysis' in w:
                for item in w['analysis']:
                    for gramm_info in parse_gr(item['gr']):
                        # print item['gr'], '->', gramm_info
                        writer.write("\t".join(
                            [gramm_info,
                             item['lex'],
                             w['text'].lower()]).encode("utf-8") + "\n")

writer.close()

dict = {}

for line in open("pairs.uniq.tsv", "r+"):
    if line.strip():
        desc, normal, form = line.strip().split("\t")
        if not desc in dict:
            dict[desc] = []
        dict[desc].append((normal, form))

logging.info("Pairs acquired")

writer = open("relations.pairs.tsv", "w+")

for desc in dict:
    for p0 in dict[desc]:
        for p1 in dict[desc]:
            if not p0 == p1:
                writer.write("\t".join([p0[0], p0[1], p1[0], p1[1]]) + "\n")

writer.close()

logging.info("Relations pairs acquired")

