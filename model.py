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
                             w['text']]).encode("utf-8") + "\n")

writer.close()
