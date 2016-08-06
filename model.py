from pymystem3 import Mystem
import json
import logging

print "Loading mystem"
m = Mystem()

print "Loaded mystem"


writer = open("pairs.tsv", "w+")

with open("test.txt", "r") as input_file:
    print "file opened"
    for line in input_file:
        for w in m.analyze(line):
            
            if 'analysis' in w:
                for item in w['analysis']:
                    writer.write("\t".join([item['gr'], item['lex'], w['text']]) + "\n")

writer.close()
