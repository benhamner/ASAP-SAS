#! /usr/bin/env python

import csv

def get_essay_sets(filename = "../Data/train_rel_2.tsv"):
    return sorted(set(x["EssaySet"] for x in essay_reader(filename)))

def essays_by_set(essay_set, filename = "../Data/train_rel_2.tsv"):
    return (essay for essay in essay_reader(filename)
            if essay["EssaySet"]==essay_set)

def essay_reader(filename = "../Data/train_rel_2.tsv"):
    f = open(filename)
    reader = csv.reader(f, delimiter = '\t')
    header = reader.next()
    for row in reader:
        elem = dict(zip(header, row))
        for col in ["EssayScore", "Id", "EssaySet"]:
            if col in elem:
                elem[col] = float(elem[col])
        yield elem
    f.close()