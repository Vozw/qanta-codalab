import pandas as pd
import csv

def figer_freebase_map():
    freebase_types = pd.read_csv('textrazor-freebase-types-en.tsv', sep='\t', header= None)

    figer_entities = pd.read_csv('data/figer_list.tsv')
    figer_list = figer_entities['Type'].tolist()

    with open("slim-freebase-types.csv", "wb") as out_file:
        for row in freebase_types.iterrows():
            if row[1][1].split('/')[-1] in figer_list:
                out = "\t".join(row[1][:-1]) + '\n'
                out_file.write(out.encode("utf-8"))

if __name__ == '__main__':
    figer_freebase_map()
