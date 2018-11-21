# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd


def figer_freebase_map():
    
    #preparing freebase_entities table
    col = ['Word', 'Type', 'Num']
    freebase_types = pd.read_csv('textrazor-freebase-types-en.tsv', sep='\t', header= None)
    freebase_types.columns = col
    temp = freebase_types["Type"].str.split("/", n = 2, expand = True) 
    freebase_types["Type1"] = temp[1]
    freebase_types["Type2"] = temp[2]
    freebase_types = freebase_types.drop(['Type'], axis=1)
    
    #preparing the figer_list from Ousia paper
    figer_entities = pd.read_csv('figer_list.csv')
    figer_list = figer_entities['Type'].tolist()
    
    #dropping rows with type values not present in figer_list. If either type 1 or type 2 is present, we won't drop the row
    temp = freebase_types
    for index, row in freebase_types.iterrows():
        if row['Type1'] not in figer_list and row['Type2'] not in figer_list: 
        #print('atleast coming here')
            temp = temp.drop(temp.index[index]) 
        
    return temp

if __name__ == '__main__':
    figer_freebase_map()
