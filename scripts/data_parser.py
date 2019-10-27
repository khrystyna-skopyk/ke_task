import re
import os
from collections import defaultdict
import spacy
import json

MAIN_DATA = defaultdict(dict)
nlp = spacy.load("en_core_web_sm")

def parse_hulth(dir_path, out_file):
    """
    Parses dataset into more workable format, uses some heuristics to clean data
    :param dir_path: string, path to the directory with files to parse
    :param out_file: string, name of the new file where to save parsed data
    :return: None, saves data into a json file
    """
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            #print(filename)
            with open(os.path.join(dirpath, filename)) as my_f:
                idx = filename.split(".")[0]
                if filename.endswith("abstr"):
                    abstract = my_f.read().splitlines()
                    if abstract[1].startswith("\t"):
                        abstract = " ".join(abstract[2:])
                    else:
                        abstract = " ".join(abstract[1:])
                    MAIN_DATA[idx]["abstract"] = re.sub("\s+", " ", abstract)
                elif filename.endswith("contr"):
                    keywrds = re.sub("\s+", " ", my_f.read()).strip().split("; ")
                    if MAIN_DATA[idx].get("keywords"):
                        MAIN_DATA[idx]["keywords"] += keywrds
                    else:
                        MAIN_DATA[idx]["keywords"] = keywrds
    to_delete = []
    for key in MAIN_DATA.keys():
        abstract = MAIN_DATA[key]["abstract"]
        old_keywords = MAIN_DATA[key]["keywords"]
        doc = nlp(abstract)
        tokens = [token.text for token in doc]
        if len(tokens) < 60:
            to_delete.append(key)
        if not key in to_delete:
            new_keywords = []
            for keywrd in old_keywords:
                if keywrd in abstract and not keywrd in new_keywords:
                    new_keywords.append(keywrd)
            if len(new_keywords) > 1:
                MAIN_DATA[key]["keywords"] = new_keywords
            else:
                to_delete.append(key)
    print(len(to_delete))
    for k in to_delete:
        del MAIN_DATA[k]
    with open(out_file, 'w') as outfile:
        json.dump(MAIN_DATA, outfile)


def parse_journals(dir_path, out_file):
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            #print(filename)
            with open(os.path.join(dirpath, filename)) as my_f:
                idx = filename.split(".")[0]
                if filename.endswith("abstr"):
                    abstract = my_f.read().splitlines()
                    if abstract[1].startswith("\t"):
                        abstract = " ".join(abstract[2:])
                    else:
                        abstract = " ".join(abstract[1:])
                    MAIN_DATA[idx]["abstract"] = re.sub("\s+", " ", abstract)

if __name__ == "__main__":
    # parse_hulth('../data/Hulth2003/Training', '../data/hulth_ke_train.json')      # len = 1251
    parse_hulth('../data/Hulth2003/Testing', '../data/hulth_ke_test.json')          # len = 407