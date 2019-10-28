import json
import random
from ke_doc2vec import get_keyphrases

modelname = "scripts/d2v.model"

def compare_ke_from_random_abstracts():
    with open("data/hulth_ke_test.json") as json_file:
        data = json.load(json_file)
        for key in random.sample(data.keys(), 10):
            print("On abstract", key)
            print("ORIGINAL KEYPHRASES: ", data[key]["keywords"])
            print("{: <50} {: >20}".format("KEYPHRASE", "ITS SIMILARITY SCORE"))
            for k, s in get_keyphrases(modelname, data[key]["abstract"]):
                print("{: <50} : {: >20}".format(k, s))
            print("\n")


if __name__ == "__main__":
    compare_ke_from_random_abstracts()