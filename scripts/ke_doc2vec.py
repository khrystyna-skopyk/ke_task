import json
import sys
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.parse.corenlp import CoreNLPParser
from nltk import sent_tokenize
import pickle

parser = CoreNLPParser()


def get_np_keyphrases(abstract_sents):
    """
    With the help of CoreNLP constituency parser, get all noun phrase constituents as potential keyphrases.
    :param abstract_sents: list of strings, list of sentences
    :return: keyphrases, a list of lists of strings(tokenized keyphrases)
    """
    post_parses = [list(s_parse) for s_parse in list(parser.raw_parse_sents(abstract_sents))]
    keyphrases = []
    for sent_parse in post_parses:
        for tree in sent_parse[0]:
            for subtree in tree.subtrees():
                if subtree.label() == "NP" and len(subtree.leaves()) < 6 and not subtree.leaves() in keyphrases:
                    keyphrases.append(subtree.leaves())
    return keyphrases


def get_tagged_data(in_file, tagged_file):
    """
    Creates list of TaggedDocument objects (abstracts and keyphrases) for training with doc2vec.
    :param in_file: string, path to the file with training abstracts
    :param tagged_file: string, name of the file where the list of tagged docs should be saved
    :return: None
    """
    tagged_data = []
    with open(in_file) as json_file:
        data = json.load(json_file)
        for key in data.keys():
            print("On key: ", key)
            abstract = data[key]["abstract"]
            tokens = list(parser.tokenize(abstract))
            abstract_sents = sent_tokenize(abstract)
            if len(abstract_sents) < 200:
                tagged_data.append(TaggedDocument(words=tokens, tags=[key]))
                keyphrases = get_np_keyphrases(abstract_sents)
                for e, keyphrase in enumerate(keyphrases):
                    tagged_data.append(TaggedDocument(words=keyphrase, tags=[key + "-" + str(e)]))
    with open(tagged_file, 'wb') as tf:
        pickle.dump(tagged_data, tf)
    print("Tagged data dumped.")


def train_model(max_epochs, vec_size, alpha, tagged_data_file):
    """
    Trains a doc2vec model with tagged data from tagged_data_file.
    :param max_epochs: int
    :param vec_size: int
    :param alpha: int
    :param tagged_data_file: string, name of the file where the list of tagged docs is
    :return: None, saves the model
    """
    tagged_data = []
    with open(tagged_data_file, "rb") as openfile:
        while True:
            try:
                tagged_data.append(pickle.load(openfile))
            except EOFError:
                break
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    model.save("d2v.model")
    print("Model Saved")


def get_keyphrases(model, abstract):
    """
    Gets keyphrases to the abstract sorted in the decreasing order.
    :param model: string, path to the doc2vec model
    :param abstract: string,
    :param original_keywords: list of strings, optional
    :return: !!!!!!!!!!!!
    """
    model = Doc2Vec.load(model)
    tokens = list(parser.tokenize(abstract))
    keyphrases = get_np_keyphrases(sent_tokenize(abstract))
    resulting_similarities = []
    for keyphrase in keyphrases:
        resulting_similarities.append((" ".join(keyphrase),
                                      model.docvecs.similarity_unseen_docs(model, tokens, keyphrase)))
    resulting_similarities.sort(key=lambda x: x[1], reverse=True)
    return resulting_similarities


if __name__ == "__main__":
    # get_tagged_data("../data/hulth_ke_train.json", "../data/hulth_ke_train_tagged")
    #
    # max_epochs = 100
    # vec_size = 20
    # alpha = 0.025
    # train_model(max_epochs, vec_size, alpha, "../data/hulth_ke_train_tagged")

    modelname = "../d2v.model"

    with open(sys.argv[1]) as abstract_file:
        print("{: <30} {: >20}".format("KEYPHRASE", "ITS SIMILARITY SCORE"))
        for k,s in get_keyphrases(modelname, abstract_file.read()):
            print("{: <30} : {: >20}".format(k, s))