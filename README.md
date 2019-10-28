# Keyword extraction task
This repo contains the solution for keyword extraction task: doc2vec model trained on [Inspect data](https://github.com/gfigueroa/nlp-datasets/tree/master/Abstracts%20and%20Keywords/Hulth2003). The idea for the solution was taken from [this paper](https://arxiv.org/pdf/1801.04470.pdf). 

To test the model, do the following:
1. clone this repo to your local machine
2. go the the repo's root and build the docker image: `docker-compose build` (will take a few mins)
3. choose the abstract you'd like to extract keywords from, save it into the file in the root of this project. Then run your file: 
`docker-compose run --rm app bash -c "python scripts/ke_doc2vec.py myfile.txt"`. You can also run test abstracts saved in the root (`test_abstract_1.txt` or `test_abstract_2.txt`)
4. Since the model was trained on a very specific data set (mainly technical abstracts), it's better to test the model with its test set abstracts (`data/hulth_ke_test.json`). To extract keyphrases from 10 random abstracts from the test set, run: `docker-compose run --rm app bash -c "python scripts/compare_keyphrases.py"`. 