from dataloader import AmazonProductDataloader
from inverted_index import InvertedIndex
from utils import preprocess_text
import numpy as np

class TFIDFSearchRelevance:
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index
        self.total_documents = inverted_index.dataloader.dataset.shape[0]

    def score_query(self, query, k=3):
        scores = {}
        preprocessed_query = preprocess_text(query, tokens_only=True)
        for query_term in preprocessed_query:
            if query_term in self.inverted_index.term_dictionary:
                term_frequencies = self.inverted_index.term_dictionary[query_term]
                for term_frequency in term_frequencies:
                    if term_frequency["document"] not in scores:
                        scores[term_frequency["document"]] = 0
                    scores[term_frequency["document"]] += self.tdidf_score(term_frequency["frequency"], len(term_frequency))

        scores = dict(sorted(sorted(scores.items(), key=lambda x: x[1])))
        if k > len(scores.keys()):
            k = len(scores.keys())
        return list(scores.keys())[:k] ## returns top k documents

    def tdidf_score(self, term_frequency, document_frequency):
        tf = np.log(1 + term_frequency)
        idf = np.log(self.total_documents/ document_frequency)
        return tf * idf

