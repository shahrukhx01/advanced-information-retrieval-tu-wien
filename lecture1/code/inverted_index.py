from dataloader import AmazonProductDataloader
from collections import Counter
from nltk.tokenize import word_tokenize

class InvertedIndex:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._create_term_dictionary()
    
    def _create_term_dictionary(self):
        self.term_dictionary = {}
        for idx, document in self.dataloader.dataset.iterrows():
            tokens = word_tokenize(document[self.dataloader.preprocessed_text_field])
            unique_terms = set(tokens)
            term_counts = Counter(tokens)
            for term in unique_terms:
                term = term.strip()
                if term:
                    if term in self.term_dictionary:
                        self.term_dictionary[term].append({
                            "document": document[self.dataloader.id_field],
                            "frequency": term_counts[term],
                            "document_length": len(tokens)
                        })
                    else:
                        self.term_dictionary[term] = [{
                            "document": document[self.dataloader.id_field],
                            "frequency": term_counts[term],
                            "document_length": len(tokens)
                        }]
