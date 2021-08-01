from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Text, Union

def preprocess_text(text: Text, tokens_only=False) -> Union[Text, List]:
    result = None
    # tokenize text
    preproccesed_tokens = word_tokenize(text.strip())
    # stemming tokens to roots
    stemmer = PorterStemmer()
    preproccesed_tokens = [stemmer.stem(token) for token in preproccesed_tokens]
    # reomove stopwords
    preproccesed_tokens = [token for token in preproccesed_tokens if token not in stopwords.words('english')]
    if tokens_only:
        result =  preproccesed_tokens
    else:
        result =  " ".join(preproccesed_tokens)
        
    return result