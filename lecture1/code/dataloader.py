import pandas as pd
from pandas import DataFrame
from typing import Text
from utils import preprocess_text
from tqdm import tqdm
tqdm.pandas()

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Loads dataset from file into memory
"""

class AmazonProductDataloader:
    def __init__(self, file_path, text_field, id_field):
        self.text_field = text_field
        self.id_field = id_field
        self.preprocessed_text_field = f"preprocessed_{text_field}"
        self.dataset = self._load_data(file_path, text_field, id_field)
        self.dataset.fillna("", inplace=True)
        logger.info("preprocessing texts")
        self.dataset[self.preprocessed_text_field] =  self.dataset[text_field].progress_apply(preprocess_text)

    def _load_data(self, file_path:Text, text_field, id_field) -> DataFrame:
        dataset = pd.read_csv(file_path, low_memory=False)
        return dataset[[text_field, id_field]]
