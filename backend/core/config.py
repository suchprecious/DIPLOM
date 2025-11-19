import os
from pathlib import Path

class Settings:
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / 'core' / 'dataset.json'
    MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
    

    BERT_MODEL_NAME = 'cointegrated/rubert-tiny2'  
    SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    

    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_LENGTH = 128

    ENABLE_NATASHA = True
    
    TEXT_EMBEDDING_DIM = 312  
    CATEGORICAL_FEATURES = ['project_type', 'project_scale', 'industry', 'subdomain', 'team_size', 'complexity']

    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 20
    
    LGB_N_ESTIMATORS = 100
    LGB_MAX_DEPTH = 15
    
    XGB_N_ESTIMATORS = 100
    XGB_MAX_DEPTH = 15
    
    GB_N_ESTIMATORS = 100
    GB_MAX_DEPTH = 10

    MIN_TECHNOLOGY_FREQUENCY = 5
    CONFIDENCE_THRESHOLD = 0.3

    ENTITY_EXTRACTOR_TYPE = 'simple'
    ENABLE_ENTITY_EXTRACTION = True


    CATEGORICAL_FEATURES = [
        'project_type', 
        'project_scale',  
        'industry',       
        'budget', 
        'team_size', 
        'complexity'
    ]

settings = Settings()

@classmethod
def create_directories(cls):
    """Создание необходимых директорий"""
    cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Settings.create_directories()