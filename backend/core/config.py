import os
from pathlib import Path

class Settings:
    # Пути к данным
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / 'core' / 'dataset.json'
    MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
    
    # NLP Модели
    BERT_MODEL_NAME = 'cointegrated/rubert-tiny2'  # Легкая русскоязычная модель
    SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    
    # Параметры обучения
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_LENGTH = 128  # Для токенизатора BERT
    
    # NATASHA
    ENABLE_NATASHA = True
    
    # Features
    TEXT_EMBEDDING_DIM = 312  # Для rubert-tiny2
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
    
    # Entity Extraction
    ENTITY_EXTRACTOR_TYPE = 'simple'
    ENABLE_ENTITY_EXTRACTION = True

    # Обновляем список категориальных признаков согласно JSON структуре
    CATEGORICAL_FEATURES = [
        'project_type', 
        'project_scale',  # бывший 'scale'
        'industry',       # бывший 'domain' 
        'budget', 
        'team_size', 
        'complexity'
    ]

settings = Settings()

@classmethod
def create_directories(cls):
    """Создание необходимых директорий"""
    cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

# # Вызов при инициализации
# Settings.create_directories()