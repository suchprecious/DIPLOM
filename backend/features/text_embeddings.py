# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer
# from core.config import settings

# class TextEmbeddingGenerator:
#     def __init__(self):
#         self.bert_model_name = settings.BERT_MODEL_NAME
#         self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
#         self.model = AutoModel.from_pretrained(self.bert_model_name)
#         self.sentence_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
    
#     def get_bert_embeddings(self, texts: list) -> torch.Tensor:
#         # Генерация контекстных эмбеддингов через BERT
#         inputs = self.tokenizer(texts, padding=True, truncation=True, 
#                               max_length=settings.MAX_LENGTH, return_tensors='pt')
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] token
    
#     def get_sentence_embeddings(self, texts: list) -> np.ndarray:
#         # Генерация эмбеддингов через SentenceTransformer
#         return self.sentence_model.encode(texts)

import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union, Dict, Any
import os
from pathlib import Path
from core.config import settings

logger = logging.getLogger(__name__)

class TextEmbeddingGenerator:
    """
    Генератор векторных представлений текста
    Поддерживает BERT и SentenceTransformer с кэшированием и обработкой ошибок
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.bert_model_name = settings.BERT_MODEL_NAME
        self.sentence_model_name = settings.SENTENCE_TRANSFORMER_MODEL
        self.cache_dir = cache_dir or Path("models/pretrained")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация моделей
        self.bert_tokenizer = None
        self.bert_model = None
        self.sentence_model = None
        self.models_loaded = False
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Инициализация моделей с обработкой ошибок и кэшированием"""
        # BERT модель
        try:
            logger.info(f"Загрузка BERT модели: {self.bert_model_name}")
            
            # Пытаемся загрузить из кэша
            bert_cache_path = self.cache_dir / self.bert_model_name.replace('/', '_')
            if bert_cache_path.exists():
                logger.info(f"Загрузка BERT из кэша: {bert_cache_path}")
                self.bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_cache_path))
                self.bert_model = AutoModel.from_pretrained(str(bert_cache_path))
            else:
                logger.info(f"Загрузка BERT из HuggingFace Hub: {self.bert_model_name}")
                self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
                # Сохраняем в кэш
                self.bert_model.save_pretrained(str(bert_cache_path))
                self.bert_tokenizer.save_pretrained(str(bert_cache_path))
            
            # Переводим в режим оценки
            self.bert_model.eval()
            logger.info("✅ BERT модель успешно загружена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки BERT модели: {e}")
            self.bert_model = None
        
        # SentenceTransformer модель
        try:
            logger.info(f"Загрузка SentenceTransformer модели: {self.sentence_model_name}")
            
            sentence_cache_path = self.cache_dir / self.sentence_model_name.replace('/', '_')
            if sentence_cache_path.exists():
                logger.info(f"Загрузка SentenceTransformer из кэша: {sentence_cache_path}")
                self.sentence_model = SentenceTransformer(str(sentence_cache_path))
            else:
                logger.info(f"Загрузка SentenceTransformer из Hub: {self.sentence_model_name}")
                self.sentence_model = SentenceTransformer(self.sentence_model_name)
                # Сохраняем в кэш
                self.sentence_model.save(str(sentence_cache_path))
            
            logger.info("✅ SentenceTransformer модель успешно загружена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки SentenceTransformer модели: {e}")
            self.sentence_model = None
        
        self.models_loaded = (self.bert_model is not None) or (self.sentence_model is not None)
        
        if not self.models_loaded:
            logger.error("❌ Ни одна модель не загружена! Эмбеддинги недоступны.")
        else:
            logger.info("✅ Модели эмбеддингов готовы к работе")
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Предобработка текстов перед генерацией эмбеддингов"""
        if not texts:
            return []
        
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            # Базовая очистка
            text = text.strip()
            if not text:
                text = "Нет описания"
            
            # Ограничение длины для BERT
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            processed_texts.append(text)
        
        return processed_texts
    
    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Генерация контекстных эмбеддингов через BERT
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            Матрица эмбеддингов [n_samples, embedding_dim]
        """
        if self.bert_model is None:
            logger.error("BERT модель не загружена")
            return self._get_fallback_embeddings(texts, 312)  # Размерность rubert-tiny2
        
        try:
            # Предобработка текстов
            processed_texts = self._preprocess_texts(texts)
            if not processed_texts:
                return self._get_fallback_embeddings(texts, 312)
            
            # Токенизация
            inputs = self.bert_tokenizer(
                processed_texts, 
                padding=True, 
                truncation=True, 
                max_length=settings.MAX_LENGTH, 
                return_tensors='pt'
            )
            
            # Генерация эмбеддингов
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Используем [CLS] token как представление всего текста
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            logger.info(f"✅ Сгенерированы BERT эмбеддинги для {len(texts)} текстов. Размер: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации BERT эмбеддингов: {e}")
            return self._get_fallback_embeddings(texts, 312)
    
    def get_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Генерация эмбеддингов через SentenceTransformer
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            Матрица эмбеддингов [n_samples, embedding_dim]
        """
        if self.sentence_model is None:
            logger.error("SentenceTransformer модель не загружена")
            return self._get_fallback_embeddings(texts, 384)  # Размерность multilingual-MiniLM
        
        try:
            # Предобработка текстов
            processed_texts = self._preprocess_texts(texts)
            if not processed_texts:
                return self._get_fallback_embeddings(texts, 384)
            
            # Генерация эмбеддингов
            embeddings = self.sentence_model.encode(
                processed_texts, 
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=32
            )
            
            logger.info(f"✅ Сгенерированы SentenceTransformer эмбеддинги для {len(texts)} текстов. Размер: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации SentenceTransformer эмбеддингов: {e}")
            return self._get_fallback_embeddings(texts, 384)
    
    def _get_fallback_embeddings(self, texts: List[str], embedding_dim: int) -> np.ndarray:
        """
        Резервные эмбеддинги в случае ошибки
        
        Args:
            texts: Список текстов
            embedding_dim: Размерность эмбеддингов
            
        Returns:
            Матрица нулевых эмбеддингов
        """
        n_texts = len(texts) if texts else 1
        fallback_embeddings = np.zeros((n_texts, embedding_dim))
        
        logger.warning(f"Используются резервные эмбеддинги: {fallback_embeddings.shape}")
        return fallback_embeddings
    
    def get_combined_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Комбинирование эмбеддингов от обеих моделей
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            Объединенная матрица эмбеддингов
        """
        bert_embeddings = self.get_bert_embeddings(texts)
        sentence_embeddings = self.get_sentence_embeddings(texts)
        
        # Проверяем размерности
        if bert_embeddings.shape[0] != sentence_embeddings.shape[0]:
            logger.error("Размерности эмбеддингов не совпадают, используем только BERT")
            return bert_embeddings
        
        # Объединяем по горизонтали
        combined = np.hstack([bert_embeddings, sentence_embeddings])
        logger.info(f"✅ Созданы комбинированные эмбеддинги. Размер: {combined.shape}")
        
        return combined
    
    def get_embedding_dimensions(self) -> Dict[str, Any]:
        """Получение размерностей эмбеддингов"""
        dimensions = {}
        
        if self.bert_model:
            dimensions['bert'] = settings.TEXT_EMBEDDING_DIM  # 312 для rubert-tiny2
        
        if self.sentence_model:
            try:
                dimensions['sentence_transformer'] = self.sentence_model.get_sentence_embedding_dimension()
            except:
                dimensions['sentence_transformer'] = 384  # Размерность по умолчанию
        
        dimensions['combined'] = sum(dimensions.values())
        
        logger.info(f"Размерности эмбеддингов: {dimensions}")
        return dimensions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о загруженных моделях"""
        info = {
            "models_loaded": self.models_loaded,
            "bert_loaded": self.bert_model is not None,
            "sentence_transformer_loaded": self.sentence_model is not None,
            "embedding_dimensions": self.get_embedding_dimensions()
        }
        
        return info


# Упрощенная версия для тестов (без зависимостей от трансформеров)
class SimpleEmbeddingGenerator:
    """
    Упрощенный генератор эмбеддингов для тестирования
    """
    
    def __init__(self, embedding_dim: int = 100):
        self.embedding_dim = embedding_dim
        logger.info(f"Используется упрощенный генератор эмбеддингов (dim={embedding_dim})")
    
    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Упрощенные BERT-подобные эмбеддинги"""
        n_texts = len(texts) if texts else 1
        embeddings = np.random.randn(n_texts, self.embedding_dim) * 0.1
        logger.info(f"Сгенерированы упрощенные BERT эмбеддинги: {embeddings.shape}")
        return embeddings
    
    def get_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """Упрощенные SentenceTransformer-подобные эмбеддинги"""
        n_texts = len(texts) if texts else 1
        embeddings = np.random.randn(n_texts, self.embedding_dim) * 0.1 + 0.5
        logger.info(f"Сгенерированы упрощенные SentenceTransformer эмбеддинги: {embeddings.shape}")
        return embeddings
    
    def get_combined_embeddings(self, texts: List[str]) -> np.ndarray:
        """Комбинированные упрощенные эмбеддинги"""
        bert_emb = self.get_bert_embeddings(texts)
        sentence_emb = self.get_sentence_embeddings(texts)
        combined = np.hstack([bert_emb, sentence_emb])
        logger.info(f"Сгенерированы комбинированные упрощенные эмбеддинги: {combined.shape}")
        return combined


# Создание глобального экземпляра с обработкой ошибок
try:
    embedding_generator = TextEmbeddingGenerator()
    if not embedding_generator.models_loaded:
        logger.warning("Основные модели не загружены, используем упрощенный генератор")
        embedding_generator = SimpleEmbeddingGenerator(embedding_dim=100)
except Exception as e:
    logger.error(f"Не удалось создать TextEmbeddingGenerator: {e}")
    embedding_generator = SimpleEmbeddingGenerator(embedding_dim=100)