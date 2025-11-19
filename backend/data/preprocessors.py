import re
import logging
from typing import Optional
from core.config import settings

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Предобработка текста с опциональной поддержкой NATASHA
    """
    
    def __init__(self):
        self.natasha_available = False
        self._initialize_natasha()
    
    def _initialize_natasha(self) -> None:
        """Инициализация NATASHA с обработкой ошибок"""
        if not settings.ENABLE_NATASHA:
            logger.info("NATASHA отключена в настройках")
            return
            
        try:
            from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
            
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            self.emb = NewsEmbedding()
            self.morph_tagger = NewsMorphTagger(self.emb)
            self.natasha_available = True
            
            logger.info("NATASHA успешно инициализирована")
            
        except ImportError as e:
            logger.warning(f"NATASHA не установлена: {e}")
        except Exception as e:
            logger.error(f"Ошибка инициализации NATASHA: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Базовая очистка текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Приведение к нижнему регистру
            text = text.lower()
            
            # Удаление специальных символов, кроме букв, цифр и пробелов
            text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)
            
            # Замена множественных пробелов одним
            text = re.sub(r'\s+', ' ', text)
            
            # Удаление пробелов в начале и конце
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Ошибка предобработки текста: {e}")
            return text if isinstance(text, str) else ""
    
    def lemmatize_with_natasha(self, text: str) -> Optional[str]:
        """
        Лемматизация с помощью NATASHA
        
        Args:
            text: Текст для лемматизации
            
        Returns:
            Лемматизированный текст или None при ошибке
        """
        if not self.natasha_available:
            logger.warning("NATASHA недоступна для лемматизации")
            return None
        
        if not text or not isinstance(text, str):
            return None
        
        try:
            from natasha import Doc
            
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)
            
            lemmas = [token.lemma for token in doc.tokens if token.lemma]
            result = ' '.join(lemmas)
            
            return result if result else None
            
        except Exception as e:
            logger.error(f"Ошибка лемматизации с NATASHA: {e}")
            return None
    
    def lemmatize_simple(self, text: str) -> str:
        """
        Упрощенная лемматизация (без NATASHA)
        
        Args:
            text: Текст для обработки
            
        Returns:
            Обработанный текст
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Простая обработка - можно добавить стемминг или другие методы
        processed = self.preprocess_text(text)
        return processed
    
    def full_preprocess(self, text: str, use_lemmatization: bool = True) -> str:
        """
        Полная предобработка текста
        
        Args:
            text: Исходный текст
            use_lemmatization: Использовать лемматизацию
            
        Returns:
            Полностью обработанный текст
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Базовая очистка
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text:
            return ""
        
        # Лемматизация если требуется и доступна
        if use_lemmatization and self.natasha_available:
            lemmatized = self.lemmatize_with_natasha(cleaned_text)
            if lemmatized:
                return lemmatized
        
        # Возвращаем очищенный текст если лемматизация не удалась
        return cleaned_text
    
    def batch_preprocess(self, texts: list, use_lemmatization: bool = True) -> list:
        """
        Пакетная обработка текстов
        
        Args:
            texts: Список текстов
            use_lemmatization: Использовать лемматизацию
            
        Returns:
            Список обработанных текстов
        """
        if not texts:
            return []
        
        processed_texts = []
        for text in texts:
            processed = self.full_preprocess(text, use_lemmatization)
            processed_texts.append(processed)
        
        return processed_texts
    
    def get_preprocessor_info(self) -> dict:
        """Информация о препроцессоре"""
        return {
            "natasha_available": self.natasha_available,
            "natasha_enabled": settings.ENABLE_NATASHA,
            "methods_available": {
                "basic_cleaning": True,
                "lemmatization": self.natasha_available
            }
        }


# Альтернативная упрощенная версия без NATASHA
class SimpleTextPreprocessor:
    """
    Упрощенный препроцессор без зависимостей
    """
    
    def __init__(self):
        logger.info("Используется упрощенный препроцессор текста")
    
    def preprocess_text(self, text: str) -> str:
        """Базовая очистка текста"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Приведение к нижнему регистру
            text = text.lower()
            
            # Удаление специальных символов
            text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)
            
            # Нормализация пробелов
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Ошибка упрощенной предобработки: {e}")
            return text if isinstance(text, str) else ""
    
    def full_preprocess(self, text: str) -> str:
        """Полная обработка (аналогично базовой)"""
        return self.preprocess_text(text)
    
    def batch_preprocess(self, texts: list) -> list:
        """Пакетная обработка"""
        return [self.preprocess_text(text) for text in texts]


# Создание экземпляра в зависимости от настроек
if settings.ENABLE_NATASHA:
    try:
        text_preprocessor = TextPreprocessor()
    except Exception as e:
        logger.warning(f"Не удалось создать TextPreprocessor, используем упрощенный: {e}")
        text_preprocessor = SimpleTextPreprocessor()
else:
    text_preprocessor = SimpleTextPreprocessor()