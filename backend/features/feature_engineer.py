import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Optional, List, Dict, Any
from core.config import settings

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.categorical_features = settings.CATEGORICAL_FEATURES
        self.preprocessor = None
        self.is_fitted = False
        self.expected_feature_dim = None
        self.entity_feature_dim = 6 
        
        logger.info(f"Инициализирован FeatureEngineer с категориальными признаками: {self.categorical_features}")
    
    def fit(self, categorical_data: pd.DataFrame) -> None:
        if categorical_data is None or categorical_data.empty:
            logger.warning("Переданы пустые данные для обучения")
            self.is_fitted = True
            return
        try:
            categorical_prepared = self.prepare_categorical_features(categorical_data)
            if categorical_prepared.empty:
                logger.warning("Нет категориальных признаков для обучения")
                self.is_fitted = True
                return
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
                ],
                remainder='drop'
            )
            
            self.preprocessor.fit(categorical_prepared)
            self.is_fitted = True

            categorical_dim = self.preprocessor.transformers_[0][1].get_feature_names_out().shape[0]
            logger.info(f"Препроцессор обучен. Категориальных фичей: {categorical_dim}")
            
        except Exception as e:
            logger.error(f"Ошибка обучения препроцессора: {e}")
            self.is_fitted = False
    
    def create_feature_vector(self, 
                            text_embedding: np.ndarray,
                            categorical_data: pd.DataFrame = None,
                            entity_features: np.ndarray = None) -> np.ndarray:

        features_list = []
        
        if text_embedding is not None and text_embedding.size > 0:
            if len(text_embedding.shape) == 1:
                text_embedding = text_embedding.reshape(1, -1)
            features_list.append(text_embedding)
        else:
            logger.error("Текстовые эмбеддинги обязательны!")
            raise ValueError("Text embeddings are required")

        categorical_processed = self._process_categorical_features(categorical_data)
        if categorical_processed is not None:
            features_list.append(categorical_processed)

        entity_processed = self._process_entity_features(entity_features)
        if entity_processed is not None:
            features_list.append(entity_processed)

        feature_vector = self._safe_hstack(features_list)

        if self.expected_feature_dim is None:
            self.expected_feature_dim = feature_vector.shape[1]
            logger.info(f"Установлена ожидаемая размерность признаков: {self.expected_feature_dim}")
        
        return feature_vector
    
    def _process_categorical_features(self, categorical_data: pd.DataFrame) -> Optional[np.ndarray]:
        if categorical_data is None or categorical_data.empty:
            return None
        
        try:
            if not self.is_fitted:
                logger.info("Автоматическое обучение препроцессора на категориальных данных")
                self.fit(categorical_data)
            
            categorical_prepared = self.prepare_categorical_features(categorical_data)
            if categorical_prepared.empty:
                return None
            
            categorical_processed = self.preprocessor.transform(categorical_prepared)
            return categorical_processed
            
        except Exception as e:
            logger.error(f"Ошибка обработки категориальных признаков: {e}")
            return None
    
    def _process_entity_features(self, entity_features: np.ndarray) -> Optional[np.ndarray]:
        if entity_features is None:
            return None
        
        try:
            if len(entity_features.shape) == 1:
                entity_features = entity_features.reshape(1, -1)
            
            if entity_features.shape[1] != self.entity_feature_dim:
                logger.warning(f"Entity features имеют размерность {entity_features.shape[1]}, ожидается {self.entity_feature_dim}")
                if entity_features.shape[1] < self.entity_feature_dim:
                    padding = np.zeros((entity_features.shape[0], self.entity_feature_dim - entity_features.shape[1]))
                    entity_features = np.hstack([entity_features, padding])
                else:
                    entity_features = entity_features[:, :self.entity_feature_dim]
            
            return entity_features
            
        except Exception as e:
            logger.error(f"Ошибка обработки entity features: {e}")
            return None
    
    def _safe_hstack(self, arrays: List[np.ndarray]) -> np.ndarray:
        if not arrays:
            raise ValueError("Нет массивов для объединения")
        
        n_samples = arrays[0].shape[0]
        for arr in arrays[1:]:
            if arr.shape[0] != n_samples:
                raise ValueError(f"Несовпадение количества samples: {arr.shape[0]} != {n_samples}")
        
        return np.hstack(arrays)
    
    def prepare_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        available_columns = [col for col in self.categorical_features if col in df.columns]
        if not available_columns:
            logger.warning("Нет доступных категориальных признаков в данных")
            return pd.DataFrame()

        categorical_data = df[available_columns].fillna('unknown')
        logger.debug(f"Подготовлены категориальные признаки: {available_columns}")
        
        return categorical_data
    
    def get_feature_dimensions(self, 
                             text_embedding_dim: int,
                             has_categorical: bool = True,
                             has_entity_features: bool = False) -> int:
        total_dim = text_embedding_dim

        if has_categorical and self.is_fitted and self.preprocessor is not None:
            try:
                categorical_dim = self.preprocessor.transformers_[0][1].get_feature_names_out().shape[0]
                total_dim += categorical_dim
                logger.debug(f"Категориальные признаки добавляют {categorical_dim} фичей")
            except Exception as e:
                logger.warning(f"Не удалось получить размерность категориальных признаков: {e}")

        if has_entity_features:
            total_dim += self.entity_feature_dim
            logger.debug(f"Entity features добавляют {self.entity_feature_dim} фичей")
        
        logger.info(f"Общая ожидаемая размерность признаков: {total_dim}")
        return total_dim
    
    def ensure_feature_dimension(self, features: np.ndarray) -> np.ndarray:
        if self.expected_feature_dim is None:
            return features
        
        current_dim = features.shape[1]
        if current_dim == self.expected_feature_dim:
            return features
        
        logger.warning(f"Корректировка размерности признаков: {current_dim} -> {self.expected_feature_dim}")
        
        if current_dim < self.expected_feature_dim:
            padding = np.zeros((features.shape[0], self.expected_feature_dim - current_dim))
            return np.hstack([features, padding])
        else:
            return features[:, :self.expected_feature_dim]
    
    def get_engineer_info(self) -> Dict[str, Any]:
        info = {
            "is_fitted": self.is_fitted,
            "categorical_features": self.categorical_features,
            "expected_feature_dim": self.expected_feature_dim,
            "entity_feature_dim": self.entity_feature_dim
        }
        
        if self.is_fitted and self.preprocessor is not None:
            try:
                categorical_dim = self.preprocessor.transformers_[0][1].get_feature_names_out().shape[0]
                info["categorical_feature_dim"] = categorical_dim
            except:
                info["categorical_feature_dim"] = "unknown"
        
        return info