import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
from xgboost import XGBClassifier
import logging
from pathlib import Path
import json
from datetime import datetime

from core.config import settings
from core.schemas import TechStackPrediction, TechnologyCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleStackPredictor:
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EnsembleStackPredictor еще не реализован. Используйте StackPredictor.")

class StackPredictor:  
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.label_columns = None
        self.feature_columns = None
        self.technology_categories = {}
        self.confidence_threshold = 0.3
        self.is_trained = False
        
        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.model_type == "random_forest":
            base_estimator = RandomForestClassifier(
                n_estimators=settings.RF_N_ESTIMATORS,
                max_depth=settings.RF_MAX_DEPTH,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.model = MultiOutputClassifier(base_estimator)
            
        elif self.model_type == "gradient_boosting":
            base_estimator = GradientBoostingClassifier(
                n_estimators=settings.GB_N_ESTIMATORS,
                max_depth=settings.GB_MAX_DEPTH,
                random_state=settings.RANDOM_STATE,
                subsample=0.8 
            )
            self.model = MultiOutputClassifier(base_estimator)
            
        elif self.model_type == "lightgbm":
            base_estimator = lgb.LGBMClassifier(
                n_estimators=settings.LGB_N_ESTIMATORS,
                max_depth=settings.LGB_MAX_DEPTH,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                min_child_weight=0.001,
                reg_alpha=0.1,
                reg_lambda=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                is_unbalance=True, 
                verbose=-1 
            )
            self.model = MultiOutputClassifier(base_estimator)
            
        elif self.model_type == "xgboost":
            base_estimator = XGBClassifier(
                n_estimators=settings.XGB_N_ESTIMATORS,
                max_depth=settings.XGB_MAX_DEPTH,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss',
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=1
            )
            self.model = MultiOutputClassifier(base_estimator)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_labels(self, tech_stacks: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
        from sklearn.preprocessing import MultiLabelBinarizer
        
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(tech_stacks)
        self.label_columns = mlb.classes_.tolist()
        
        logger.info(f"Подготовлено {len(self.label_columns)} уникальных технологий")
        logger.info(f"Размерность матрицы меток: {y.shape}")
        
        return y, self.label_columns
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        if self.model is None:
            self._initialize_model()
        
        logger.info(f"Начало обучения модели {self.model_type}")
        logger.info(f"Размерность данных: X={X.shape}, y={y.shape}")

        self.model.fit(X, y)
        self.is_trained = True

        metrics = self._calculate_metrics(X, y, "train")
        
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self._calculate_metrics(X_val, y_val, "validation")
            metrics.update(val_metrics)

        cv_scores = self._cross_validate(X, y)
        metrics.update(cv_scores)
        
        logger.info("Обучение завершено")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        try:
            probas = self.model.predict_proba(X)
            proba_matrix = np.column_stack([proba[:, 1] for proba in probas])
            return proba_matrix
        except AttributeError:
            logger.warning("Модель не поддерживает predict_proba, используем predict")
            return self.predict(X).astype(float)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probas = self.predict_proba(X)
        predictions = (probas >= self.confidence_threshold).astype(int)
        
        return predictions, probas
    
    def recommend_tech_stack(self, X: np.ndarray, 
                           project_features: Optional[Dict[str, Any]] = None) -> List[TechStackPrediction]:
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        predictions, probabilities = self.predict_with_confidence(X)
        
        recommendations = []
        for i, (tech, prob) in enumerate(zip(self.label_columns, probabilities[0])):
            if predictions[0, i] == 1: 
                category = self._categorize_technology(tech)
                reasoning = self._generate_reasoning(tech, prob, project_features)
                alternatives = self._get_alternatives(tech, probabilities[0])
                
                recommendation = TechStackPrediction(
                    technology=tech,
                    confidence=float(prob),
                    category=category,
                    reasoning=reasoning,
                    alternatives=alternatives
                )
                recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations
    
    def _categorize_technology(self, technology: str) -> TechnologyCategory:
        tech_lower = technology.lower()

        if any(tech in tech_lower for tech in ['spring', 'django', 'flask', 'express', 'laravel', 'rails', 'node']):
            return TechnologyCategory.BACKEND

        elif any(tech in tech_lower for tech in ['react', 'vue', 'angular', 'svelte']):
            return TechnologyCategory.FRONTEND

        elif any(tech in tech_lower for tech in ['flutter', 'android', 'ios', 'react native', 'xamarin']):
            return TechnologyCategory.MOBILE

        elif any(tech in tech_lower for tech in ['mysql', 'postgresql', 'mongodb', 'redis', 'couchbase', 'firestore']):
            return TechnologyCategory.DATABASE

        elif any(tech in tech_lower for tech in ['docker', 'kubernetes', 'jenkins', 'gitlab', 'azure', 'aws', 'gcp', 'heroku', 'vercel']):
            return TechnologyCategory.DEVOPS

        elif any(tech in tech_lower for tech in ['pandas', 'numpy', 'lightgbm', 'xgboost']):
            return TechnologyCategory.DATA_SCIENCE

        elif any(tech in tech_lower for tech in ['tensorflow', 'keras', 'pytorch', 'bert']):
            return TechnologyCategory.AI_ML
        
        else:
            return TechnologyCategory.BACKEND 
    
    def _generate_reasoning(self, technology: str, confidence: float, 
                          project_features: Optional[Dict[str, Any]] = None) -> str:
        reasoning_parts = []
        
        if confidence > 0.8:
            reasoning_parts.append("Высокая уверенность модели")
        elif confidence > 0.6:
            reasoning_parts.append("Средняя уверенность модели")
        else:
            reasoning_parts.append("Умеренная уверенность модели")
        
        if project_features:
            if project_features.get('project_scale') == 'startup':
                reasoning_parts.append("оптимально для стартапов")
            elif project_features.get('project_scale') == 'enterprise':
                reasoning_parts.append("подходит для enterprise-решений")
            
            if project_features.get('complexity') == 'highly_complex':
                reasoning_parts.append("рекомендуется для сложных проектов")
            elif project_features.get('complexity') == 'simple':
                reasoning_parts.append("подходит для простых проектов")

        tech_reasons = {
            'python': "универсальный язык с богатой экосистемой",
            'java': "надежное enterprise-решение",
            'javascript': "идеально для веб-разработки",
            'docker': "обеспечивает контейнеризацию и переносимость",
            'kubernetes': "масштабируемость и оркестрация контейнеров",
            'react': "популярный фреймворк для пользовательских интерфейсов",
            'mysql': "надежная реляционная база данных",
            'mongodb': "гибкая NoSQL база данных",
            'tensorflow': "мощный фреймворк для машинного обучения"
        }
        
        for tech, reason in tech_reasons.items():
            if tech in technology.lower():
                reasoning_parts.append(reason)
                break
        
        return ". ".join(reasoning_parts) + "."
    
    def _get_alternatives(self, technology: str, probabilities: np.ndarray, 
                         top_n: int = 3) -> List[str]:
        tech_probs = list(zip(self.label_columns, probabilities))
        alternatives = [(tech, prob) for tech, prob in tech_probs 
                       if tech != technology and prob > 0.1]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return [tech for tech, prob in alternatives[:top_n]]
    
    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray, 
                          dataset_type: str = "train") -> Dict[str, float]:
        predictions = self.predict(X)
        
        metrics = {
            f"{dataset_type}_accuracy": accuracy_score(y, predictions),
            f"{dataset_type}_precision": precision_score(y, predictions, average='micro', zero_division=0),
            f"{dataset_type}_recall": recall_score(y, predictions, average='micro', zero_division=0),
            f"{dataset_type}_f1": f1_score(y, predictions, average='micro', zero_division=0)
        }
        
        return metrics
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 3) -> Dict[str, float]:
        try:
            cv_scores = cross_val_score(
                self.model, X, y, 
                cv=min(cv, len(X)),  
                scoring='f1_micro',
                n_jobs=1
            )
            
            return {
                "cv_f1_mean": np.mean(cv_scores),
                "cv_f1_std": np.std(cv_scores)
            }
        except Exception as e:
            logger.warning(f"Кросс-валидация не удалась: {e}")
            return {}
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_trained:
            return None
        
        try:
            if hasattr(self.model, 'estimators_'):
                importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        except Exception as e:
            logger.warning(f"Не удалось получить важность признаков: {e}")
        
        return None
    
    def save(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'label_columns': self.label_columns,
            'confidence_threshold': self.confidence_threshold,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Модель сохранена: {filepath}")
    
    def load(self, filepath: str) -> None:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Файл модели не найден: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.label_columns = model_data['label_columns']
        self.confidence_threshold = model_data.get('confidence_threshold', 0.3)
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Модель загружена: {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "num_technologies": len(self.label_columns) if self.label_columns else 0,
            "confidence_threshold": self.confidence_threshold
        }


class SimpleStackPredictor:   
    def __init__(self):
        self.model = None
        self.label_columns = None
        self.is_trained = False
    
    def prepare_labels(self, tech_stacks: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(tech_stacks)
        self.label_columns = mlb.classes_.tolist()
        return y, self.label_columns
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        
        self.model = MultiOutputClassifier(RandomForestClassifier(n_estimators=10, random_state=42))
        self.model.fit(X, y)
        self.is_trained = True
        
        return {"train_accuracy": 0.8, "train_f1": 0.7}
    
    def recommend_tech_stack(self, X: np.ndarray, **kwargs) -> List[TechStackPrediction]:
        if not self.is_trained or self.label_columns is None:
            return []

        import random
        recommendations = []
        for tech in random.sample(self.label_columns, min(3, len(self.label_columns))):
            recommendations.append(
                TechStackPrediction(
                    technology=tech,
                    confidence=random.uniform(0.5, 0.9),
                    category=TechnologyCategory.BACKEND,
                    reasoning="Тестовая рекомендация",
                    alternatives=[]
                )
            )
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "simple_test_predictor",
            "is_trained": self.is_trained,
            "num_technologies": len(self.label_columns) if self.label_columns else 0
        }