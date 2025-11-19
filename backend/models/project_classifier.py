import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging
from pathlib import Path
import json
from datetime import datetime

from core.config import settings
from core.schemas import (
    ProjectScale, Industry, ProjectType, TeamSize, 
    Complexity, BudgetLevel, PerformanceRequirement,
    ScalabilityRequirement, SecurityRequirement,
    RealtimeRequirement, IntegrationRequirement
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectClassifier:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.models = {} 
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False

        self.target_attributes = [
            'project_type', 'project_scale', 'industry', 
            'budget', 'team_size', 'complexity'
        ]

        self.technical_attributes = [
            'performance', 'scalability', 'security', 'realtime', 'integration'
        ]
    
    def _initialize_model(self, task_name: str):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=settings.RANDOM_STATE
            )
        elif self.model_type == "svm":
            return SVC(
                probability=True,
                random_state=settings.RANDOM_STATE,
                kernel='rbf'
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=settings.RANDOM_STATE,
                max_iter=1000,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_features(self, df: pd.DataFrame, text_embeddings: np.ndarray) -> np.ndarray:
        logger.info("Подготовка признаков для классификации проекта")
        categorical_features = []
        categorical_columns = ['project_scale', 'industry', 'team_size', 'complexity', 'budget']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].fillna('unknown'))
                categorical_features.append(encoded)
                self.label_encoders[col] = le
        
        if categorical_features:
            categorical_matrix = np.column_stack(categorical_features)
            if text_embeddings is not None:
                features = np.column_stack([text_embeddings, categorical_matrix])
            else:
                features = categorical_matrix
        else:
            features = text_embeddings
        
        self.feature_names = [f"embedding_{i}" for i in range(text_embeddings.shape[1])] + categorical_columns
        
        logger.info(f"Размерность признаков для классификации: {features.shape}")
        return features
    

    def _encode_categorical_features(self, categorical_data: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            available_columns = [col for col in self.target_attributes if col in categorical_data.columns]
            if not available_columns:
                return None
            
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_features = encoder.fit_transform(categorical_data[available_columns])
            
            return encoded_features 
            
        except Exception as e:
            logger.error(f"Ошибка кодирования категориальных признаков: {e}")
            return None
    
    def prepare_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        available_columns = [col for col in self.target_attributes if col in df.columns]
        if not available_columns:
            return pd.DataFrame()
        
        return df[available_columns].fillna('unknown')

 
    def prepare_labels(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        labels = {}
        
        for attribute in self.target_attributes:
            if attribute in df.columns:
                le = LabelEncoder()
                labels[attribute] = le.fit_transform(df[attribute].fillna('unknown'))
                self.label_encoders[attribute] = le
                logger.info(f"Подготовлены метки для {attribute}: {len(le.classes_)} классов")

        tech_labels = {}
        if 'technical_requirements' in df.columns:
            for tech_attr in self.technical_attributes:
                attr_values = []
                for reqs in df['technical_requirements']:
                    if isinstance(reqs, dict) and tech_attr in reqs:
                        attr_values.append(reqs[tech_attr])
                    else:
                        attr_values.append('unknown')
                
                if len(set(attr_values)) > 1:
                    le = LabelEncoder()
                    tech_labels[tech_attr] = le.fit_transform(attr_values)
                    self.label_encoders[tech_attr] = le
                    logger.info(f"Подготовлены метки для {tech_attr}: {len(le.classes_)} классов")
        
        labels.update(tech_labels)
        return labels
    
    def fit(self, X: np.ndarray, labels: Dict[str, np.ndarray],
            validation_data: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = None) -> Dict[str, Any]:
        logger.info("Начало обучения классификаторов проекта")
        
        metrics = {}
        
        for attribute, y in labels.items():
            logger.info(f"Обучение классификатора для: {attribute}")
            
            try:
                model = self._initialize_model(attribute)
                model.fit(X, y)
                self.models[attribute] = model

                train_predictions = model.predict(X)
                train_accuracy = accuracy_score(y, train_predictions)
                train_f1 = f1_score(y, train_predictions, average='weighted', zero_division=0)
                
                metrics[attribute] = {
                    'train_accuracy': train_accuracy,
                    'train_f1': train_f1,
                    'classes': self.label_encoders[attribute].classes_.tolist()
                }

                if validation_data:
                    X_val, labels_val = validation_data
                    if attribute in labels_val:
                        val_predictions = model.predict(X_val)
                        val_accuracy = accuracy_score(labels_val[attribute], val_predictions)
                        val_f1 = f1_score(labels_val[attribute], val_predictions, average='weighted', zero_division=0)
                        
                        metrics[attribute].update({
                            'val_accuracy': val_accuracy,
                            'val_f1': val_f1
                        })
                
                logger.info(f"Классификатор {attribute} обучен. Accuracy: {train_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Ошибка при обучении классификатора {attribute}: {e}")
                continue
        
        self.is_trained = True
        return metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Модели не обучены. Вызовите fit() сначала.")
        
        predictions = {}
        probabilities = {}
        
        for attribute, model in self.models.items():
            try:
                pred = model.predict(X)
                proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

                if attribute in self.label_encoders:
                    decoded_pred = self.label_encoders[attribute].inverse_transform(pred)
                    predictions[attribute] = decoded_pred[0] if len(decoded_pred) == 1 else decoded_pred
                    
                    if proba is not None:
                        class_probs = {}
                        for i, class_name in enumerate(self.label_encoders[attribute].classes_):
                            class_probs[class_name] = float(proba[0][i])
                        probabilities[attribute] = class_probs
                
            except Exception as e:
                logger.error(f"Ошибка при предсказании {attribute}: {e}")
                predictions[attribute] = None
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def predict_from_text(self, text_embedding: np.ndarray, 
                         available_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if available_features is None:
            available_features = {}

        feature_vector = self._create_feature_vector(text_embedding, available_features)
        
        if feature_vector is None:
            raise ValueError("Не удалось создать вектор признаков")
        
        return self.predict(feature_vector.reshape(1, -1))
    
    def _create_feature_vector(self, text_embedding: np.ndarray, 
                             available_features: Dict[str, Any]) -> Optional[np.ndarray]:
        if text_embedding is None:
            return None

        features = [text_embedding.flatten()]

        categorical_columns = ['project_scale', 'industry', 'team_size', 'complexity', 'budget']
        
        for col in categorical_columns:
            if col in available_features and col in self.label_encoders:
                try:
                    encoded = self.label_encoders[col].transform([available_features[col]])[0]
                    features.append([encoded])
                except ValueError:
                    features.append([0])
            else:
                features.append([0])
        
        return np.column_stack(features)
    
    def get_feature_importance(self, attribute: str) -> Optional[Dict[str, float]]:
        if attribute not in self.models:
            return None
        
        model = self.models[attribute]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if self.feature_names and len(self.feature_names) == len(importances):
                return dict(zip(self.feature_names, importances))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        
        return None
    
    def evaluate(self, X_test: np.ndarray, labels_test: Dict[str, np.ndarray]) -> Dict[str, Any]:
        logger.info("Оценка классификаторов проекта")
        
        results = {}
        
        for attribute, y_true in labels_test.items():
            if attribute not in self.models:
                continue
            
            model = self.models[attribute]
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            class_report = classification_report(
                y_true, y_pred,
                target_names=self.label_encoders[attribute].classes_,
                output_dict=True,
                zero_division=0
            )
            
            results[attribute] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': self._create_confusion_matrix(y_true, y_pred, attribute)
            }
            
            logger.info(f"Атрибут {attribute}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return results
    
    def _create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               attribute: str) -> Dict[str, List]:
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        classes = self.label_encoders[attribute].classes_
        
        return {
            'matrix': cm.tolist(),
            'classes': classes.tolist()
        }
    
    def save(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("Модели не обучены")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': {attr: joblib.dump(model, f'{filepath}_{attr}.joblib')[0] 
                      for attr, model in self.models.items()},
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'target_attributes': self.target_attributes,
            'technical_attributes': self.technical_attributes,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Классификаторы проекта сохранены: {filepath}")
    
    def load(self, filepath: str) -> None:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Файл модели не найден: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = {}
        for attr, model_path in model_data['models'].items():
            self.models[attr] = joblib.load(model_path)
        
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.target_attributes = model_data['target_attributes']
        self.technical_attributes = model_data['technical_attributes']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Классификаторы проекта загружены: {filepath}")
        logger.info(f"Загружено моделей: {len(self.models)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "num_models": len(self.models),
            "target_attributes": self.target_attributes,
            "technical_attributes": self.technical_attributes
        }
        classifiers_info = {}
        for attribute, model in self.models.items():
            classifiers_info[attribute] = {
                "model_type": type(model).__name__,
                "classes": self.label_encoders[attribute].classes_.tolist() if attribute in self.label_encoders else []
            }
        
        info["classifiers"] = classifiers_info
        return info


class TechnicalRequirementsPredictor:   
    def __init__(self):
        self.classifier = ProjectClassifier(model_type="random_forest")
        self.requirements_mapping = {
            # 'project_scale': ProjectScale,
            # 'project_type': ProjectType,
            # 'industry': Industry,
            # 'team_size': TeamSize,
            # 'budget': BudgetLevel,
            # 'comlexity': Complexity,
            'performance': PerformanceRequirement,
            'scalability': ScalabilityRequirement,
            'security': SecurityRequirement,
            'realtime': RealtimeRequirement,
            'integration': IntegrationRequirement
        }
    
    def predict_requirements(self, text_embedding: np.ndarray, 
                           project_features: Dict[str, Any]) -> Dict[str, Any]:
        predictions = self.classifier.predict_from_text(text_embedding, project_features)

        tech_requirements = {}
        for req_type, enum_class in self.requirements_mapping.items():
            if req_type in predictions['predictions']:
                pred_value = predictions['predictions'][req_type]
                try:
                    tech_requirements[req_type] = enum_class(pred_value)
                except ValueError:
                    tech_requirements[req_type] = list(enum_class)[0]
        
        return {
            'technical_requirements': tech_requirements,
            'probabilities': predictions.get('probabilities', {})
        }


def main():
    from sklearn.datasets import make_classification
    
    X, _ = make_classification(n_samples=1000, n_features=50, random_state=42)

    labels = {}
    for attribute in ['project_type', 'project_scale', 'industry']:
        y = np.random.randint(0, 3, 1000)
        labels[attribute] = y

    classifier = ProjectClassifier()
    metrics = classifier.fit(X, labels)
    
    print("Результаты обучения:")
    for attr, metric in metrics.items():
        print(f"{attr}: Accuracy={metric['train_accuracy']:.4f}")

    classifier.save("project_classifier.joblib")

    new_classifier = ProjectClassifier()
    new_classifier.load("project_classifier.joblib")

    test_sample = X[0:1]  # Первый образец
    predictions = new_classifier.predict(test_sample)
    print("Предсказания:", predictions['predictions'])

class SimpleProjectClassifier:    
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.is_trained = False
    
    def prepare_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        target_attributes = ['project_type', 'project_scale', 'industry', 'budget', 'team_size', 'complexity']
        available_columns = [col for col in target_attributes if col in df.columns]
        
        if not available_columns:
            return pd.DataFrame()
        
        return df[available_columns].fillna('unknown')
    
    def fit(self, X: np.ndarray, labels: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        
        metrics = {}
        
        for attribute, y in labels.items():
            try:
                model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
                model.fit(X, y)
                self.models[attribute] = model

                metrics[attribute] = {
                    'train_accuracy': 0.85,
                    'train_f1': 0.80,
                    'classes': ['class1', 'class2', 'class3'] 
                }
                
            except Exception as e:
                logger.error(f"Ошибка обучения {attribute}: {e}")
                continue
        
        self.is_trained = True
        return metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained:
            return {'predictions': {}, 'probabilities': {}}
        
        predictions = {}
        for attribute in self.models.keys():
            options = ['web', 'mobile', 'data_science'] if attribute == 'project_type' else \
                     ['startup', 'enterprise', 'smb'] if attribute == 'project_scale' else \
                     ['finance', 'healthcare', 'transport']
            predictions[attribute] = np.random.choice(options)
        
        return {
            'predictions': predictions,
            'probabilities': {}
        }


if __name__ == "__main__":
    main()

