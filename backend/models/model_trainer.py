import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import joblib

# Визуализация (опционально)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger = logging.getLogger(__name__)##хзхзхз
    logger.warning("Matplotlib/Seaborn не установлены, визуализация отключена")

from core.config import settings
from core.schemas import TrainingConfig, ModelMetadata, DatasetStatistics
from models.stack_predictor import StackPredictor
from features.feature_engineer import FeatureEngineer
from features.text_embeddings import TextEmbeddingGenerator
from features.entity_extractors import EntityExtractor
from data.dataset_loader import DatasetLoader
from data.preprocessors import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Класс для обучения и оценки моделей рекомендации технологических стеков
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset_loader = DatasetLoader()
        self.text_preprocessor = TextPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.embedding_generator = TextEmbeddingGenerator()
        self.entity_extractor = EntityExtractor()
        
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.experiment_results = {}
        self.technologies = []
    
    def load_and_prepare_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, str, List[List[str]]]:
        """
        Загрузка и подготовка данных
        
        Returns:
            Tuple с DataFrame, названием текстовой колонки и списком стеков технологий
        """
        logger.info("Загрузка JSON датасета...")
        
        try:
            df = self.dataset_loader.load_data(sample_size=sample_size)
            
            logger.info("Предобработка текстовых описаний...")
            df['processed_description'] = df['description'].apply(
                lambda x: self.text_preprocessor.preprocess_text(str(x))
            )
            
            # Лемматизация с NATASHA
            if settings.ENABLE_NATASHA:
                logger.info("Лемматизация текстов с NATASHA...")
                df['lemmatized_description'] = df['processed_description'].apply(
                    lambda x: self.text_preprocessor.lemmatize_with_natasha(str(x))
                )
                text_column = 'lemmatized_description'
            else:
                text_column = 'processed_description'
            
            # Подготовка меток (технологических стеков)
            logger.info("Подготовка меток...")
            tech_stacks = df['tech_stack'].tolist()
            
            logger.info(f"Загружено {len(df)} проектов")
            logger.info(f"Используется текстовая колонка: {text_column}")
            
            return df, text_column, tech_stacks
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise
    
    def extract_features(self, df: pd.DataFrame, text_column: str) -> np.ndarray:
        """
        Извлечение признаков из данных
        """
        logger.info("Извлечение признаков...")
        
        # Текстовые эмбеддинги
        texts = df[text_column].tolist()
        logger.info(f"Генерация BERT эмбеддингов для {len(texts)} текстов...")
        text_embeddings = self.embedding_generator.get_bert_embeddings(texts)
        
        # Категориальные признаки
        logger.info("Обработка категориальных признаков...")
        categorical_data = self.feature_engineer.prepare_categorical_features(df)
        
        # ОБУЧАЕМ FeatureEngineer на категориальных данных
        if not categorical_data.empty:
            logger.info("Обучение FeatureEngineer на категориальных данных...")
            self.feature_engineer.fit(categorical_data)
        
        # Извлечение сущностей
        entity_features = None
        if settings.ENABLE_ENTITY_EXTRACTION:
            logger.info("Извлечение сущностей из текстов...")
            entity_features = []
            for text in df['description'].tolist():
                entities = self.entity_extractor.extract_entities(str(text))
                entity_vector = self._entities_to_features(entities)
                entity_features.append(entity_vector)
            
            entity_features = np.array(entity_features)
            logger.info(f"Извлечено entity features: {entity_features.shape}")
        
        # Создание финального вектора признаков
        feature_vector = self.feature_engineer.create_feature_vector(
            text_embedding=text_embeddings, ##=---------------------------------------------------------
            categorical_data=categorical_data,
            entity_features=entity_features
        )
        
        logger.info(f"Размерность финального вектора признаков: {feature_vector.shape}")
        return feature_vector
    
    def _entities_to_features(self, entities: Dict[str, List]) -> np.ndarray:
        """
        Преобразование сущностей в числовые признаки
        """
        features = []
        
        # Количество сущностей каждого типа
        features.append(len(entities.get('names', [])))
        features.append(len(entities.get('locations', [])))
        features.append(len(entities.get('dates', [])))
        features.append(len(entities.get('technologies', [])))
        
        # Дополнительные признаки из сущностей
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        features.append(total_entities)
        
        # Бинарные признаки наличия сущностей
        features.append(1 if entities.get('names') else 0)
        features.append(1 if entities.get('locations') else 0)
        features.append(1 if entities.get('dates') else 0)
        features.append(1 if entities.get('technologies') else 0)
        
        return np.array(features)
    
    def prepare_labels(self, tech_stacks: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
        """
        Подготовка мульти-лабельных меток
        
        Args:
            tech_stacks: Список списков технологий
            
        Returns:
            Tuple с матрицей меток и списком технологий
        """
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(tech_stacks)
        technologies = mlb.classes_.tolist()
        self.technologies = technologies
        
        logger.info(f"Подготовлено {len(technologies)} уникальных технологий")
        logger.info(f"Размерность матрицы меток: {y.shape}")
        
        return y, technologies
    

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Обучение нескольких моделей и выбор лучшей
        """
        logger.info("Начало обучения моделей...")
        
        model_types = ['random_forest', 'lightgbm', 'xgboost']
        results = {}
        
        for model_type in model_types:
            logger.info(f"Обучение модели: {model_type}")
            
            try:
                # Создание и обучение модели с оптимизированными параметрами
                predictor = StackPredictor(model_type=model_type)
                predictor.label_columns = self.technologies
                
                # Оптимизированные параметры для каждой модели
                if model_type == 'lightgbm':
                    # Специальные параметры для LightGBM
                    predictor.model.estimator.set_params(
                        num_leaves=31,
                        min_data_in_leaf=20,  # Увеличиваем минимальное количество данных в листе
                        max_depth=-1,  # Автоматическое определение глубины
                        learning_rate=0.1,
                        n_estimators=100,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        min_gain_to_split=0.02,  # Минимальный gain для разделения
                        subsample=0.8,
                        colsample_bytree=0.8,
                        is_unbalance=True,  # Для несбалансированных данных
                        verbose=-1  # Убираем лишние предупреждения
                    )
                elif model_type == 'xgboost':
                    predictor.model.estimator.set_params(
                        max_depth=6,
                        learning_rate=0.1,
                        n_estimators=100,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        scale_pos_weight=1  # Балансировка классов
                    )
                
                metrics = predictor.fit(X_train, y_train, validation_data=(X_val, y_val))
                
                # Оценка на валидационной выборке
                val_predictions = predictor.predict(X_val)
                val_f1 = f1_score(y_val, val_predictions, average='micro', zero_division=0)
                
                results[model_type] = {
                    'model': predictor,
                    'metrics': metrics,
                    'validation_f1': val_f1,
                    'feature_importance': predictor.get_feature_importance()
                }
                
                logger.info(f"Модель {model_type} обучена. Validation F1: {val_f1:.4f}")
                
                # Обновление лучшей модели
                if val_f1 > self.best_score:
                    self.best_score = val_f1
                    self.best_model = predictor
                    logger.info(f"Новая лучшая модель: {model_type} с F1: {val_f1:.4f}")
                    
            except Exception as e:
                logger.error(f"Ошибка при обучении модели {model_type}: {e}")
                continue
        
        self.models = results
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Детальная оценка моделей на тестовой выборке
        """
        logger.info("Оценка моделей на тестовой выборке...")
        
        evaluation_results = {}
        
        for model_name, model_data in self.models.items():
            predictor = model_data['model']
            
            try:
                # Предсказания
                predictions = predictor.predict(X_test)
                
                # Метрики
                metrics = {
                    'f1_micro': f1_score(y_test, predictions, average='micro', zero_division=0),
                    'f1_macro': f1_score(y_test, predictions, average='macro', zero_division=0),
                    'precision_micro': precision_score(y_test, predictions, average='micro', zero_division=0),
                    'recall_micro': recall_score(y_test, predictions, average='micro', zero_division=0),
                    'accuracy': accuracy_score(y_test, predictions)
                }
                
                # Детальный отчет по классам (только для топ-10 технологий)
                top_technologies = predictor.label_columns[:10] if len(predictor.label_columns) > 10 else predictor.label_columns
                
                # Создаем маски для топ технологий
                top_indices = [predictor.label_columns.index(tech) for tech in top_technologies if tech in predictor.label_columns]
                y_test_top = y_test[:, top_indices]
                predictions_top = predictions[:, top_indices]
                
                class_report = classification_report(
                    y_test_top, predictions_top, 
                    target_names=top_technologies,
                    output_dict=True,
                    zero_division=0
                )
                
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'classification_report': class_report,
                    'top_technologies_report': top_technologies
                }
                
                logger.info(f"Модель {model_name} - Test F1 micro: {metrics['f1_micro']:.4f}")
                
            except Exception as e:
                logger.error(f"Ошибка при оценке модели {model_name}: {e}")
                continue
        
        return evaluation_results
    
    def save_models(self, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Сохранение обученных моделей
        """
        if output_dir is None:
            output_dir = settings.MODELS_DIR / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Сохранение отдельных моделей
        for model_name, model_data in self.models.items():
            try:
                model_path = output_path / f"{model_name}_model.joblib"
                model_data['model'].save(str(model_path))
                saved_paths[model_name] = str(model_path)
                logger.info(f"Сохранена модель {model_name}: {model_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения модели {model_name}: {e}")
        
        # Сохранение лучшей модели
        if self.best_model:
            try:
                best_model_path = output_path / "best_model.joblib"
                self.best_model.save(str(best_model_path))
                saved_paths['best_model'] = str(best_model_path)
                logger.info(f"Сохранена лучшая модель: {best_model_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения лучшей модели: {e}")
        
        # Сохранение FeatureEngineer
        try:
            feature_engineer_path = output_path / "feature_engineer.joblib"
            joblib.dump(self.feature_engineer, feature_engineer_path)
            saved_paths['feature_engineer'] = str(feature_engineer_path)
            logger.info(f"Сохранен FeatureEngineer: {feature_engineer_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения FeatureEngineer: {e}")
        
        # Сохранение метаданных
        metadata = self._create_metadata()
        metadata_path = output_path / "training_metadata.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            saved_paths['metadata'] = str(metadata_path)
            logger.info(f"Сохранены метаданные: {metadata_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных: {e}")
        
        # Сохранение результатов эксперимента
        results_path = output_path / "experiment_results.json"
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_results, f, indent=2, ensure_ascii=False, default=str)
            saved_paths['results'] = str(results_path)
        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {e}")
        
        logger.info(f"Все модели сохранены в: {output_dir}")
        return saved_paths
    
    def _create_metadata(self) -> Dict[str, Any]:
        """Создание метаданных обучения"""
        # Исправление для Pydantic V2
        try:
            config_dict = self.config.model_dump()  # Pydantic V2
        except AttributeError:
            config_dict = self.config.dict()  # Pydantic V1
        
        return {
            "training_date": datetime.now().isoformat(),
            "config": config_dict,
            "best_model": {
                "type": self.best_model.model_type if self.best_model else None,
                "score": self.best_score,
                "num_technologies": len(self.best_model.label_columns) if self.best_model else 0
            },
            "models_trained": list(self.models.keys()),
            "dataset_info": {
                "total_technologies": len(self.technologies),
                "technologies_sample": self.technologies[:20]  # Первые 20 технологий
            },
            "feature_engineer_info": self.feature_engineer.get_engineer_info() if hasattr(self.feature_engineer, 'get_engineer_info') else {}
        }
    
    def plot_training_results(self, output_dir: Path) -> None:
        """
        Визуализация результатов обучения
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Визуализация недоступна (matplotlib/seaborn не установлены)")
            return
            
        if not self.experiment_results:
            logger.warning("Нет данных для визуализации")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # График сравнения моделей
            self._plot_model_comparison(output_dir)
            logger.info("Создан график сравнения моделей")
        except Exception as e:
            logger.error(f"Ошибка создания графика сравнения моделей: {e}")
    
    def _plot_model_comparison(self, output_dir: Path) -> None:
        """График сравнения моделей"""
        if not VISUALIZATION_AVAILABLE:
            return
            
        model_names = []
        f1_scores = []
        
        evaluation_results = self.experiment_results.get('evaluation', {})
        for model_name, results in evaluation_results.items():
            model_names.append(model_name)
            f1_scores.append(results['metrics']['f1_micro'])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Сравнение моделей по F1-score')
        plt.ylabel('F1-score (micro)')
        plt.ylim(0, 1)
        
        # Добавление значений на столбцы
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_training_pipeline(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Запуск полного пайплайна обучения
        
        Args:
            sample_size: Размер выборки для быстрого тестирования
            
        Returns:
            Словарь с результатами обучения
        """
        logger.info("Запуск пайплайна обучения...")
        
        try:
            # 1. Загрузка и подготовка данных
            df, text_column, tech_stacks = self.load_and_prepare_data(sample_size=sample_size)
            
            # 2. Подготовка меток
            y, technologies = self.prepare_labels(tech_stacks)
            
            # 3. Извлечение признаков
            X = self.extract_features(df, text_column)
            
            logger.info(f"Данные подготовлены: X={X.shape}, y={y.shape}")
            logger.info(f"Уникальных технологий: {len(technologies)}")
            
            # 4. Разделение на train/val/test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.config.test_size, 
                random_state=settings.RANDOM_STATE
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.2,
                random_state=settings.RANDOM_STATE
            )
            
            logger.info(f"Разделение данных: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
            
            # 5. Обучение моделей
            training_results = self.train_models(X_train, y_train, X_val, y_val)
            
            if not self.models:
                raise ValueError("Ни одна модель не была успешно обучена")
            
            # 6. Оценка на тестовой выборке
            evaluation_results = self.evaluate_models(X_test, y_test)
            
            # 7. Сохранение результатов
            self.experiment_results = {
                'training': training_results,
                'evaluation': evaluation_results,
                'technologies': technologies,
                'data_shape': {
                    'train': X_train.shape,
                    'val': X_val.shape,
                    'test': X_test.shape
                }
            }
            
            # 8. Сохранение моделей
            saved_paths = self.save_models()
            
            # 9. Визуализация результатов
            self.plot_training_results(Path(saved_paths.get('best_model', Path('.'))).parent)
            
            logger.info("Пайплайн обучения успешно завершен!")
            
            return {
                'success': True,
                'best_score': self.best_score,
                'best_model_type': self.best_model.model_type if self.best_model else None,
                'saved_paths': saved_paths,
                'results': {
                    'num_technologies': len(technologies),
                    'models_trained': len(self.models),
                    'best_f1_score': self.best_score
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка в пайплайне обучения: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Основная функция для запуска обучения из командной строки"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение модели для рекомендации технологических стеков')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'lightgbm', 'xgboost', 'all'],
                       help='Тип модели для обучения')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Директория для сохранения моделей')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Доля тестовой выборки')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Размер выборки для быстрого тестирования')
    
    args = parser.parse_args()
    
    # Конфигурация обучения
    config = TrainingConfig(
        model_type=args.model_type,
        test_size=args.test_size
    )
    
    # Запуск обучения
    trainer = ModelTrainer(config)
    results = trainer.run_training_pipeline(sample_size=args.sample_size)
    
    if results['success']:
        print("✅ Обучение завершено успешно!")
        print(f"📊 Лучший результат: {results['best_score']:.4f}")
        print(f"🤖 Лучшая модель: {results['best_model_type']}")
        print(f"💾 Модели сохранены в: {list(results['saved_paths'].keys())}")
        print(f"🔧 Технологий в модели: {results['results']['num_technologies']}")
    else:
        print(f"❌ Ошибка обучения: {results['error']}")
        exit(1)


if __name__ == "__main__":
    main()