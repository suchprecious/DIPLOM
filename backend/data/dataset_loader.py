import pandas as pd
import json
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from core.config import settings

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or settings.DATA_PATH
        self._check_dataset_exists()
    
    def _check_dataset_exists(self) -> None:
        """Проверка существования датасета"""
        if not self.dataset_path.exists():
            abs_path = self.dataset_path.absolute()
            raise FileNotFoundError(
                f"JSON датасет не найден по пути: {abs_path}\n"
                f"Убедитесь, что файл tech_stack_dataset.json находится в папке data/raw/"
            )
        else:
            logger.info(f"JSON датасет найден: {self.dataset_path.absolute()}")
    
    def _validate_json_structure(self, sample_record: Dict[str, Any]) -> None:
        """Валидация структуры JSON данных"""
        required_fields = ['project_description', 'project_type', 'tech_stack']
        
        for field in required_fields:
            if field not in sample_record:
                raise ValueError(f"Обязательное поле отсутствует: {field}")
        
        if not isinstance(sample_record.get('tech_stack', []), list):
            raise ValueError("Поле tech_stack должно быть списком")
    
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Загрузка датасета из JSON файла
        """
        logger.info(f"Загрузка JSON датасета из: {self.dataset_path}")
        
        try:
            # Чтение JSON файла
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Валидация структуры
            if not isinstance(data, list):
                raise ValueError("JSON должен содержать список проектов")
            
            if len(data) == 0:
                raise ValueError("JSON файл пустой")
            
            self._validate_json_structure(data[0])
            
            logger.info(f"Успешно загружено {len(data)} записей из JSON")
            
            # Если нужна выборка
            if sample_size and sample_size < len(data):
                data = data[:sample_size]
                logger.info(f"Используется выборка: {len(data)} записей")
            
            # Конвертация в DataFrame
            df = pd.DataFrame(data)
            
            # Переименование колонок для совместимости
            df = self._standardize_column_names(df)
            
            # Дополнительная обработка
            df = self._process_data(df)
            
            logger.info(f"Датасет подготовлен. Размер: {df.shape}")
            logger.info(f"Колонки: {df.columns.tolist()}")
            
            return df
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки JSON датасета: {e}")
            raise
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Стандартизация названий колонок для совместимости
        """
        column_mapping = {
            'project_description': 'description',
            'scale': 'project_scale', 
            'domain': 'industry',
            'subdomain': 'subdomain',
            'constraints': 'business_requirements',
            'description_style': 'project_format'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Убедимся, что все необходимые колонки существуют
        required_columns = ['description', 'project_type', 'project_scale', 'industry', 
                          'budget', 'team_size', 'complexity', 'technical_requirements',
                          'tech_stack', 'business_requirements']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Отсутствуют колонки: {missing_columns}")
        
        return df
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Дополнительная обработка данных
        """
        # Обработка технических требований
        if 'technical_requirements' in df.columns:
            # Убедимся, что это словари
            df['technical_requirements'] = df['technical_requirements'].apply(
                lambda x: x if isinstance(x, dict) else {}
            )
            logger.info("Колонка technical_requirements обработана")
        
        # Обработка tech_stack
        if 'tech_stack' in df.columns:
            # Убедимся, что это списки
            df['tech_stack'] = df['tech_stack'].apply(
                lambda x: x if isinstance(x, list) else []
            )
            
            # Статистика по технологиям
            all_tech = []
            for stack in df['tech_stack']:
                all_tech.extend(stack)
            
            logger.info(f"Всего уникальных технологий в датасете: {len(set(all_tech))}")
        
        # Обработка business_requirements
        if 'business_requirements' in df.columns:
            df['business_requirements'] = df['business_requirements'].apply(
                lambda x: x if isinstance(x, list) else []
            )
        
        # Заполнение пропущенных описаний
        if 'description' in df.columns:
            df['description'] = df['description'].fillna('Нет описания')
        
        # Заполнение пропусков в категориальных признаках
        categorical_columns = ['project_type', 'project_scale', 'industry', 'budget', 'team_size', 'complexity']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Получение подробной информации о датасете"""
        try:
            df = self.load_data(sample_size=1000)  # Для скорости берем выборку
        except Exception as e:
            logger.error(f"Не удалось загрузить данные для статистики: {e}")
            return {}
        
        info = {
            'total_projects': len(df),
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'project_types_distribution': df['project_type'].value_counts().to_dict(),
            'industries_distribution': df['industry'].value_counts().to_dict(),
            'budget_distribution': df['budget'].value_counts().to_dict(),
            'team_size_distribution': df['team_size'].value_counts().to_dict(),
            'complexity_distribution': df['complexity'].value_counts().to_dict(),
            'tech_stack_statistics': self._get_tech_stack_stats(df),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return info
    
    def _get_tech_stack_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Статистика по технологическим стекам"""
        if 'tech_stack' not in df.columns:
            return {}
        
        all_tech = []
        for stack in df['tech_stack']:
            if isinstance(stack, list):
                all_tech.extend(stack)
        
        from collections import Counter
        tech_counts = Counter(all_tech)
        
        return {
            'total_technologies_used': len(all_tech),
            'unique_technologies': len(tech_counts),
            'most_common_technologies': tech_counts.most_common(20),
            'avg_technologies_per_project': len(all_tech) / len(df) if len(df) > 0 else 0,
            'technology_frequency_distribution': dict(tech_counts)
        }
    
    def get_sample_projects(self, n: int = 5) -> List[Dict[str, Any]]:
        """Получение примеров проектов"""
        try:
            df = self.load_data(sample_size=n)
            samples = df.to_dict('records')
            return samples
        except Exception as e:
            logger.error(f"Не удалось получить примеры проектов: {e}")
            return []