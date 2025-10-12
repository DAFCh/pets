# !pip install sqlalchemy psycopg2 yadisk[sync-defaults] python-dotenv logger-config pandas numpy

from dotenv import load_dotenv
from yadisk import YaDisk
import os
from logger_config import configure_logging
import pandas as pd
import numpy as np
import logging
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, select, Column, Integer, String, Float, DateTime, Numeric
from sqlalchemy.sql import text
from tqdm import tqdm
import sys
import re
from pathlib import Path
from io import StringIO
import psycopg2

# функции:
# загрузка датасета с выводом общей информации
def load_and_inspect(file_path, date_column=None, date_format=None, sep=',', index_col=None, dtype=None, decimal='.', logger=None):
    """Загружает данные из CSV файла, очищает имена столбцов и выводит информацию о данных.

    Функция читает CSV файл в pandas DataFrame, приводит имена столбцов к формату snake_case,
    при необходимости преобразует указанный столбец с датами в формат datetime и выводит первые строки,
    общую информацию, количество пропусков, дубликатов и диапазон дат через логгер или print.

    Args:
        file_path (str): Путь к CSV файлу.
        date_column (str, optional): Название столбца с датами для преобразования в datetime.
        date_format (str, optional): Формат даты (например, '%Y%m%d%H') для столбца date_column.
        sep (str, optional): Разделитель в CSV файле. Defaults to ','.
        index_col (int or str, optional): Столбец для использования в качестве индекса DataFrame.
        dtype (dict, optional): Словарь с типами данных для столбцов (например, {'zip_code': str}).
        decimal (str, optional): Символ десятичного разделителя (например, '.' или ','). Defaults to '.'.
        logger (logging.Logger, optional): Объект логгера для вывода информации. Если None, используется print.

    Returns:
        pandas.DataFrame: Загруженный и обработанный DataFrame.

    Raises:
        FileNotFoundError: Если указанный файл не найден.
        pd.errors.ParserError: Если не удалось разобрать CSV файл.
        Exception: Для других ошибок при обработке данных (например, некорректный формат даты).

    Examples:
        >>> import pandas as pd
        >>> import logging
        >>> logging.basicConfig(level=logging.INFO)
        >>> logger = logging.getLogger(__name__)
        >>> df = load_and_inspect('cars.csv', date_column='sale_date', logger=logger)
        >>> print(df.head())

    Notes:
        Требуются библиотеки pandas, re и io. Если date_column указан, он должен существовать в данных.
    """
    # Функция логирования или печати
    def log(msg, level='info'):
        if logger:
            getattr(logger, level)(msg)
        else:
            print(msg)
    
    # Загружаем данные из CSV
    try:
        data = pd.read_csv(file_path, sep=sep, index_col=index_col, dtype=dtype, decimal=decimal)
    except FileNotFoundError:
        log(f"Файл '{file_path}' не найден.", level='error')
        raise
    except pd.errors.ParserError as e:
        log(f"Ошибка при разборе CSV файла: {e}", level='error')
        raise

    def clean_column_name(name):
        """Приводит имя столбца к формату snake_case."""
        cleaned = name.strip().lower().replace("-", "_")
        cleaned = re.sub(r'\s+', '_', cleaned)
        cleaned = re.sub(r'[^\w_]', '', cleaned)
        return cleaned

    # Приводим имена столбцов к snake_case
    data.columns = [clean_column_name(col) for col in data.columns]

    # Преобразование столбца с датами, если указан
    if date_column:
        date_column_clean = clean_column_name(date_column)
        if date_column_clean in data.columns:
            try:
                if date_format:
                    data[date_column_clean] = pd.to_datetime(data[date_column_clean], format=date_format)
                else:
                    data[date_column_clean] = pd.to_datetime(data[date_column_clean])
            except Exception as e:
                log(f"Ошибка при преобразовании даты в столбце '{date_column_clean}': {e}", level='error')
        else:
            log(f"Внимание: столбец '{date_column}' не найден после очистки имен (ищем '{date_column_clean}')", level='warning')

    # Выводим первые строки
    log('Данные (первые строки):')
    log(data.head().to_string())

    # Выводим общую информацию
    log('\nОбщая информация:')
    buffer = StringIO()
    data.info(buf=buffer)
    log(buffer.getvalue())

    # Выводим информацию о пропусках
    log('\nКоличество пропусков:')
    log(data.isna().sum().sort_values(ascending=False).to_string())

    # Выводим информацию о дубликатах
    duplicates = data.duplicated().sum()
    log(f'\nКоличество дубликатов: {duplicates}')

    # Выводим минимальную и максимальную дату, если столбец с датами существует
    if date_column and date_column_clean in data.columns:
        min_date = data[date_column_clean].min()
        max_date = data[date_column_clean].max()
        if pd.notna(min_date):
            log(f'\nМинимальная дата в столбце "{date_column_clean}": {min_date}')
        if pd.notna(max_date):
            log(f'Максимальная дата в столбце "{date_column_clean}": {max_date}')

    return data

# перекодирование типа топлива:
def encode_fuel_type(fuel):
    """Кодирует тип топлива в сокращенную форму: F (бензин), D (дизель), E (электро), HYB (гибрид).

    Функция принимает строку с типом топлива, приводит её к нижнему регистру и удаляет пробелы.
    Если тип топлива не соответствует предопределенным категориям, возвращается исходное значение.

    Args:
        fuel (str): Тип топлива (например, 'бензин', 'дизель', 'электро', 'гибрид'). Регистр и пробелы игнорируются.

    Returns:
        str: Кодированное значение ('F', 'D', 'E', 'HYB') или исходное значение, если тип не распознан.

    Examples:
        >>> encode_fuel_type('бензин')
        'F'
        >>> encode_fuel_type('ЭЛЕКТРО ')
        'E'
        >>> encode_fuel_type('неизвестно')
        'неизвестно'
    """
    value = str(fuel).lower().strip()
    if value == 'бензин':
        return 'F'
    if value == 'дизель':
        return 'D'
    if value == 'электро' or value == 'электричество':
        return 'E'
    if value == 'гибрид':
        return 'HYB'
    return fuel

# кодировка типа привода
def normalize_drive_type(val):
    """Стандартизирует тип привода автомобиля в категории '4WD', '2WD' или 'неизвестно'.

    Функция принимает значение типа привода, приводит его к строке, удаляет пробелы и приводит к нижнему регистру.
    Значения, соответствующие полному приводу, стандартизируются как '4WD', а передний или задний привод — как '2WD'.
    Пропущенные значения (NaN) возвращаются как 'неизвестно'. Все прочие значения возвращаются без изменений.

    Args:
        val (str or None): Тип привода автомобиля (например, 'полный', 'передний', '4x4', 'quattro'). 
                           Может быть None или NaN.

    Returns:
        str: Стандартизированный тип привода ('4WD', '2WD', 'неизвестно') или исходное значение, если оно не распознано.

    Examples:
        >>> normalize_drive_type('полный')
        '4WD'
        >>> normalize_drive_type('передний')
        '2WD'
        >>> normalize_drive_type('AWD')
        '4WD'
        >>> normalize_drive_type(None)
        'неизвестно'
        >>> normalize_drive_type('другое')
        'другое'
    """
    if pd.isna(val):
        return 'неизвестно'
    
    val_str = str(val).lower().strip().replace(' ', '')
    
    if val_str in ['quattro', 'awd', '4wd', '4x4', '4х4', '4motion', 'полный']:
        return '4WD'
    elif val_str in ['передний', 'задний', '2wd', '4x2', '4х2', 'передний(ff)', 'ff', 'rwd', 'fwd', '4х2.2']:
        return '2WD'
    else:
        return val  # оставляем как есть для дальнейшего анализа

# нормализация КПП
def classify_transmission(transmission):
    """Классифицирует тип трансмиссии в категории 'M' (механическая), 'A' (автоматическая) или 'неизвестно'.

    Функция принимает значение типа трансмиссии, приводит его к строке, удаляет пробелы и приводит к нижнему регистру.
    Использует списки ключевых слов для определения типа трансмиссии. Если значение не соответствует ни одной категории,
    возвращается исходное значение. Пропущенные значения (NaN) возвращаются как 'неизвестно'.

    Args:
        transmission (str or None): Тип трансмиссии (например, 'АКПП', 'MT', 'вариатор'). 
                                   Может быть None или NaN. Регистр и пробелы игнорируются.

    Returns:
        str: Код трансмиссии ('M' для механической, 'A' для автоматической, 'неизвестно' для NaN) 
             или исходное значение, если тип не распознан.

    Examples:
        >>> classify_transmission('АКПП')
        'A'
        >>> classify_transmission('6М')
        'M'
        >>> classify_transmission(None)
        'неизвестно'
        >>> classify_transmission('другое')
        'другое'
    """
    if pd.isna(transmission):
        return 'неизвестно'

    transmission = str(transmission).lower().strip()

    # Словари ключевых фрагментов
    manual_keywords = [
        'mt', 'мт', 'мкп', 'мкпп', 'м/т', 'm/t', 'мех',
        '5м', '5m', '6м', 'мt', 'м/t', 'tdi'
    ]
    auto_keywords = [
        'a', 'ат', 'акп', 'акпп', 'at', 'a/t', 'а', 'steptronic', 'tiptronic',
        'dsg', 's-tronic', 's tronic', 'cvt', 'вариатор', 'amt',
        'pdk', 'powershift', 'g-tronic', 'ступ акпп', 'dct',
        '6a', '6а', '8a', 'аt', 'редуктор', '8'
    ]

    if any(keyword in transmission for keyword in manual_keywords):
        return 'M'

    if any(keyword in transmission for keyword in auto_keywords):
        return 'A'

    return transmission

# нормализация объема двигателя
def clean_engine_capacity(val):
    """Преобразует значение объема двигателя в числовой формат (float).

    Функция принимает значение объема двигателя, приводит его к строке, удаляет пробелы и приводит к нижнему регистру.
    Затем удаляет все нечисловые символы, кроме цифр, точки и запятой, заменяет запятую на точку и очищает строку от
    некорректных числовых форматов (например, лишних точек). Если результат не является валидным числом, возвращается np.nan.

    Args:
        val: Значение объема двигателя (например, '1,6 MPI', '4.98 L', '2.0h', None). 
             Может быть любого типа, преобразуемого в строку.

    Returns:
        float: Объем двигателя в литрах (например, 1.6, 4.98).
        np.nan: Если входное значение None, NaN, пустое или не может быть преобразовано в число.

    Examples:
        >>> clean_engine_capacity('1,6 MPI')
        1.6
        >>> clean_engine_capacity('4.98 L')
        4.98
        >>> clean_engine_capacity('4.3.')
        4.3
        >>> clean_engine_capacity(None)
        nan
        >>> clean_engine_capacity('abc')
        nan

    Notes:
        Функция использует регулярные выражения для удаления нечисловых символов и очистки числового формата.
        Требуются библиотеки pandas, numpy и re.
    """
    if pd.isna(val):
        return np.nan

    # Приводим к строке и очищаем от пробелов
    val = str(val).strip().lower()

    # Оставляем только цифры, точку, запятую
    val = re.sub(r'[^0-9.,]', '', val)

    # Заменяем запятую на точку
    val = val.replace(',', '.')

    # Удаляем лишние точки: несколько точек подряд заменяем на одну, убираем точки в начале и конце
    val = re.sub(r'\.+', '.', val)
    val = val.strip('.')

    # Проверяем, является ли строка валидным числом
    try:
        return float(val) if val else np.nan
    except ValueError:
        return np.nan

# загрузка в БД
def pandas_to_sqlalchemy(df, table_class, db_url='sqlite:///cars_database.db', batch_size=1000):
    """Загружает данные из pandas DataFrame в указанную таблицу базы данных через SQLAlchemy.

    Функция создает таблицу на основе переданной модели SQLAlchemy, если она еще не существует, и загружает
    данные из DataFrame пакетами для оптимизации производительности. Прогресс загрузки отображается
    с помощью прогресс-бара tqdm. Столбцы DataFrame должны соответствовать полям модели.

    Args:
        df (pandas.DataFrame): DataFrame с данными, содержащий столбцы, соответствующие полям модели.
        table_class: Класс SQLAlchemy, описывающий таблицу (например, CarData, GeoPolygons).
        db_url (str, optional): URL базы данных (например, 'sqlite:///cars_database.db'). 
                                По умолчанию используется SQLite.
        batch_size (int, optional): Размер пакета для загрузки данных (по умолчанию 1000).

    Returns:
        None: Функция ничего не возвращает, но выводит сообщение об успешной загрузке.

    Raises:
        ValueError: Если DataFrame не содержит всех необходимых столбцов для модели.
        Exception: Если происходит ошибка при загрузке данных, транзакция откатывается, 
                   и исключение передается дальше.

    Examples:
        >>> import pandas as pd
        >>> df = pd.read_csv("cars.csv", parse_dates=['sale_date'])
        >>> pandas_to_sqlalchemy(df, CarData, db_url='sqlite:///cars_database.db', batch_size=1000)
        Загружено 5000 строк в таблицу car_sales.
    """
    # Проверка наличия необходимых столбцов
    required_columns = [c.name for c in table_class.__table__.columns if c.name != 'id']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame не содержит всех необходимых столбцов для модели {table_class.__tablename__}")

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    Base.metadata.create_all(engine)

    try:
        # Загрузка данных с прогресс-баром
        for i in tqdm(range(0, len(df), batch_size), desc=f"Загрузка данных в {table_class.__tablename__}"):
            batch = df.iloc[i:i + batch_size]
            records = [table_class(**row.to_dict()) for _, row in batch.iterrows()]
            session.add_all(records)
            session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка при загрузке данных в {table_class.__tablename__}: {e}")
        raise
    finally:
        session.close()

    logger.info(f"Загружено {len(df)} строк в таблицу {table_class.__tablename__}.")

# Загрузка окружения
load_dotenv()

YADISK_TOKEN = os.getenv('YADISK_TOKEN')
DB_URL = os.getenv('DATABASE_URL')
DATABASE_KEY = os.getenv('DATABASE_KEY')
USER = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
DBNAME = os.getenv('DBNAME')

DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

# === Режим разработки ===
DEBUG_MODE = True  # False — для продакшена, True — для отладки

# === Логгер ===
logger_name = 'root_logger'
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
configure_logging(logger_name, log_dir='logs', log_level=log_level)
logger = logging.getLogger(logger_name)

logger.info(f"Скрипт запущен. Режим отладки: {DEBUG_MODE}")

# подключаемся к Яндекс.Диску и забираем файл данными
y = YaDisk(token=YADISK_TOKEN)

# Путь для сохранения файлов
DOWNLOAD_DIR = Path('download')

if y.check_token():
    try:
        # Создаем директорию для загрузки, если её нет
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Перебираем файлы в директории ETL_pet_auto
        for el in y.listdir('ETL_pet_auto'):
            if 'autokz2019' in el['path']:
                file = el['path']
                try:
                    # Формируем путь для сохранения
                    downloaded_file = DOWNLOAD_DIR / file.split('/')[-1]
                    
                    # Скачиваем файл
                    y.download(file.split(':')[1], str(downloaded_file))
                    
                    # Логируем успешную загрузку
                    logger.info(
                        f"Файл {file.split('/')[-1]} успешно скачан, "
                        f"размер: {downloaded_file.stat().st_size} байт"
                    )
                    
                    # Загружаем данные для обработки
                    data = load_and_inspect(
                        str(downloaded_file),
                        sep=';',
                        decimal=',',
                        logger=logger
                    )
                    
                except Exception as e:
                    logger.error(f"Ошибка при скачивании файла {file}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Ошибка при работе с Яндекс.Диском: {str(e)}")
        raise
else:
    logger.error("Токен недействителен. Проверьте переменную окружения YADISK_TOKEN.")
    raise ValueError("Недействительный токен Яндекс.Диска")

data = load_and_inspect(downloaded_file, sep=';', decimal=',', logger=logger)

# переименовываем столбцы на english_snake_case
# Словарь переименования
column_rename_map = {
    'год': 'year',
    'месяц': 'month',
    'компания': 'company',
    'бренд': 'brand',
    'модель': 'model',
    'модификация': 'modification',
    'год_выпуска': 'release_year',
    'страна_производитель': 'country_origin',
    'вид_топлива': 'fuel_type',
    'объём_двиг_л': 'engine_capacity_l',
    'коробка_передач': 'transmission',
    'тип_привода': 'drive_type',
    'сегмент': 'segment',
    'регион': 'region',
    'наименование_дилерского_центра': 'dealer',
    'тип_клиента': 'client_type',
    'форма_расчета': 'payment_type',
    'количество': 'quantity',
    'цена_usd': 'price_usd',
    'продажа_usd': 'total_usd',
    'область': 'province',
    'сегментация_2013': 'segment_2013',
    'класс_2013': 'class_2013',
    'сегментация_eng': 'segment_eng',
    'локализация_производства': 'localization',
}

data = data.rename(columns=column_rename_map)

# удалим столбцы форма_расчета, сегмент, наименование_дилерского_центра, тип_клиента, сегментация_eng, локализация_производства, модификацию пока оставим
data = data.drop(columns=[
    'payment_type', 
    'segment', 
    'dealer', 
    'client_type', 
    'segment_eng', 
    'localization',
])

# Подсчитываем, сколько значений '#Н/Д' в таблице перед заменой
count_before = (data == '#Н/Д').sum().sum()

# Заменяем все вхождения '#Н/Д' на пропуски (NaN)
data.replace('#Н/Д', np.nan, inplace=True)

# Подсчитываем, сколько значений '#Н/Д' осталось (должно быть 0)
count_after = (data == '#Н/Д').sum().sum()

# Выводим результат
logger.info(f"Заменено значений '#Н/Д' на NaN: {count_before - count_after}")

# обработаем пропуски
count_missing = data[
    data[['drive_type', 'engine_capacity_l', 'fuel_type', 'transmission']].isna().any(axis=1)
].shape[0]

logger.info(f"Количество строк с пропусками в технических характеристиках: {count_missing}")

# Убираем пробелы в modification для Renault
mask = data['brand'] == 'Renault'
data.loc[mask, 'modification'] = (
    data.loc[mask, 'modification']
    .astype(str)
    .str.upper()
    .str.replace(r'\s+', '', regex=True)
)

# Общие правила по бренду и модели
fill_rules = {
    ('Renault Россия', 'Renault', 'Sandero'): {
        'drive_type': 'передний',
        'fuel_type': 'бензин'
    },
    ('Renault Россия', 'Renault', 'Logan'): {
        'drive_type': 'передний',
        'fuel_type': 'бензин'
    },
    ('Renault Россия', 'Renault', 'Duster'): {
        'fuel_type': 'бензин'
    },
    ('Renault Россия', 'Renault', 'Kaptur'): {
        'fuel_type': 'бензин'
    },
    ('Mercur Auto', 'Volkswagen', 'Polo'): {
        'drive_type': 'передний',
        'fuel_type': 'бензин'
    },
    ('Равон Моторс Казахстан', 'Ravon', 'Nexia R3'): {
        'drive_type': 'передний',
        'fuel_type': 'бензин'
    },
    ('УзАвто-Казахстан', 'Ravon', 'Nexia R3'): {
        'drive_type': 'передний',
        'fuel_type': 'бензин'
    },
    ('ТК КАМАЗ', 'KAMAZ', '43118'): {
        'drive_type': 'полный',
        'fuel_type': 'дизель',
        'engine_capacity_l': 10.85
    },
    ('ТК КАМАЗ', 'KAMAZ', '6520'): {
        'engine_capacity_l': 11.76
    },
        ('ТК КАМАЗ', 'KAMAZ', '45143'): {
        'engine_capacity_l': 10.85
    },
    ('ТК КАМАЗ', 'KAMAZ', '65115'): {
        'drive_type': 'задний',
        'fuel_type': 'дизель',
        'engine_capacity_l': 10.85
    },
    ('Allur Auto', 'Jac', 'S3'): {
        'drive_type': 'передний',
        'fuel_type': 'бензин'
    },
    ('Allur Auto', 'Jac', 'S5'): {
        'drive_type': 'передний',
        'fuel_type': 'бензин'
    },
    ('Allur Auto', 'ANKAI', 'HFF6850G'): {
        'drive_type': 'задний',
        'fuel_type': 'дизель'
    },
    ('Вираж', 'GAZ', '3302'): {
        'drive_type': 'задний'
    },
    ('Вираж', 'GAZ', 'Next'): {
        'drive_type': 'задний'
    },
    ('СВС-ТРАНС', 'Isuzu', 'NMR'): {
        'drive_type': 'задний',
        'fuel_type': 'дизель'
    }
}

# Специфические правила по модификации
modification_rules = {
    # Renault Sandero
    ('Renault Россия', 'Renault', 'Sandero', 'SXPA16PA5RB'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'CXPA16MV5RB'): {
        'transmission': 'механическая', 'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'SXPA16M5RB'): {
        'transmission': 'механическая', 'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'SUTA16PA5RB'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'CUTA16MV5RB'): {
        'transmission': 'механическая', 'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'SXPA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'SUTA16M5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'SUTA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'STW16PA5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'ACCA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'SXPA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'SUTA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'SUTA16M5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'STW16PA5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Sandero', 'ACCA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },

    # Renault Dokker
    ('Renault Россия', 'Renault', 'Dokker', 'YKLAURMVEM'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },

    # Renault Arkana
    ('Renault Россия', 'Renault', 'Arkana', 'B32M1TX5C'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Arkana', 'B22M1TX5C'): {
        'engine_capacity_l': 1.3, 'drive_type': '2WD'
    },

    # Renault Logan
    ('Renault Россия', 'Renault', 'Logan', 'AUTA16K5RB'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'CXPA16MV5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'ACCA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'EXPA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'CUTA16MV5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'SXPA16M5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'DYNA16PA5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'AUTA16PA5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'SUTA16PA5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'CXPA16MV5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'ACCA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'EXPA16K5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'CUTA16MV5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'SXPA16M5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'DYNA16PA5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'AUTA16PA5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },
    ('Renault Россия', 'Renault', 'Logan', 'SUTA16PA5RB'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD'
    },

    # Renault Kaptur
    ('Renault Россия', 'Renault', 'Kaptur', 'ZB34AAA5C'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 2.0, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Kaptur', 'MB34AAA5C'): {
        'engine_capacity_l': 1.6, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Kaptur', 'PLY4AAA5C'): {
        'engine_capacity_l': 1.6, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Kaptur', 'ZB12JAX5C'): {
        'engine_capacity_l': 1.2, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Kaptur', 'MB12JAX5C'): {
        'engine_capacity_l': 1.2, 'drive_type': '4WD'
    },

    # Ravon Nexia
    ('УзАвто-Казахстан', 'Ravon', 'Nexia R3', 'ELEGANT AT'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 1.5
    },

    # Jac
    ('Allur Auto', 'Jac', 'S5', 'Intelligent 2.0T MT6'): {
        'transmission': 'механическая', 'engine_capacity_l': 2.0
    },
    ('Allur Auto', 'Jac', 'S3', 'Intelligent 1.6 CVT'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 1.6
    },
    ('Allur Auto', 'Jac', 'S3', 'Luxury'): {
        'transmission': 'автоматическая'
    },
    ('Allur Auto', 'Jac', 'S5', 'FL Intelligent'): {
        'transmission': 'автоматическая'
    },

    # Shacman
    ('СемАЗ', 'Shacman', 'SX3258DR384', 'Автомобиль-самосвал SHACMAN Евро5 модель SX3258DR384'): {
        'transmission': 'механическая'
    },

    # Renault Duster
    ('Renault Россия', 'Renault', 'Duster', 'Z2GB4AG'): {
        'transmission': 'механическая', 'engine_capacity_l': 2.0, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Duster', 'Z2GB4AGA'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 2.0, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Duster', 'Z1FB4AG'): {
        'transmission': 'механическая', 'engine_capacity_l': 2.0, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Duster', 'Z1FB4AGA'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 2.0, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Duster', 'Z2PGB4AGA'): {
        'transmission': 'автоматическая', 'engine_capacity_l': 2.0, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Duster', 'Z2PGB4AG'): {
        'transmission': 'механическая', 'engine_capacity_l': 2.0, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Duster', 'ADGB4AG'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD', 'transmission': 'механическая'
    },
    ('Renault Россия', 'Renault', 'Duster', 'ADGB4AGA'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD', 'transmission': 'автоматическая'
    },
    ('Renault Россия', 'Renault', 'Duster', 'Z1FB2JA'): {
        'engine_capacity_l': 1.5, 'drive_type': '4WD'
    },
    ('Renault Россия', 'Renault', 'Duster', 'E2GB4AGA'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD', 'transmission': 'автоматическая'
    },
    ('Renault Россия', 'Renault', 'Duster', 'E2GB4AG'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD', 'transmission': 'механическая'
    },
    ('Renault Россия', 'Renault', 'Duster', 'Z0DB2JA'): {
        'engine_capacity_l': 1.5, 'drive_type': '2WD'  # дизель
    },
    ('Renault Россия', 'Renault', 'Duster', 'DKGB4AG'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD', 'transmission': 'механическая'
    },
    ('Renault Россия', 'Renault', 'Duster', 'DKGB4AGA'): {
        'engine_capacity_l': 1.6, 'drive_type': '2WD', 'transmission': 'автоматическая'
    },
}

# Счётчики заполнений
filled_counts = {col: 0 for col in ['drive_type', 'engine_capacity_l', 'fuel_type', 'transmission']}
total_filled = 0

for idx, row in data.iterrows():
    key3 = (row['company'], row['brand'], row['model'])
    key4 = (row['company'], row['brand'], row['model'], row['modification'])

    # Общие правила
    if key3 in fill_rules:
        for col, value in fill_rules[key3].items():
            if pd.isna(row[col]):
                data.at[idx, col] = value
                filled_counts[col] += 1
                total_filled += 1

    # Правила по модификации
    if key4 in modification_rules:
        for col, value in modification_rules[key4].items():
            if pd.isna(row[col]):
                data.at[idx, col] = value
                filled_counts[col] += 1
                total_filled += 1

# Логируем статистику заполнения
logger.info("═" * 50)
logger.info("СТАТИСТИКА ЗАПОЛНЕНИЯ ПРОПУСКОВ")
logger.info("═" * 50)
for col, count in filled_counts.items():
    logger.info(f"{col:<20} | Заполнено пропусков: {count}")
logger.info("═" * 50)
logger.info(f"ИТОГО: {total_filled} пропусков заполнено")
logger.info("═" * 50)

# Логируем оставшиеся пропуски
remaining_nulls = data[['drive_type', 'engine_capacity_l', 'fuel_type', 'transmission']].isna().sum()
logger.info("\nОСТАВШИЕСЯ ПРОПУСКИ:")
for col, cnt in remaining_nulls.items():
    logger.info(f"{col}: {cnt}")

# удалим модификацию, теперь она не нужна:
data = data.drop(columns=[
    'modification'
])

# Количество строк до удаления
before = data.shape[0]

# Удаляем строки с любыми пропусками
data = data.dropna(how='any')

# Количество строк после удаления
after = data.shape[0]

# Разница и доля потерь
removed = before - after
loss_pct = removed / before * 100

logger.info(f'Было строк: {before}')
logger.info(f'Удалено строк с пропусками: {removed} ({loss_pct:.2f}%)')
logger.info(f'Осталось строк: {after}')

# проверим дубликаты
logger.info(f'Количество дубликатов: {data.duplicated().sum()}. Не трогаем, по словам заказчика - ОК')

# ------------------
# Обработка столбцов:
# Страны производства:
# выгрузим список стран с сайта
countries = pd.read_table('https://www.artlebedev.ru/country-list/tab/')
# Создание словарей для быстрого доступа к кодам стран
name_dict = dict(zip(countries['name'], countries['alpha3']))
full_name_dict = dict(zip(countries['fullname'], countries['alpha3']))
# Заменим UK на Соединенное Королевство и остальные неформаты
data['country_origin'] = data['country_origin'].str.replace('UK', 'Соединенное Королевство')
data['country_origin'] = data['country_origin'].str.replace('Корея', 'Республика Корея')
data['country_origin'] = data['country_origin'].str.replace('Белоруссия', 'Беларусь')
data['country_origin'] = data['country_origin'].str.replace('США', 'Соединенные Штаты Америки')
# Применяем оба словаря для замены кодов стран
data['country_origin'] = data['country_origin'].map(lambda x: name_dict.get(x, full_name_dict.get(x, x)))
logger.info('Страны перекодированы в alpha3')

# ------------------
# топливо
logger.info(f"Общее количество пропусков в топливе: {data.loc[data['fuel_type'].isin(['2', '1,6', '0'])].shape[0]}")
# удалим строки с неверно заполненным типом топлива
data = data.loc[~data['fuel_type'].isin(['2', '1,6', '0'])]

# кодируем тип топлива:
data.loc[:, 'fuel_type'] = data['fuel_type'].apply(encode_fuel_type)
logger.info('Тип топлива - перекодировано в F, D, HYB, E')

# ------------------
# Тип привода:
# Правила для типа привода
drive_type_rules = {
    # Duster
    ('Renault', 'Duster', 'RUS', '2018', 'F', 1.6, 'автоматическая'): '2WD',
    ('Renault', 'Duster', 'RUS', '2018', 'F', 1.6, 'механическая'): '2WD',
    
    # Kaptur
    ('Renault', 'Kaptur', 'RUS', '2018', 'F', 2.0, 'автоматическая'): '4WD',
    
    # Sandero
    ('Renault', 'Sandero', 'RUS', '2018', 'F', 1.6, 'автоматическая'): '2WD'
}

# Создаем маску для строк с drive_type == '0'
mask = data['drive_type'] == '0'

# Для строк с mask формируем ключи для словаря
keys = list(zip(
    data.loc[mask, 'brand'],
    data.loc[mask, 'model'],
    data.loc[mask, 'country_origin'],
    data.loc[mask, 'release_year'],
    data.loc[mask, 'fuel_type'],
    data.loc[mask, 'engine_capacity_l'],
    data.loc[mask, 'transmission']
))

# Счётчик успешных замен
replaced_count = 0

# Обходим индексы и ключи, заменяем по словарю
for idx, key in zip(data.loc[mask].index, keys):
    if key in drive_type_rules:
        data.at[idx, 'drive_type'] = drive_type_rules[key]
        replaced_count += 1

# Вывод результата
logger.info(f'Заменено drive_type "0" на значение из правил: {replaced_count}')

# Применяем нормализацию
data['drive_type'] = data['drive_type'].map(normalize_drive_type)

# Логируем результат
unique_values = data['drive_type'].value_counts()
logger.info("Нормализация drive_type завершена.")
if DEBUG_MODE:
    logger.debug(f"Список уникальных значений: {unique_values}")

# ------------------
# коробка передач
data['transmission'] = data['transmission'].apply(classify_transmission)
# Считаем распределение
transmission_counts = data['transmission'].value_counts()

logger.info("Нормализация transmission завершена.")
if DEBUG_MODE:
    logger.debug(f"Распределение типов трансмиссии:\n{transmission_counts.to_string()}")

# ------------------
# количество
try:
    data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce')  # Преобразуем в числа, невалидные значения станут NaN
except ValueError:
    logger.info("В 'quantity' есть нечисловые значения. Проверьте данные.")

# 3. Преобразуем в int64, отбрасывая дробную часть
data['quantity'] = data['quantity'].astype('int64')

# Проверяем результат
logger.info(f"Количество успешно переведено в формат: {data['quantity'].dtype}")  # Должно вывести: int64

# ------------------
# месяц продажи
# Соответствие русских названий номерам месяцев
month_map = {
    'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4,
    'Май': 5, 'Июнь': 6, 'Июль': 7, 'Август': 8,
    'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12
}
# Преобразуем русские месяцы в номера
data['month'] = data['month'].map(month_map)

# Формируем дату конца месяца
data['sale_date'] = pd.to_datetime(
    data['year'].astype(str) + '-' + data['month'].astype(str) + '-01'
) + pd.offsets.MonthEnd(0)
# принудительно укажем тип данных
data['sale_date'] = pd.to_datetime(data['sale_date'], errors='coerce')
# удалим ненужные теперь столбцы
data = data.drop(columns=['year', 'month'])
logger.info('Получена дата продажи "год-месяц-последний день"')

# ------------------
# компании
# создаем словарь для замены
company_merge_map = {
    'Автокапитал': 'Autokapital',
    'Равон Моторс Казахстан': 'Ravon Motors Kazakhstan',
    'Ravon Motors Kazakstan': 'Ravon Motors Kazakhstan',
    'Каспиан Моторс': 'Caspian Motors',
    'Hino Motors': 'Хино Моторс Казахстан',
    'Mercur Autos': 'Mercur Auto',
    'ММС Рус': 'MMC RUS',
}
# применяем замену
data['company'] = data['company'].astype(str).str.strip()
data['company'] = data['company'].replace(company_merge_map)
logger.info('Убрали повторы из названий компаний')
if DEBUG_MODE:
    logger.debug(f"Названиия компаний:\n{data['company'].value_counts().to_string()}")

# ------------------
# Объем двигателя
# заменяем значение engine_capacity_l для электромобилей на 0
data.loc[data['fuel_type']=='E', 'engine_capacity_l'] = 0
# Применяем к столбцу нормализацию
data['engine_capacity_l'] = data['engine_capacity_l'].apply(clean_engine_capacity)
# обработка аномалий у Niva Chevrolet
data.loc[
    (data['company'] == 'Вираж')&
    (data['brand'] == 'Chevrolet')&
    (data['model'] == 'Niva')&
    (data['release_year'] == '2019')&
    (data['engine_capacity_l'] > 1.7),
    ['engine_capacity_l']
] = 1.7
# Volkswagen Polo
data.loc[
    (data['model'] == 'Polo') &
    (data['release_year'] == 2018)&
    (data['engine_capacity_l'] > 2),
    'engine_capacity_l'
] = 1.6
logger.info('Объем двигателя - обработано')

# ------------------
# регион и область
# редактируем значения в столбцах
data['region'] = data['region'].str.strip().str.lower().str.capitalize()
data['province'] = data['province'].str.strip().str.lower().str.capitalize()
# уберем Г. из названий области
data.loc[data['province']=='Г.алматы', 'province'] = 'Алматы'
data.loc[data['province']=='Г.нур-султан', 'province'] = 'Нур-султан'
# заменяем ЮКО на Шымкент
data.loc[data['province']=='Южно-казахстанская область', 'province'] = 'Шымкент'
logger.info('Названия регионов - обработано')

# ------------------
# сегменты:
# присвоим значения сегмента
data['segment'] = 'b2c'
data.loc[
    (data['segment_2013'] == 'Коммерческие автомобили') | (data['quantity'] > 2),
    'segment'
] = 'b2b'
logger.info('Добавили разделение на b2b и b2c')

# ------------------
# год выпуска
# заменим кракозябры
data['release_year'] = data['release_year'].str.replace('\xa0', '')
# приведем год выпуска к числовому типу
data['release_year'] = data['release_year'].astype('int64')
logger.info('Год выпуска - обработано')

# ------------------
# Количество строк после обработки
after = data.shape[0]

# Разница и доля потерь
removed = before - after
loss_pct = removed / before * 100

logger.info(f'Было строк: {before}')
logger.info(f'Удалено строк: {removed} ({loss_pct:.2f}%)')
logger.info(f'Осталось строк: {after}')

# Сохраняем очищенные данные
data.to_csv('./download/clean_data.csv', encoding='utf8')
logger.info(f'Очищенные данные сохранены. Записей: {data.shape[0]}, признаков: {data.shape[1]}')

# открываем очищенные данные
finish = pd.read_csv('./download/clean_data.csv', index_col=0, encoding='utf8')

# Загрузка геоданных
geo_polygons = load_and_inspect('./download/datalens_geopolygons.csv', sep=';')[['geopolygon', 'province_ru']].dropna()
geo_points = load_and_inspect('./download/datalens_geopoints.csv', sep=';')[['geopoint', 'province_ru']].dropna()

# создаем подключение к БД
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# Модель для car_sales
class CarData(Base):
    __tablename__ = 'car_sales'

    id = Column(Integer, primary_key=True, autoincrement=True)
    company = Column(String)
    brand = Column(String)
    model = Column(String)
    release_year = Column(Integer)
    country_origin = Column(String)
    fuel_type = Column(String)
    engine_capacity_l = Column(Numeric(5, 2))
    transmission = Column(String)
    drive_type = Column(String)
    region = Column(String)
    quantity = Column(Integer)
    price_usd = Column(Float)
    total_usd = Column(Float)
    province = Column(String)
    segment_2013 = Column(String)
    class_2013 = Column(String)
    sale_date = Column(DateTime)
    segment = Column(String)

# Модель для geopolygons
class GeoPolygons(Base):
    __tablename__ = 'geopolygons'
    id = Column(Integer, primary_key=True, autoincrement=True)
    geopolygon = Column(String)
    province_ru = Column(String)

# Модель для geopoints
class GeoPoints(Base):
    __tablename__ = 'geopoints'
    id = Column(Integer, primary_key=True, autoincrement=True)
    geopoint = Column(String)
    province_ru = Column(String)

# Сбрасываем базы данных 
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Загрузка данных в таблицы
pandas_to_sqlalchemy(finish, CarData, db_url=DATABASE_URL)
pandas_to_sqlalchemy(geo_polygons, GeoPolygons, db_url=DATABASE_URL)
pandas_to_sqlalchemy(geo_points, GeoPoints, db_url=DATABASE_URL)

# Проверяем сохранение
try:
    # Проверка car_sales
    stmt = select(CarData).order_by(CarData.sale_date.desc()).limit(10)
    cars = session.execute(stmt).scalars().all()
    
    logger.info('Проверка загрузки данных в car_sales:')
    logger.info("Последние 10 автомобилей (по дате продажи):")
    for car in cars:
        logger.info(f"ID: {car.id}, Модель: {car.model}, Бренд: {car.brand}, Цена: ${car.price_usd:.2f}, Дата продажи: {car.sale_date}")

    # Проверка связи с geopolygons
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM car_sales cs JOIN geopolygons gp ON cs.province = gp.province_ru")).fetchone()
        logger.info(f"Связанных записей car_sales с geopolygons: {result[0]}")

    # Проверка связи с geopoints
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM car_sales cs JOIN geopoints gp ON cs.province = gp.province_ru")).fetchone()
        logger.info(f"Связанных записей car_sales с geopoints: {result[0]}")

except Exception as e:
    logger.error(f"Ошибка при проверке данных: {e}")
finally:
    session.close()