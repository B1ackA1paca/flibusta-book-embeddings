import os
import re
import json
import requests
import pandas as pd

# --- НАСТРОЙКИ ---
BASE_DIR = "/data/flibusta/data"
CHECKPOINT_DIR = "/data/flibusta/checkpoints"
CONSOLIDATED_FILE = "/data/flibusta/checkpoints/passports_consolidated.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

# Список целевых фантастических жанров Флибусты для фильтрации датасета
SF_TAGS = {
    'sf_fantasy', 'sf', 'sf_action', 'popadancy', 
    'sf_space', 'sf_heroic', 'sf_humor', 'sf_history', 'sf_social'
}

# --- УНИВЕРСАЛЬНЫЕ ПРОМПТЫ ДЛЯ ЛЮБОГО ЖАНРА ---
CHUNK_SYSTEM_PROMPT = """Ты — литературный аналитик. Твоя задача — извлечь объективные факты и особенности из фрагмента книги любого жанра.
Сделай краткую выжимку (до 100 слов), отразив:
1. СЮЖЕТ И ДЕКОРАЦИИ: Где происходит действие (сеттинг) и ключевые события.
2. ГЛАВНЫЙ ГЕРОЙ: Его ключевые черты характера, стиль поведения и то, какое впечатление/реакцию (шок, удивление, страх, восхищение, недоумение) он вызывает у окружающих людей.
3. ТЕМЫ И ТРОПЫ: Очевидные литературные приемы или штампы (например: социальное неравенство, месть, запретная любовь, выживание).
4. ОТНОШЕНИЯ: Наличие романтических линий, любовных многоугольников или их полное отсутствие.
Пиши строго фактами, без пафосной воды."""

FINAL_SYSTEM_PROMPT = """Ты — эксперт по анализу художественной литературы. Перед тобой выжимки из разных частей одной книги.
Объедини их в структурированный "Семантический паспорт книги", заполнив строго разделы:
1. СЮЖЕТ И ЛОКАЦИИ: Главная сюжетная арка, места действия, динамика событий. Обязательно укажи наличие или ОТСУТСТВИЕ любовных линий / гаремов.
2. ГЛАВНЫЙ ГЕРОЙ И ОКРУЖАЮЩИЕ: Характер персонажа, его особенности и то, как мир реагирует на его действия (в шоке ли они, уважают, боятся или недоумевают).
3. КЛЮЧЕВЫЕ ТРОПЫ: Выдели основные тэги произведения через запятую (например: Детектив, Трикстер, Без романтики, Сильный герой, Академия, Историческая драма).
4. ТОН И АТМОСФЕРА: Оцени стиль произведения (комедия, трагедия, драма, легкое развлечение, фарс, мрачный реализм).

Выдай результат на русском языке. Будь краток и точен."""

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def extract_text_from_fb2(filepath):
    """Быстрый и чистый сборщик абзацев из fb2 без тяжелых зависимостей."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    paragraphs = re.findall(r'<p>(.*?)</p>', content, flags=re.DOTALL)
    cleaned = [re.sub(r'<.*?>', '', p).strip() for p in paragraphs]
    return " ".join([p for p in cleaned if p])

def get_book_chunks(text, num_chunks=8, chunk_size=3500):
    """Берет ~1/3 книги, равномерно распределяя num_chunks по всему тексту."""
    total_len = len(text)
    if total_len < num_chunks * chunk_size:
        step = max(1, total_len // num_chunks)
    else:
        step = total_len // num_chunks
        
    chunks = []
    for i in range(num_chunks):
        start = i * step
        end = min(start + chunk_size, total_len)
        chunks.append(text[start:end])
    return chunks

def call_ollama(prompt, system_prompt):
    """Запрос к локальному Ollama API."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.2, # Низкая температура для стабильного извлечения фактов
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return response.json().get('response', '')
    except Exception as e:
        print(f"Ошибка запроса к Ollama: {e}")
        return ""

# --- ОЧИСТКА, УМНАЯ ДЕДУПЛИКАЦИЯ И ФИЛЬТР ЖАНРОВ ---

def clean_and_deduplicate(df):
    print(f"Запущена очистка. Исходно строк: {len(df)}")
    
    # 0. Явное копирование датафрейма для избежания SettingWithCopyWarning
    df = df.copy()
    
    # Приводим критические колонки к числовому типу
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['del'] = pd.to_numeric(df['del'], errors='coerce')
    df['file'] = pd.to_numeric(df['file'], errors='coerce')
    
    # [FIX] Сразу отфильтровываем битые/пустые ID книг и приводим file к целочисленному типу
    df = df.dropna(subset=['file']).copy()
    df['file'] = df['file'].astype(int)
    
    # 1. Оставляем ТОЛЬКО фантастику и фэнтези на основе нашего списка тегов
    df = df[df['genre'].fillna('').apply(lambda x: any(t in SF_TAGS for t in x.split(':')))]
    print(f"После фильтрации только фантастических жанров (sf_tags): {len(df)}")
    
    # Исключаем удаленные/битые книги и берем только русский сегмент
    df = df[df['del'] == 0]
    df = df[df['lang'] == 'ru'].copy()
    
    # Фильтр по минимальному размеру (чтобы отсечь мелкие рассказы/статьи)
    df = df[df['size'] >= 250000]
    
    # Быстрая очистка названий от черновиков (с использованием незахватывающей группы)
    draft_pattern = r'(?i)\b(?:глава|гл\.|продолжение|черновик|кусок|фрагмент|отрывок|обновлено|обновление)\b'
    df = df[~df['title'].str.contains(draft_pattern, na=False, regex=True)]
    
    # Создаем нормализованное название для дедупликации Самиздата
    def normalize_title(t):
        if pd.isna(t): return ""
        t = t.lower()
        t = re.sub(r'\[си\]|\(си\)', '', t)
        t = re.sub(r'[^а-яёa-z0-9]', '', t)
        return t.strip()
    
    df['norm_title'] = df['title'].apply(normalize_title)
    
    # Умная векторная дедупликация:
    # dropna=False предотвращает случайное удаление книг без указанного автора
    max_sizes = df.groupby(['author', 'norm_title'], dropna=False)['size'].transform('max')
    df_candidates = df[df['size'] >= max_sizes * 0.95]
    
    df_clean = df_candidates.sort_values('file').drop_duplicates(
        subset=['author', 'norm_title'], 
        keep='last',
        ignore_index=True
    ).drop(columns=['norm_title'])
    
    print(f"Очистка завершена. Подготовлено качественных книг фантастики: {len(df_clean)}")
    return df_clean

# --- ОСНОВНОЙ ПАЙПЛАЙН СУММАРИЗАЦИИ ---

def run_dataset_generation(df_clean):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Запуск генерации семантических паспортов для {len(df_clean)} книг...")
    
    for idx, row in df_clean.iterrows():
        book_id = str(row['file'])
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{book_id}.json")
        
        # Чекпоинт: если книга уже есть в папке checkpoints, просто идем дальше
        if os.path.exists(checkpoint_path):
            continue
            
        print(f"\n[{idx + 1}/{len(df_clean)}] Книга: {row['title']} (ID: {book_id})")
        
        try:
            # [FIX] Находим имя папки на диске методом динамического перебора всех возможных вариантов префиксов
            opts = [row['archive'], f"f.{row['archive']}", f"d.{row['archive']}"]
            folder_name = next((o for o in opts if os.path.exists(os.path.join(BASE_DIR, o))), None)
            
            if not folder_name:
                print(f"Папка {row['archive']} не найдена на диске (пробовали варианты f., d. и без префикса) в {BASE_DIR}")
                continue
                
            book_path = os.path.join(BASE_DIR, folder_name, f"{book_id}.fb2")
            
            if not os.path.exists(book_path):
                # На всякий случай пробуем расширение в верхнем регистре (.FB2)
                book_path_caps = os.path.join(BASE_DIR, folder_name, f"{book_id}.FB2")
                if os.path.exists(book_path_caps):
                    book_path = book_path_caps
                else:
                    print(f"Файл не найден на диске: {book_path}")
                    continue
            
            # Читаем текст и берем срезы общей сложностью ~1/3 книги
            full_text = extract_text_from_fb2(book_path)
            if len(full_text) < 10000:
                print("Файл пуст или поврежден. Пропускаем.")
                continue
                
            chunks = get_book_chunks(full_text, num_chunks=8, chunk_size=3500)
            
            # Шаг 1: Почаночная выжимка
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"  -> Анализ фрагмента {i+1}/8...")
                summary = call_ollama(prompt=chunk, system_prompt=CHUNK_SYSTEM_PROMPT)
                if summary:
                    chunk_summaries.append(summary)
            
            if not chunk_summaries:
                print("Отказ генерации выжимок фрагментов. Пропускаем.")
                continue
                
            # Шаг 2: Финальный паспорт книги
            print("  -> Сборка финального паспорта...")
            combined_summaries = "\n\n".join(chunk_summaries)
            final_passport = call_ollama(prompt=combined_summaries, system_prompt=FINAL_SYSTEM_PROMPT)
            
            if not final_passport:
                print("Ошибка генерации финального паспорта.")
                continue
            
            # Подготовка структуры результата
            result_data = {
                "book_id": book_id,
                "title": row['title'],
                "author": row['author'],
                "genre": row['genre'],
                "semantic_passport": final_passport
            }
            
            # Сохранение индивидуального чекпоинта (чтобы не обрабатывать повторно)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)
                
            # Дозапись в ЕДИНЫЙ обновляемый файл passports_consolidated.jsonl
            with open(CONSOLIDATED_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                
            print(f"Готово! Паспорт книги успешно записан в общий файл.")
            
        except Exception as e:
            print(f"Критическая ошибка при обработке книги {book_id}: {e}")

# --- ЗАПУСК ---
if __name__ == "__main__":
    # 1. Читаем .inpx файл вашей функцией load_inpx
    raw_data = load_inpx("./data/fb2.flibusta.lib.rus.ec.7z.inpx")
    
    # 2. Применяем ваш начальный фильтр
    filtered_data = raw_data[raw_data["sourcelib"].str.contains("Flibusta") & raw_data["ext"].str.contains("fb2")]
    
    # 3. Запускаем умную очистку, дедупликацию и фильтрацию ТОЛЬКО фантастики
    clean_data = clean_and_deduplicate(filtered_data)
    
    # 5. Запуск генерации паспортов (без лимитов, обработает сколько успеет)
    run_dataset_generation(clean_data)