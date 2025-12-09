import os
import base64
import requests
import tempfile
import mimetypes
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import json

# Загружаем переменные окружения
load_dotenv()

app = Flask(__name__,
            static_url_path='/static',
            static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Включаем CORS для всех маршрутов
CORS(app)

# Проверяем API ключ при запуске
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ ОШИБКА: OPENAI_API_KEY не найден в .env файле!")
    print("Создайте .env файл и добавьте: OPENAI_API_KEY=sk-proj-ваш_ключ")
    exit(1)

if not api_key.startswith('sk-'):
    print("❌ ОШИБКА: OPENAI_API_KEY должен начинаться с 'sk-'")
    exit(1)

# Инициализируем OpenAI клиент
client = OpenAI(api_key=api_key)
print("✅ OpenAI API ключ найден и инициализирован")

# Проверяем работоспособность API ключа
try:
    test_response = client.models.list()
    print("✅ OpenAI API подключение успешно")
except Exception as e:
    print(f"❌ ОШИБКА: Не удалось подключиться к OpenAI API: {e}")
    print("Проверьте API ключ и интернет-соединение")
    exit(1)

def save_image_from_url(image_url: str, filename: str) -> str:
    """Скачивает изображение по URL и сохраняет локально"""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Создаём путь для сохранения
        save_dir = Path('static/generated')
        save_dir.mkdir(exist_ok=True)
        
        # Генерируем уникальное имя файла
        file_path = save_dir / f"{filename}.png"
        
        # Сохраняем изображение
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        # Возвращаем относительный путь для веба
        return f'/static/generated/{filename}.png'
    except Exception as e:
        print(f"⚠️ Ошибка сохранения изображения: {e}")
        return image_url  # Возвращаем исходный URL если не удалось сохранить

def encode_image(image_path: str) -> str:
    """Кодирует изображение в base64 с правильным MIME типом"""
    # Определяем MIME тип файла
    mime_type, _ = mimetypes.guess_type(image_path)

    print(f"🔍 Определяем MIME тип для {image_path}: {mime_type}")

    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"Неподдерживаемый тип файла. MIME: {mime_type}")

    # OpenAI поддерживает только определенные форматы
    supported_formats = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    if mime_type not in supported_formats:
        raise ValueError(f"Формат {mime_type} не поддерживается OpenAI. Поддерживаемые: {supported_formats}")

    # Читаем и кодируем файл
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    result = f"data:{mime_type};base64,{encoded_string}"
    print(f"✅ Изображение закодировано, MIME: {mime_type}, длина: {len(encoded_string)} символов")

    return result

def generate_story_with_images(image_path: str, age_range: str, extra_prompt: str = None) -> dict:
    """Генерирует сказку и иллюстрации"""
    try:
        # Кодируем изображение для OpenAI
        image_data = encode_image(image_path)

        # Промпт для генерации сказки
        story_prompt = f"""Создай тёплую сказку для ребёнка {age_range} лет на основе изображения.

ТРЕБОВАНИЯ:
- Сохрани добрый тон, избегай страшных элементов
- Используй простые фразы и короткие предложения
- Сказка должна быть разделена на 3 логические части
- Каждая часть: 2-3 предложения

ОТВЕТЬ ТОЛЬКО В ФОРМАТЕ JSON:
{{
    "title": "Заголовок сказки",
    "part1": "Текст первой части сказки",
    "part2": "Текст второй части сказки",
    "part3": "Текст третьей части сказки"
}}

НЕ ДОБАВЛЯЙ НИКАКОГО ТЕКСТА ВНЕ JSON!"""

        if extra_prompt:
            story_prompt += f"\nДополнительные пожелания: {extra_prompt}"

        # Генерируем сказку
        story_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ты пишешь добрые сказки перед сном для детей 3-5 лет. Отвечай только в формате JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": story_prompt},
                        {"type": "image_url", "image_url": {"url": image_data, "detail": "low"}}
                    ]
                }
            ],
            temperature=0.8,
            max_tokens=500
        )

        story_content = story_response.choices[0].message.content.strip()

        if not story_content:
            raise Exception("OpenAI вернул пустой ответ. Проверьте API ключ и баланс.")

        try:
            story_data = json.loads(story_content)
        except json.JSONDecodeError as e:
            raise Exception(f"OpenAI вернул некорректный JSON. Ответ: {story_content[:200]}...")

        # Генерируем иллюстрации для каждой части
        illustrations = []
        for i, part_key in enumerate(['part1', 'part2', 'part3'], 1):
            try:
                illustration_prompt = f"""
Создай яркую, добрую иллюстрацию для детей в стиле сказки для части истории:
"{story_data[part_key]}"

Стиль: яркие цвета, дружелюбные персонажи, сказочная атмосфера, подходящая для детей {age_range} лет.
"""

                illustration_response = client.images.generate(
                    model="dall-e-3",
                    prompt=illustration_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )

                # Альтернатива: если DALL-E-3 недоступен, попробуем DALL-E-2
                if not illustration_response.data:
                    print(f"⚠️ DALL-E-3 не вернул результат для иллюстрации {i}, пробуем DALL-E-2")
                    illustration_response = client.images.generate(
                        model="dall-e-2",
                        prompt=illustration_prompt,
                        size="512x512",
                        n=1,
                    )

                # Получаем URL изображения от DALL-E
                image_url = illustration_response.data[0].url
                
                # Сохраняем изображение локально
                unique_filename = f"story_{uuid.uuid4().hex[:8]}_part{i}"
                local_url = save_image_from_url(image_url, unique_filename)
                
                illustrations.append(local_url)
                print(f"✅ Иллюстрация {i} сгенерирована и сохранена")

            except Exception as img_error:
                print(f"⚠️ Ошибка генерации иллюстрации {i}: {img_error}")
                # Fallback: используем placeholder изображение
                placeholder_url = f"https://via.placeholder.com/400x400/fecfef/2d3436?text=Иллюстрация+{i}"
                illustrations.append(placeholder_url)

        return {
            'title': story_data.get('title', 'Волшебная сказка'),
            'parts': [
                {'text': story_data['part1'], 'image': illustrations[0]},
                {'text': story_data['part2'], 'image': illustrations[1]},
                {'text': story_data['part3'], 'image': illustrations[2]}
            ]
        }

    except Exception as e:
        raise Exception(f"Ошибка при генерации: {str(e)}")

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Обслуживание favicon"""
    return app.send_static_file('favicon.ico')

@app.route('/test')
def test():
    """Тестовый маршрут"""
    return jsonify({
        'status': 'OK',
        'message': 'Flask сервер работает!',
        'cors': 'enabled'
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Генерирует сказку с иллюстрациями"""
    print("🔄 Получен POST запрос к /generate")
    try:
        # Получаем данные из формы
        age_range = request.form.get('age_range', '3-5')
        extra_prompt = request.form.get('extra_prompt', '').strip()

        # Обрабатываем загруженное изображение
        if 'image' not in request.files:
            return jsonify({'error': 'Изображение не загружено'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        # Проверяем MIME тип файла
        allowed_mimes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
        file_mime = image_file.mimetype

        print(f"📁 Загружен файл: {image_file.filename}, MIME (browser): {file_mime}")

        if file_mime not in allowed_mimes:
            return jsonify({'error': f'Неподдерживаемый формат файла. Разрешены: JPEG, PNG, GIF, WebP. Получен: {file_mime}'}), 400

        # Определяем расширение на основе MIME типа
        ext_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp'
        }
        file_ext = ext_map.get(file_mime, '.jpg')

        # Сохраняем изображение во временную директорию
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{os.urandom(8).hex()}{file_ext}")
        image_file.save(temp_path)

        print(f"✅ Файл сохранен: {temp_path}, MIME: {file_mime}")

        # Дополнительная проверка MIME типа сохраненного файла
        detected_mime, _ = mimetypes.guess_type(temp_path)
        print(f"🔍 MIME тип файла на диске: {detected_mime}")

        if detected_mime != file_mime:
            print(f"⚠️ Несоответствие MIME типов: browser={file_mime}, file={detected_mime}")

        try:
            # Генерируем сказку с иллюстрациями
            result = generate_story_with_images(temp_path, age_range, extra_prompt)
            return jsonify(result)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Ошибка генерации: {error_msg}")

            # Специфичные сообщения об ошибках
            if "unsupported_country_region_territory" in error_msg:
                error_msg = "Ваша страна не поддерживается OpenAI API напрямую. Используйте VPN или другой способ обхода."
            elif "insufficient_quota" in error_msg:
                error_msg = "Недостаточно средств на балансе OpenAI. Пополните баланс."
            elif "invalid_api_key" in error_msg:
                error_msg = "Неверный API ключ. Проверьте .env файл."
            elif "rate_limit" in error_msg:
                error_msg = "Превышен лимит запросов. Подождите немного и попробуйте снова."

            return jsonify({'error': error_msg}), 500

        finally:
            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
