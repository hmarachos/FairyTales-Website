import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

def encode_image(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    mime, _ = mimetypes.guess_type(image_path.name)
    if not mime or not mime.startswith("image/"):
        raise ValueError("Поддерживаются только файлы изображений (jpg/png/webp и т.п.).")
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def build_messages(age_range: str, image_data: str, extra_prompt: str | None):
    user_text = (
        f"Создай тёплую сказку для ребёнка {age_range} лет. "
        "Сохрани добрый тон, избегай страшных элементов, используй простые фразы и короткие предложения. "
        "120-160 слов. Добавь заголовок в первой строке."
    )
    if extra_prompt:
        user_text += f" Дополнительные пожелания: {extra_prompt}"
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Ты пишешь добрые сказки перед сном для детей 3-5 лет. "
                        "Опирайся на изображение пользователя, но не описывай его дословно."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data, "detail": "low"}},
            ],
        },
    ]


def generate_story(
    image_path: Path,
    age_range: str = "3-5",
    model: str = "gpt-4o-mini",
    temperature: float = 0.8,
    max_tokens: int = 350,
    extra_prompt: str | None = None,
) -> str:
    # Загружаем .env рядом со скриптом и в текущей директории
    env_path = Path(__file__).with_name(".env")
    loaded_script_dir = load_dotenv(env_path, override=True)
    loaded_cwd = load_dotenv(override=True)  # пытаемся также из CWD
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        details = (
            f"OPENAI_API_KEY не найден. "
            f".env рядом со скриптом: {env_path} (exists={env_path.exists()}, loaded={loaded_script_dir}); "
            f".env из текущей директории loaded={loaded_cwd}. "
            "Проверьте формат строки: OPENAI_API_KEY=sk-... без кавычек."
        )
        raise EnvironmentError(details)
    client = OpenAI()
    image_data = encode_image(image_path)
    messages = build_messages(age_range, image_data, extra_prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генератор сказок по изображению (3-5 лет).")
    parser.add_argument("image_path", help="Путь до изображения (jpg/png/webp и т.п.)")
    parser.add_argument("--age-range", default="3-5", help="Диапазон возраста ребёнка, напр. '3-5'")
    parser.add_argument("--model", default="gpt-4o-mini", help="Модель OpenAI, поддерживающая изображение")
    parser.add_argument("--temperature", type=float, default=0.8, help="Креативность ответа")
    parser.add_argument("--max-tokens", type=int, default=350, help="Максимум токенов в ответе")
    parser.add_argument("--note", default=None, help="Дополнительные пожелания к сюжету")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        story = generate_story(
            Path(args.image_path),
            age_range=args.age_range,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_prompt=args.note,
        )
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + story)


if __name__ == "__main__":
    main()

