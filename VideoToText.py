import ffmpeg
import whisper
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import warnings
from tqdm import tqdm
import time
import numpy as np
import subprocess

# Вимикаємо конкретні попередження
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_audio_from_video(video_file, output_audio_file):
    """
    Витягує аудіо з відеофайлу та зберігає його в форматі .wav, приглушуючи вивід FFmpeg.
    """
    try:
        # Перевіряємо, чи існує файл, і видаляємо його
        if os.path.exists(output_audio_file):
            os.remove(output_audio_file)
            print(f"Видалено існуючий файл: {output_audio_file}")
        
        # Використовуємо FFmpeg для витягання аудіо, приглушуємо його вивід
        ffmpeg.input(video_file).output(output_audio_file).run(overwrite_output=True, quiet=True)
        print(f"Аудіо успішно витягнуто: {output_audio_file}")
    except ffmpeg.Error as e:
        print(f"Сталася помилка під час витягнення аудіо: {e}")

def transcribe_audio_in_chunks(audio_file, chunk_duration=30):
    """
    Транскрибує аудіофайл за допомогою моделі Whisper з показом прогрес-бару через tqdm, розбиваючи файл на частини.
    """
    try:
        # Завантажуємо більшу модель Whisper для покращення точності
        model = whisper.load_model("medium")  # Використовуємо "medium" для покращення точності
        
        # Завантажуємо аудіо
        audio = whisper.load_audio(audio_file)
        audio_duration = audio.shape[-1] / whisper.audio.SAMPLE_RATE
        
        # Розбиваємо аудіо на частини по chunk_duration секунд
        total_chunks = int(np.ceil(audio_duration / chunk_duration))
        
        print("Початок транскрипції...")
        transcribed_text = ""
        
        # Використовуємо tqdm для відображення прогрес-бару
        with tqdm(total=total_chunks, unit="частин", desc="Транскрибування", ncols=100) as pbar:
            for i in range(total_chunks):
                start = int(i * chunk_duration * whisper.audio.SAMPLE_RATE)
                end = int((i + 1) * chunk_duration * whisper.audio.SAMPLE_RATE)
                audio_chunk = audio[start:end]
                
                # Транскрибуємо кожен сегмент
                result = model.transcribe(audio_chunk, language='ru')
                transcribed_text += result['text'] + " "
                
                # Оновлюємо прогрес-бар після кожної частини
                pbar.update(1)
        
        # Повертаємо транскрибований текст
        return transcribed_text
    except Exception as e:
        print(f"Сталася помилка під час транскрибування аудіо: {e}")
        return ""

def video_to_text(video_file, output_audio_file="output_audio.wav"):
    """
    Основна функція для конвертації відео в текст:
    1. Витягує аудіо з відео.
    2. Транскрибує аудіо в текст з показом прогрес-бару.
    """
    # 1. Витягуємо аудіо з відео
    extract_audio_from_video(video_file, output_audio_file)
    
    # 2. Транскрибуємо аудіо з прогрес-баром, розбиваючи на частини
    transcribed_text = transcribe_audio_in_chunks(output_audio_file)
    
    # Повертаємо результат
    return transcribed_text

def choose_files():
    """
    Функція для вибору файлів через GUI.
    """
    # Створюємо вікно Tkinter, яке ми не будемо відображати
    root = Tk()
    root.withdraw()  # Ховаємо головне вікно

    # Обираємо відеофайл
    video_file = askopenfilename(title="Виберіть відеофайл", filetypes=[("Video Files", "*.mp4 *.mkv *.avi")])
    
    if not video_file:
        print("Відеофайл не обрано.")
        return None, None

    # Генеруємо стандартну назву текстового файлу на основі відеофайлу
    default_text_file = os.path.splitext(os.path.basename(video_file))[0] + ".txt"

    # Обираємо місце для збереження результату
    output_text_file = asksaveasfilename(title="Збережіть транскрибований текст", initialfile=default_text_file,
                                         defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    
    if not output_text_file:
        print("Місце для збереження не обрано.")
        return video_file, None

    return video_file, output_text_file

# Використання програми
if __name__ == "__main__":
    # Викликаємо функцію для вибору файлів
    video_file, output_text_file = choose_files()
    
    if video_file and output_text_file:
        # Тимчасовий аудіофайл для обробки
        output_audio_file = "output_audio.wav"
        
        # Конвертуємо відео у текст
        text = video_to_text(video_file, output_audio_file)

        # Зберігаємо транскрибований текст у файл
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Транскрибований текст збережено у: {output_text_file}")
    else:
        print("Операцію скасовано.")
