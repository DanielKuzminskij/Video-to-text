import ffmpeg
import whisper
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import warnings
from tqdm import tqdm
import numpy as np

# Вимикаємо конкретні попередження
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_audio_from_video(video_file, output_audio_file):
    """
    Витягує аудіо з відеофайлу та зберігає його в форматі .wav, приглушуючи вивід FFmpeg.
    """
    try:
        if os.path.exists(output_audio_file):
            os.remove(output_audio_file)
            print(f"Видалено існуючий файл: {output_audio_file}")
        
        # Використовуємо FFmpeg для витягання аудіо, приглушуємо вивід
        ffmpeg.input(video_file).output(output_audio_file).run(overwrite_output=True, quiet=True)
        print(f"Аудіо успішно витягнуто: {output_audio_file}")
    except ffmpeg.Error as e:
        print(f"Сталася помилка під час витягнення аудіо: {e}")

def transcribe_audio_in_chunks(audio_file, chunk_duration=30):
    """
    Транскрибує аудіофайл за допомогою моделі Whisper з показом прогрес-бару через tqdm, розбиваючи файл на частини.
    """
    try:
        model = whisper.load_model("medium")  # Використовуємо "medium" для кращої точності
        audio = whisper.load_audio(audio_file)
        audio_duration = audio.shape[-1] / whisper.audio.SAMPLE_RATE
        
        total_chunks = int(np.ceil(audio_duration / chunk_duration))
        
        print("Початок транскрипції...")
        transcribed_text = ""
        
        with tqdm(total=total_chunks, unit="частин", desc="Транскрибування", ncols=100) as pbar:
            for i in range(total_chunks):
                start = int(i * chunk_duration * whisper.audio.SAMPLE_RATE)
                end = int((i + 1) * chunk_duration * whisper.audio.SAMPLE_RATE)
                audio_chunk = audio[start:end]
                result = model.transcribe(audio_chunk, language='ru')
                transcribed_text += result['text'] + " "
                pbar.update(1)
        
        return transcribed_text
    except Exception as e:
        print(f"Сталася помилка під час транскрибування аудіо: {e}")
        return ""

def video_to_text(video_file):
    """
    Основна функція для конвертації відео в текст.
    1. Витягує аудіо з відео.
    2. Транскрибує аудіо в текст з показом прогрес-бару.
    """
    output_audio_file = os.path.splitext(video_file)[0] + ".wav"
    output_text_file = os.path.splitext(video_file)[0] + ".txt"
    
    extract_audio_from_video(video_file, output_audio_file)
    transcribed_text = transcribe_audio_in_chunks(output_audio_file)
    
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(transcribed_text)
    
    print(f"Транскрибований текст збережено у: {output_text_file}")

def choose_files():
    """
    Функція для вибору декількох файлів через GUI.
    """
    root = Tk()
    root.withdraw()  # Ховаємо головне вікно

    video_files = askopenfilenames(title="Виберіть відеофайли", filetypes=[("Video Files", "*.mp4 *.mkv *.avi")])
    
    if not video_files:
        print("Файли не обрано.")
        return []

    return video_files

# Використання програми
if __name__ == "__main__":
    video_files = choose_files()
    
    if video_files:
        for video_file in video_files:
            video_to_text(video_file)
    else:
        print("Операцію скасовано.")
