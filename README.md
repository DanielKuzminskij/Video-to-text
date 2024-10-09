# Video to Text Transcription using Whisper and FFmpeg

This Python script extracts audio from a video file, transcribes the audio into text using the Whisper speech recognition model, and saves the transcription as a `.txt` file. The script is designed to work with videos in formats such as `.mp4`, `.mkv`, and `.avi`, and uses FFmpeg for audio extraction.

## Features

- **Extract audio from video**: Automatically extracts the audio from any supported video format.
- **Transcription using Whisper**: Utilizes OpenAI's Whisper model to convert audio into text.
- **Progress bar**: Displays a progress bar during the transcription process.
- **Segmented transcription**: Audio is transcribed in chunks to provide better progress tracking.
- **GUI file selection**: Easy file selection for both input video and output transcription files through graphical dialog windows.

## Requirements

To use this script, you'll need the following libraries:

```bash
pip install whisper ffmpeg-python tqdm numpy
```

You'll also need to have FFmpeg installed on your system. You can download it [here](https://ffmpeg.org/download.html).

## Usage

Run the script:

You can run the script using Python. The script will prompt you to select a video file and an output location for the transcription.

```bash
python video_to_text.py
```

- **Choose a video file**: A GUI window will appear to select the video file you want to transcribe.

- **Select the output file**: You'll be prompted to choose the location and name for the output `.txt` file where the transcription will be saved.

- **Wait for the process to complete**: The script will extract the audio from the video and transcribe it in segments, showing a progress bar for each step.

- **Transcription saved**: Once the process is complete, the transcription will be saved in the `.txt` file you selected.

## Example Output

```bash
Видалено існуючий файл: output_audio.wav
Аудіо успішно витягнуто: output_audio.wav
Початок транскрипції...
Транскрибування: 100%|██████████████████████████████████████████| 22/22 [03:12<00:00,  8.74s/частин]
Транскрибований текст збережено у: /path/to/your/output.txt
```

## How it Works

- **Audio extraction**: The script uses FFmpeg to extract the audio stream from the input video.
- **Transcription**: The audio is transcribed using OpenAI's Whisper model, which supports multiple languages. The audio is transcribed in segments to show progress, and each segment is combined into a single transcription.
- **GUI**: The script uses Tkinter for file selection dialogs, allowing users to easily choose input video files and specify where to save the transcription.

## Customization

You can modify the following parameters to fit your needs:

- **Model size**: The script uses the medium Whisper model by default. You can change this to other models such as small or large depending on your accuracy and performance needs.
- **Chunk duration**: The audio is transcribed in 30-second chunks by default. You can adjust the `chunk_duration` parameter to process longer or shorter segments.
- **File types**: The default supported video file types are `.mp4`, `.mkv`, and `.avi`. You can modify the `filetypes` argument in the `askopenfilename()` function to add support for additional formats.

## Limitations

- **Longer processing time**: Using larger Whisper models like medium or large increases transcription accuracy but also significantly increases processing time.
- **Performance**: Transcription can be CPU-intensive, especially for long videos.

## Contributing

Feel free to open an issue or submit a pull request if you would like to contribute to this project.
