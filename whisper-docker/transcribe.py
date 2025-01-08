import whisper
import torch
import os
import tkinter as tk
from tkinter import filedialog, ttk
from threading import Thread

class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Транскрибация")
        self.root.geometry("600x400")
        
        # Создаем и размещаем элементы интерфейса
        self.create_widgets()
        
    def create_widgets(self):
        # Фрейм для выбора файла
        file_frame = ttk.LabelFrame(self.root, text="Выбор файла", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Выбрать файл", command=self.select_file).pack(side="left", padx=5)
        
        # Фрейм для выбора модели
        model_frame = ttk.LabelFrame(self.root, text="Настройки", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        self.model_size = tk.StringVar(value="medium")
        ttk.Label(model_frame, text="Размер модели:").pack(side="left", padx=5)
        ttk.Combobox(model_frame, textvariable=self.model_size, 
                     values=["tiny", "base", "small", "medium", "large"],
                     state="readonly").pack(side="left", padx=5)
        
        # Кнопка запуска
        ttk.Button(self.root, text="Начать транскрибацию", 
                  command=self.start_transcription).pack(pady=10)
        
        # Прогресс
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill="x", padx=10, pady=5)
        
        # Текстовое поле для вывода
        self.output_text = tk.Text(self.root, height=10, wrap=tk.WORD)
        self.output_text.pack(fill="both", expand=True, padx=10, pady=5)
        
    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Аудио файлы", "*.mp3 *.wav *.m4a")]
        )
        if file_path:
            self.file_path.set(file_path)
            
    def start_transcription(self):
        if not self.file_path.get():
            self.output_text.insert("end", "Выберите файл для транскрибации\n")
            return
            
        self.progress.start()
        Thread(target=self.transcribe_thread).start()
        
    def transcribe_thread(self):
        try:
            # Загружаем модель
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.update_output(f"Используется устройство: {device}\n")
            
            model = whisper.load_model(self.model_size.get(), device=device)
            self.update_output("Модель загружена, начинаем распознавание...\n")
            
            # Транскрибация
            result = model.transcribe(
                self.file_path.get(),
                language="ru",
                temperature=0.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True
            )
            
            # Сохраняем результат
            output_file = os.path.splitext(self.file_path.get())[0] + ".txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
                
            self.update_output(f"\nРезультат:\n{result['text']}\n")
            self.update_output(f"\nТекст сохранен в файл: {output_file}\n")
            
        except Exception as e:
            self.update_output(f"Ошибка: {str(e)}\n")
        finally:
            self.progress.stop()
            
    def update_output(self, text):
        self.output_text.insert("end", text)
        self.output_text.see("end")

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop() 