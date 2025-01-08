import whisper
import torch
import os
import tkinter as tk
from tkinter import filedialog, ttk
from threading import Thread
import gc  # для сборки мусора

class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Транскрибация")
        self.root.geometry("800x600")
        
        # Добавляем хранение модели
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Устанавливаем стиль
        style = ttk.Style()
        style.theme_use('clam')
        
        # Создаем и размещаем элементы интерфейса
        self.create_widgets()
        
    def create_widgets(self):
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Фрейм для выбора файла
        file_frame = ttk.LabelFrame(main_frame, text="Выбор файла", padding=10)
        file_frame.pack(fill="x", pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=70).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Выбрать файл", command=self.select_file).pack(side="left", padx=5)
        
        # Фрейм для настроек
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки", padding=10)
        settings_frame.pack(fill="x", pady=5)
        
        # Выбор модели
        self.model_size = tk.StringVar(value="medium")
        ttk.Label(settings_frame, text="Размер модели:").pack(side="left", padx=5)
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_size, 
                                 values=["tiny", "base", "small", "medium", "large"],
                                 state="readonly", width=10)
        model_combo.pack(side="left", padx=5)
        
        # Кнопка запуска
        ttk.Button(settings_frame, text="Начать транскрибацию", 
                  command=self.start_transcription).pack(side="right", padx=5)
        
        # Добавляем кнопку выгрузки модели
        ttk.Button(settings_frame, text="Выгрузить модель", 
                  command=self.unload_model).pack(side="right", padx=5)
        
        # Добавляем чекбокс для оптимизации памяти
        self.optimize_memory = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Оптимизировать память", 
                       variable=self.optimize_memory).pack(side="right", padx=5)
        
        # Статус и прогресс
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill="x", pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Готов к работе")
        self.status_label.pack(side="left", pady=5)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(fill="x", pady=5)
        
        # Текстовое поле для вывода
        output_frame = ttk.LabelFrame(main_frame, text="Результат", padding=10)
        output_frame.pack(fill="both", expand=True, pady=5)
        
        # Добавляем скроллбар
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.output_text = tk.Text(output_frame, height=10, wrap=tk.WORD, 
                                 yscrollcommand=scrollbar.set)
        self.output_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.output_text.yview)
        
    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Аудио файлы", "*.mp3 *.wav *.m4a *.ogg *.flac"),
                ("Все файлы", "*.*")
            ]
        )
        if file_path:
            self.file_path.set(file_path)
            
    def start_transcription(self):
        if not self.file_path.get():
            self.update_output("Пожалуйста, выберите файл для транскрибации\n")
            return
            
        self.progress.start()
        self.status_label.config(text="Выполняется транскрибация...")
        Thread(target=self.transcribe_thread).start()
        
    def unload_model(self):
        if self.model:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            self.update_output("Модель выгружена из памяти\n")
            
    def load_model(self, model_size):
        if self.model is None or self.model.model_size != model_size:
            self.unload_model()
            
            # Устанавливаем оптимизации для CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                
            # Загружаем модель с оптимизациями
            self.model = whisper.load_model(
                model_size,
                device=self.device,
                download_root="./models"  # Кэшируем модели локально
            )
            
            # Оптимизируем модель
            if self.optimize_memory.get():
                self.model = self.model.half()  # Используем половинную точность
                
        return self.model
            
    def transcribe_thread(self):
        try:
            self.update_output(f"Используется устройство: {self.device}\n")
            
            # Загружаем модель
            self.update_output(f"Загрузка модели {self.model_size.get()}...\n")
            model = self.load_model(self.model_size.get())
            
            self.update_output("Начинаем распознавание...\n")
            
            # Оптимизируем использование памяти
            if self.optimize_memory.get():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Транскрибация с оптимизациями
            with torch.inference_mode():  # Экономим память при инференсе
                result = model.transcribe(
                    self.file_path.get(),
                    language="ru",
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    batch_size=16  # Можно регулировать для баланса скорости/памяти
                )
            
            # Сохраняем результат
            output_file = os.path.splitext(self.file_path.get())[0] + "_transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
                
            self.update_output(f"\nРезультат:\n{result['text']}\n")
            self.update_output(f"\nТекст сохранен в файл: {output_file}\n")
            
            # Очищаем память после работы
            if self.optimize_memory.get():
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            self.update_output(f"Ошибка: {str(e)}\n")
        finally:
            self.progress.stop()
            self.status_label.config(text="Готов к работе")
            
    def update_output(self, text):
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop() 