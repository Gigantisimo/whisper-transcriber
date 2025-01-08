import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import whisper
except ImportError:
    print("Error: Whisper not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "git+https://github.com/openai/whisper.git"])
    import whisper

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import gc
from pathlib import Path
import uvicorn

app = FastAPI()

# Настройка CORS и таймаутов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Создаем папки для файлов
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

class WhisperTranscriber:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        
    def load_model(self, model_size="tiny"):
        if self.model is None:
            try:
                # Очищаем память перед загрузкой
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Загружаем модель
                self.model = whisper.load_model(
                    model_size,
                    device=self.device,
                    download_root="./models",
                    in_memory=True
                )
                
                self.model.eval()
                
                # Отключаем градиенты
                with torch.no_grad():
                    for param in self.model.parameters():
                        param.requires_grad = False
                    
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        return self.model
    
    async def transcribe_file(self, file_path: str, model_size: str = "tiny"):
        try:
            model = self.load_model(model_size)
            
            # Транскрибация с оптимизированными параметрами
            with torch.inference_mode():
                result = model.transcribe(
                    file_path,
                    language="ru",
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    batch_size=4,  # Уменьшаем batch_size
                    fp16=False
                )
            
            # Очищаем память после транскрибации
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            raise

transcriber = WhisperTranscriber()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def read_root():
    return RedirectResponse(url="/static/index.html")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), model_size: str = "tiny"):
    try:
        if not file.filename:
            return JSONResponse({
                "status": "error",
                "message": "No file provided"
            }, status_code=400)
            
        # Проверяем размер файла (10MB максимум)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            return JSONResponse({
                "status": "error",
                "message": "File too large. Maximum size is 10MB"
            }, status_code=400)
            
        # Сохраняем файл
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        try:
            # Транскрибация
            text = await transcriber.transcribe_file(str(file_path), model_size)
            return JSONResponse({"status": "success", "text": text})
            
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": str(e)
            }, status_code=500)
            
        finally:
            # Удаляем временный файл
            if file_path.exists():
                os.remove(file_path)
                
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"File processing error: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 