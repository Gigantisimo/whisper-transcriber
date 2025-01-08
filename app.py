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
import torch
import os
import gc
from pathlib import Path
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
                torch.hub.set_dir("./models")
                
                # Очищаем память перед загрузкой
                gc.collect()
                
                self.model = whisper.load_model(
                    model_size,
                    device=self.device,
                    download_root="./models",
                    in_memory=True
                )
                
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                    
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        return self.model
    
    async def transcribe_file(self, file_path: str, model_size: str = "small"):
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
                    batch_size=8,
                    fp16=False
                )
            
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            raise

transcriber = WhisperTranscriber()

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), model_size: str = "small"):
    try:
        # Сохраняем загруженный файл
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Выполняем транскрибацию
            text = await transcriber.transcribe_file(str(file_path), model_size)
            
            return JSONResponse({
                "status": "success",
                "text": text
            })
            
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

@app.get("/")
async def read_root():
    # Перенаправляем на веб-интерфейс
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 