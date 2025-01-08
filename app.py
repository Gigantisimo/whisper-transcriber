import warnings
import logging
import sys
from typing import Optional

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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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
    
    async def transcribe_file(self, file_path: str, model_size: str = "tiny") -> Optional[str]:
        try:
            logger.info(f"Loading model {model_size}")
            model = self.load_model(model_size)
            
            logger.info(f"Starting transcription of {file_path}")
            with torch.inference_mode():
                result = model.transcribe(
                    file_path,
                    language="ru",
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    batch_size=4
                )
            
            # Очищаем память
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Transcription completed successfully")
            return result["text"]
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
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
        logger.info(f"Received file: {file.filename}, size: {file.size}, model: {model_size}")
        
        if not file.filename:
            return JSONResponse({
                "status": "error",
                "message": "No file provided"
            }, status_code=400)
            
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        
        if file_size > 10 * 1024 * 1024:
            return JSONResponse({
                "status": "error",
                "message": "File too large. Maximum size is 10MB"
            }, status_code=400)
            
        file_path = UPLOAD_DIR / file.filename
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            
            logger.info("Starting transcription process")
            text = await transcriber.transcribe_file(str(file_path), model_size)
            
            return JSONResponse({
                "status": "success",
                "text": text
            })
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}", exc_info=True)
            return JSONResponse({
                "status": "error",
                "message": str(e)
            }, status_code=500)
            
        finally:
            if file_path.exists():
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
                
    except Exception as e:
        logger.error(f"File processing error: {str(e)}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": f"File processing error: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting server on port {port}")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            timeout_keep_alive=30,
            limit_concurrency=10
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise 