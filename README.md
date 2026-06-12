# Backend API

ECHONEP is a high-performance FastAPI service designed for English (EN) to Nepali (NP) and Nepali to English translation. It integrates Automatic Speech Recognition (ASR), Neural Machine Translation (NMT), and Text-to-Speech (TTS) into a unified pipeline. 

NOTE:
    The translation model used in this project is fine tuned under transaction based environement. The model's BLEU score is good but needs human evaluation. The translation model only filters Nepali and English language other languages are ignored during finetuning of model to avoid other-language intervension in performance.


## Key Features
- **Speech-to-Speech Pipeline**: Process base64 encoded audio to get translated text and synthesized speech in one request.
- **Hybrid Translation**: Uses a phrasebook for instant exact matches and a fine-tuned NLLB model for generative translation.
- **Multi-Backend TTS**: Supports high-quality Nepali synthesis via Piper and versatile English synthesis via pyttsx3 or gTTS.
- **Performance Optimized**: Uses `faster-whisper` for low-latency ASR and model caching to ensure fast response times.

## Models Used

### 1. Translation (NMT)
- **Primary Model**: SarjakBhandari-230383/EchoNep (filtered only nepali and english translation)
- **Base Architecture**: Facebook's NLLB-200 (No Language Left Behind).
- **Fallback**: A local `phrasebook.json` for common travel and trade expressions.

### 2. Automatic Speech Recognition (ASR)
- **Engine**: `faster-whisper`
- **Model Size**: `large-v3`
- **Optimizations**: Quantized to `int8` for CPU efficiency.

### 3. Text-to-Speech (TTS)
- **Nepali**: Piper TTS using the `ne_NP-chitwan-medium` voice.
- **English/Fallback**: `pyttsx3` (Offline) or `gTTS` (Online).

## How the Backend Works

1.  **Lifespan Management**: On startup, the API pre-loads the ASR and NMT models into memory to avoid latency during the first request.
2.  **The Pipeline Endpoint (`/pipeline`)**:
    *   **ASR Phase**: If audio is provided, it decodes the base64 string, saves it to a temporary buffer, and transcribes it using Whisper.
    *   **Translation Phase**:
        *   Checks the `phrasebook.json` for an exact match.
        *   If not found, it passes the text to the EchoNep NMT model.
        *   If translating to Nepali, it also generates a Romanized version of the script.
    *   **TTS Phase**: The translated text is converted back to audio. Nepali text uses the high-quality Piper ONNX model, while English uses system-level voices or Google TTS.
3.  **Schema Validation**: Uses Pydantic to ensure all incoming requests (audio or text) meet the required format and language directions (`en_np` or `np_en`).

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- FFmpeg (required for audio processing)
- A Hugging Face token (if accessing private/restricted models)

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd BACKEND
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: Ensure you have `torch`, `transformers`, `faster-whisper`, `fastapi`, `uvicorn`, and `piper-tts` installed.*

### 4. Environment Variables
Create a `.env` file or set your environment variables:
```bash
HF_TOKEN=your_huggingface_token_here
```

## Running the API

Start the server using Uvicorn:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

## Testing
You can use the provided `test_translate.py` script to verify the installation:
```bash
python test_translate.py
```

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | GET | Returns service status and loaded backends. |
| `/translate` | POST | Translates text and returns audio. |
| `/asr` | POST | Transcribes base64 audio to text. |
| `/pipeline` | POST | Full Audio-to-Audio translation flow. |

## Project Structure
```text
app/
├── data/           # Phrasebook and static data
├── routers/        # API route definitions
├── services/       # Logic for ASR, NMT, TTS, and Romanization
├── config.py       # Global settings and model paths
├── main.py         # FastAPI app initialization
└── schemas.py      # Pydantic models for requests/responses
```