from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os

from stt import transcribe_speech_to_text
from llm import generate_response
from tts import transcribe_text_to_speech

# Tambahan: import modul G2P
from g2p_id.g2p import G2P

app = FastAPI()
g2p = G2P()

@app.post("/voice-chat")
async def ask_audio(file: UploadFile = File(...)):
    try:
        print("[INFO] Menerima file audio...")
        audio_bytes = await file.read()
        file_ext = os.path.splitext(file.filename)[-1]

        print("[DEBUG] File extension:", file_ext)

        print("[INFO] Menjalankan proses STT...")
        text = transcribe_speech_to_text(audio_bytes, file_ext)
        print("[DEBUG] Hasil STT:", text)

        if text.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=f"STT Error: {text}")

        print("[INFO] Mengirim pertanyaan ke LLM...")
        response_text = generate_response(text)
        print("[DEBUG] Respon LLM:", response_text)

        if response_text.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=f"LLM Error: {response_text}")

        print("[INFO] Menjalankan proses G2P...")
        phonemes = g2p(response_text)
        phoneme_text = ' '.join(phonemes)
        print("[DEBUG] Hasil G2P:", phoneme_text)

        print("[INFO] Mengonversi fonem ke suara (TTS)...")
        tts_path = transcribe_text_to_speech(response_text)
        print("[DEBUG] File audio TTS disimpan di:", tts_path)

        if not os.path.exists(tts_path):
            raise HTTPException(status_code=500, detail="TTS output not found")

        return FileResponse(tts_path, media_type="audio/wav", filename="response.wav")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "API siap menerima file audio di endpoint /voice-chat"}
