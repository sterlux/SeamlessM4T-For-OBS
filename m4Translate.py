import torch
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, SeamlessM4TForSpeechToText
from obswebsocket import obsws, requests
import time
import threading
import queue

# Check if GPU is available and set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"  # Force CPU usage
print(f"Using device: {device}")

# Load the SeamlessM4T model
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TForSpeechToText.from_pretrained("facebook/hf-seamless-m4t-medium").to(device)

# OBS WebSocket parameters
host = "localhost"
port = 4455  # Ensure this matches the port in OBS WebSocket settings
password = "zMc4FoVuate9vbE1"  # Ensure this matches the password in OBS WebSocket settings

# Connect to OBS
ws = obsws(host, port, password)
try:
    ws.connect()
    print("Connected to OBS WebSocket")
except Exception as e:
    print(f"Failed to connect to OBS WebSocket: {e}")
    exit()

# Queue to manage audio processing
audio_queue = queue.Queue()

def is_silent(audio_data, threshold=0.01):
    return np.max(np.abs(audio_data)) < threshold

def normalize_audio(audio_data):
    audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize to [-1, 1]
    return audio_data

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio callback status: {status}")

    # Check if the audio data is silent
    if is_silent(indata):
        return

    # Normalize and enqueue audio data for processing
    audio_queue.put(normalize_audio(indata.copy()))

def process_audio_queue():
    last_translation_time = 0
    debounce_time = 2.5  # Increase debounce time to 2.5 seconds

    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            current_time = time.time()

            if current_time - last_translation_time > debounce_time:
                translated_text = translate_audio(audio_data)
                if translated_text:
                    update_subtitle(translated_text)
                last_translation_time = current_time

def translate_audio(audio_data):
    try:
        # Convert audio data to the format required by the model
        audio_data = np.squeeze(audio_data)
        audio_data = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(device)

        # Prepare the audio input for the model
        start_time = time.time()
        audio_inputs = processor(audios=audio_data.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        audio_inputs = {key: value.to(device) for key, value in audio_inputs.items()}

        # Run inference directly to get the translated text
        generated_ids = model.generate(input_features=audio_inputs['input_features'], tgt_lang="eng")
        end_time = time.time()

        translated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"Translated text: {translated_text[0]} (inference time: {end_time - start_time:.2f}s)")
        return translated_text[0]
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

def update_subtitle(text):
    ws.call(requests.SetInputSettings(
    inputName="Subtitles",
    inputSettings={
        "text": text,
    })
)  # Update the source name if needed

# Set up audio stream
index_of_your_microphone = 1  # Update this index based on your microphone
SAMPLE_RATE = 16000
CHUNK_SIZE = 32000  # Increase the

try:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, channels=1, callback=audio_callback, device=index_of_your_microphone)
    stream.start()
    print("Audio stream started")
except Exception as e:
    print(f"Failed to start audio stream: {e}")
    exit()

# Start the audio processing thread
threading.Thread(target=process_audio_queue, daemon=True).start()

# Keep the script running
try:
    print("Script is running...")
    while True:
        time.sleep(0.25)
except KeyboardInterrupt:
    stream.stop()
    ws.disconnect()
    print("Script stopped")
