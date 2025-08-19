import os
from pathlib import Path
import soundfile as sf
import librosa
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, send_from_directory
from googletrans import Translator
from pydub import AudioSegment

# Voice cloning imports
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

app = Flask(__name__, static_folder="static")

# -------------------
# Load Models
# -------------------
MODELS_DIR = Path(os.getcwd()) / "saved_models" / "default"
encoder.load_model(MODELS_DIR / "encoder.pt")
synthesizer = Synthesizer(MODELS_DIR / "synthesizer.pt")
vocoder.load_model(MODELS_DIR / "vocoder.pt")

translator = Translator()
reference_embedding = None


# -------------------
# Routes
# -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate")
def translate_page():
    return render_template("translate.html")

@app.route("/clone", methods=["POST"])
def clone():
    global reference_embedding

    if "ref_voice" not in request.files or "text" not in request.form:
        return jsonify({"error": "Missing input"}), 400

    ref_voice_file = request.files["ref_voice"]
    text = request.form["text"]

    # Save raw upload
    raw_path = "temp_ref_upload"
    ref_voice_file.save(raw_path)

    # Convert to wav
    temp_path = "temp_ref.wav"
    sound = AudioSegment.from_file(raw_path)
    sound.export(temp_path, format="wav")

    # Encode speaker
    wav, _ = librosa.load(temp_path, sr=None)
    wav_preprocessed = encoder.preprocess_wav(wav)
    reference_embedding = encoder.embed_utterance(wav_preprocessed)

    # Generate voice
    specs = synthesizer.synthesize_spectrograms([text], [reference_embedding])
    generated_wav = vocoder.infer_waveform(specs[0])

    out_path = "static/cloned_output.wav"
    sf.write(out_path, generated_wav, synthesizer.sample_rate)

    return jsonify({"audio_file": "cloned_output.wav"})


@app.route("/voice_translate", methods=["POST"])
def voice_translate():
    global reference_embedding

    if "voice_input" not in request.files or "language" not in request.form:
        return jsonify({"error": "Missing input"}), 400

    target_lang = request.form["language"]
    voice_file = request.files["voice_input"]

    # Save raw upload
    raw_path = "temp_input_upload"
    voice_file.save(raw_path)

    # Convert to wav
    temp_path = "temp_input.wav"
    sound = AudioSegment.from_file(raw_path)
    sound.export(temp_path, format="wav")

    # Recognize Kannada
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_path) as source:
        audio_data = recognizer.record(source)
        input_text = recognizer.recognize_google(audio_data, language="kn-IN")

    # Translate
    translated_text = translator.translate(input_text, dest=target_lang).text

    # Clone translated voice (if reference exists)
    audio_file = None
    if reference_embedding is not None:
        specs = synthesizer.synthesize_spectrograms([translated_text], [reference_embedding])
        generated_wav = vocoder.infer_waveform(specs[0])
        audio_file = "translated_output.wav"
        sf.write(os.path.join("static", audio_file), generated_wav, synthesizer.sample_rate)

    return jsonify({
        "recognized_text": input_text,
        "translated_text": translated_text,
        "audio_file": audio_file
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(debug=True)
