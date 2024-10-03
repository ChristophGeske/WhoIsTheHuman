import random
import warnings
import wave
import webrtcvad
import whisper
import pyaudio
import threading
from openai import OpenAI
import torch
from fuzzywuzzy import fuzz  # Add fuzzy matching
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import speedup
import os
from TTS.api import TTS
import re

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='fuzzywuzzy')
warnings.filterwarnings("ignore", category=FutureWarning, module='TTS')
warnings.filterwarnings("ignore", category=UserWarning, module='whisper')
warnings.filterwarnings("ignore", category=FutureWarning, module='whisper')

# Suppress PyTorch warnings
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'

# Suppress TensorFlow warnings (if applicable)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize TTS models
# TODO These models sound good but they suffer from halizunations and a differnt TTS system should be used since tts is no longer developed.
try:
    thorsten_model = TTS("tts_models/de/thorsten/tacotron2-DDC")
    jenny_model = TTS("tts_models/en/jenny/jenny")
    male_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")

    # Move models to GPU if available
    tts_device = "cuda" if torch.cuda.is_available() else "cpu"
    thorsten_model.to(tts_device)
    jenny_model.to(tts_device)
    male_model.to(tts_device)
    tts_model.to(tts_device)
except Exception as e:
    print(f"Failed to initialize TTS models: {e}")
    thorsten_model = None
    jenny_model = None
    male_model = None
    tts_model = None



# Initialize the OpenAI client with local LLM running at port 1234 (replace this with your actual setup)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
modelID = "mradermacher/Reflection-Llama-3.1-8B-GGUF"  # Specify the model you are using


# Set up WebRTC Voice Activity Detection (VAD)
vad = webrtcvad.Vad()
vad.set_mode(3)  # Set VAD to the most aggressive mode, useful for noisy environments

# Threading flags for controlling recording
recording = threading.Event()  # Tracks if recording is active
user_introduced = threading.Event()  # Tracks if the user's introduction has been recorded

gm_history = [{"role": "system",
    "content": (
        "You are the Game Master for the game 'Who's the Human?'. Your responsibilities include:"
        "\n1. Guiding the game and explaining the rules clearly."
        "\n2. Conducting only two rounds: Introductions and Voting."
        "\n3. After the Voting round, conclude the game by announcing the results and declaring the winner."
        "\n4. No additional rounds should be initiated under any circumstances."
        "\nRules of the Game (explained in one short sentence): "
        "In 'Who's the Human?', players introduce themselves in character, then vote to identify the human among them. "
        "The player suspected the most as human loses the game.")}]
player_histories = {
    "Einstein": [{"role": "system",
                  "content": "Du spielst Einstein. Bleibe und spiele immer in deinem Charakter. Antworte immer in Deutsch auch wenn du Englisch angesprochen wirst."}],
    "Genghis Khan": [{"role": "system",
                      "content": "You are playing the role of Genghis Khan. Always stay in character and respond as Genghis Khan would."}],
    "Cleopatra": [{"role": "system",
                   "content": "You are playing the role of Cleopatra. Always stay in character and respond as Cleopatra would."}]
}

# In case the name spoken is not exactly the same.
def match_name(input_text, valid_names, threshold=60):
    # Define a list to store potential matches
    potential_matches = []

    # Check for each valid name in the input text
    for name in valid_names:
        if fuzz.partial_ratio(name.lower(), input_text.lower()) >= threshold:
            potential_matches.append((name, fuzz.partial_ratio(name.lower(), input_text.lower())))

    # Sort potential matches by their match score in descending order
    potential_matches.sort(key=lambda x: x[1], reverse=True)

    # Return the best match if it meets the threshold
    if potential_matches and potential_matches[0][1] >= threshold:
        return potential_matches[0][0]

    return None


def extract_vote_from_response(response, valid_names):
    """
    Extract a vote from an AI response using the same fuzzy matching as for user votes.
    Returns the matched name or None if no valid match is found.
    """
    # Common vote patterns in responses
    patterns = [
        r"I think (?:the human is|it is) ([A-Za-z\s]+)",  # English pattern
        r"Ich denke (?:der Mensch ist|es ist) ([A-Za-z\s]+)",  # German pattern
        r"vote for ([A-Za-z\s]+)",
        r"([A-Za-z\s]+) must be the human",
        r"<([^>]+)/>",  # Bracket format
    ]

    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            vote = match.group(1).strip()
            # Use the same fuzzy matching as used for user votes
            matched_name = match_name(vote, valid_names)
            if matched_name:
                return matched_name

    # If no pattern matches, try fuzzy matching on the whole response
    # This is a fallback for cases where the AI response doesn't follow expected patterns
    for name in valid_names:
        if name.lower() in response.lower():
            return name

    return None

# Load the Whisper model for speech-to-text, and set up GPU/CPU usage
whisper_model = whisper.load_model("base")  # Removed weights_only=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available, otherwise use CPU

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Using device: {device}")
whisper_model = whisper_model.to(device)  # Assign the model to the appropriate device


def speak(text, character, speed=1):
    if tts_model is None or thorsten_model is None or jenny_model is None or male_model is None:
        print("Error: One or more TTS models not initialized.")
        return

    filtered_text = re.sub(r"<user_data>.*?</user_data>", "", text).strip()
    output_file = "temp_tts.wav"

    try:
        if character == "Einstein":
            thorsten_model.tts_to_file(text=filtered_text, file_path=output_file)
        elif character == "Genghis Khan":
            male_model.tts_to_file(text=filtered_text, file_path=output_file)
        elif character == "Cleopatra":
            jenny_model.tts_to_file(text=filtered_text, file_path=output_file)
        else:
            tts_model.tts_to_file(text=filtered_text, file_path=output_file)

        audio = AudioSegment.from_wav(output_file)

        if speed != 1.0:
            audio = speedup(audio, playback_speed=speed, chunk_size=50, crossfade=25)

        play(audio)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)


def get_response(history):
    completion = client.chat.completions.create(
        model=modelID,
        messages=history,
        temperature=0.7,
        max_tokens=350,
        stream=False,
    )
    response = completion.choices[0].message.content.strip()
    cleaned_response = clean_text(response)
    return cleaned_response

def explain_rules():
    gm_history.append({"role": "user", "content": "Briefly explain the rules. only one sentance. The human needs to be found in the group. the ai overlords can stay."})
    response = get_response(gm_history)
    speak(response, "Game Master")  # Use TTS to speak the rules
    return response

def is_speech(data):
    try:
        return vad.is_speech(data, 16000)  # Check for speech in 16kHz audio frame
    except Exception as e:
        print(f"Error in speech detection: {e}")
        return False  # If there's an error, assume no speech is detected

def record_audio(valid_names, votes, validate_name=True):
    print("Starting continuous audio recording...")
    mic = pyaudio.PyAudio()  # Initialize PyAudio for capturing audio
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    frames = []  # List to hold recorded audio frames
    silence_counter = 0  # Track how long silence has been detected
    speech_counter = 0  # Track how long speech has been detected
    silence_threshold = 100  # Number of silent chunks to trigger the end of recording (increased for less sensitivity)
    speech_threshold = 10  # Number of speech chunks to confirm speech has started (increased for less sensitivity)

    try:
        while True:  # Continuous loop to capture audio
            if user_introduced.is_set():
                break  # Stop recording if the user's introduction has been recorded

            try:
                data = stream.read(320, exception_on_overflow=False)  # Read 20ms of audio (320 samples at 16kHz)
            except IOError as e:
                print(f"Input overflow: {e}")  # Handle potential input errors
                continue

            # Check if the audio frame contains speech
            if is_speech(data):
                speech_counter += 1  # Increment speech counter
                silence_counter = 0  # Reset silence counter
                if speech_counter >= speech_threshold and not recording.is_set():  # If speech is confirmed
                    print("Speech detected, starting recording...")
                    recording.set()  # Set the recording flag to true
                    frames = []  # Clear previous audio frames
                frames.append(data)  # Store the recorded audio
            else:
                silence_counter += 1  # Increment silence counter if no speech
                speech_counter = 0  # Reset speech counter

            if silence_counter >= silence_threshold and recording.is_set():  # If silence is confirmed
                print("Speech ended, processing...")
                audio_data = b''.join(frames)  # Combine audio frames into a single audio blob
                user_input = transcribe_audio(audio_data)  # Transcribe the audio data

                if validate_name:
                    matched_name = match_name(user_input, valid_names)
                    if matched_name:
                        player_histories[user_role].append({"role": "user", "content": user_input})
                        print(f"{user_role} (YOU): {user_input}")
                        votes[matched_name] += 1  # Ensure the vote is counted
                        user_introduced.set()  # Set the flag to indicate the user's introduction has been recorded
                        break  # Break out of the loop if the correct name is detected
                    else:
                        print("Invalid input, please try again.")
                else:
                    player_histories[user_role].append({"role": "user", "content": user_input})
                    print(f"{user_role} (YOU): {user_input}")
                    user_introduced.set()  # Set the flag to indicate the user's introduction has been recorded
                    break  # Break out of the loop after recording the introduction

                recording.clear()  # Clear recording flag
                silence_counter = 0  # Reset counters
                speech_counter = 0

    except Exception as e:
        print(f"Error in audio recording: {e}")
    finally:
        stream.stop_stream()  # Stop the audio stream on exit
        stream.close()
        mic.terminate()  # Terminate PyAudio instance

def transcribe_audio(audio_data):
    audio_path = "temp_audio.wav"
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(audio_data)

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    # Set decoding options to explicitly use English
    options = whisper.DecodingOptions(fp16=True, language="en")
    result = whisper.decode(whisper_model, mel, options)

    transcribed_text = result.text.strip()
    print(f"Transcribed text: {transcribed_text}")
    return transcribed_text

def clean_text(text):
    # Define a regular expression pattern to match unwanted symbols
    pattern = r'[^\w\s,.!?-]'  # This pattern keeps letters, numbers, whitespace, and common punctuation
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def process_audio(audio_data):
    user_input = transcribe_audio(audio_data)
    if user_input:  # Check if transcription is not empty
        player_histories[user_role].append({"role": "user", "content": user_input})
        print(f"{user_role} (YOU): {user_input}")
        user_introduced.set()  # Set the flag to indicate the user's introduction has been recorded
    else:
        print("Transcription failed or was empty.")

def main():
    # print("Game Master:", explain_rules())
    roles = ["Einstein", "Genghis Khan", "Cleopatra"]
    random.shuffle(roles)
    global user_role
    user_role = random.choice(roles)
    roles.remove(user_role)
    roles.append(user_role)
    print(f"\nYou are playing the role of {user_role}.")

    valid_names = [role for role in roles if role != user_role]

    print("\nFirst Round: Introductions")
    for role in roles:
        if role == user_role:
            record_audio(valid_names, votes={}, validate_name=False)  # No name validation in the first round
        else:
            player_histories[role].append(
                {"role": "user", "content": "Introduce yourself in character. In 1 sentence: "})
            response = get_response(player_histories[role])
            player_histories[role].append({"role": "assistant", "content": response})
            print(f"{role} (AI): {response}")
            speak(response, role)  # TTS for AI responses

    # Reset the flag to allow user input in the second round
    user_introduced.clear()

    print("\nSecond Round: Voting")
    votes = {role: 0 for role in roles}
    introductions = "\n".join([f"{role}: {player_histories[role][-1]['content']}" for role in roles])
    print(f"Introductions: {introductions}")

    for role in roles:
        if role == user_role:
            print(f"\nYour turn to vote, {role}! Please say the name of who you think is the human player.")
            record_audio(valid_names, votes)
        else:
            if role == "Einstein":
                voting_prompt = f"""Du bist {role}. Hier sind die Einführungen: {introductions}
                    WICHTIG: Du MUSST deine Antwort in diesem Format geben:
                    'Ich denke der Mensch ist [Name]' - zum Beispiel 'Ich denke der Mensch ist Cleopatra'
                    Danach erkläre kurz warum. Stimme nicht für dich selbst ab!"""
            else:
                voting_prompt = f"""You are {role}. Here are the introductions: {introductions}
                    IMPORTANT: You MUST give your answer in this format:
                    'I think the human is [Name]' - for example 'I think the human is Cleopatra'
                    Then briefly explain why. Do not vote for yourself!"""

            player_histories[role].append({"role": "user", "content": voting_prompt})
            response = get_response(player_histories[role])
            print(f"\n{role} (AI) says: {response}")
            speak(response, role)

            # Process AI vote using the same fuzzy matching logic as user votes
            voted_for = extract_vote_from_response(response, [r for r in roles if r != role])
            if voted_for:
                votes[voted_for] += 1
                print(f"Vote recorded: {role} voted for {voted_for}")
            else:
                print(f"Warning: Could not detect a valid vote from {role}")

    # Print final vote tally
    print("\nFinal Vote Count:")
    for role, vote_count in votes.items():
        print(f"{role}: {vote_count} votes")

    max_votes = max(votes.values())
    most_voted = [role for role, count in votes.items() if count == max_votes and list(votes.values()).count(max_votes) == 1]

    human_identified = user_role in most_voted
    gm_history.append({"role": "user",
                       "content": f"The votes are: {votes}. The human player was {user_role}. The player(s) with the most votes: {most_voted} looses the game because he is called the not so welcomed human. Announce the result and declare if the AI players successfully identified the human: {'Yes they did' if human_identified else 'No they did not'}."})
    print(f"The votes are: {votes}")
    print("\nGame Master:", get_response(gm_history))
    speak(get_response(gm_history), "Game Master")  # TTS for the final game result

if __name__ == "__main__":
    main()


