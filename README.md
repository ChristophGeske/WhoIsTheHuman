
# WhoIsTheHuman

**WhoIsTheHuman** is an AI-powered game that challenges players to disguise themselves as AI characters while trying to figure out who among them is the real human. The game uses AI models for character roleplay and voting, voice recognition for player input, and text-to-speech (TTS) for character responses.

Players must maintain their character persona while interacting with AI-generated characters like Einstein, Genghis Khan, and Cleopatra. At the end of each round, all players (including AI) vote on who they think is the human, and the player suspected the most loses.

## Features

- **Voice Recognition**: Players use voice commands to participate in the game.
- **AI Roleplay**: The game master and other characters are simulated by AI.
- **Text-to-Speech**: Responses are spoken aloud using TTS models.
- **Two-Round System**: 
  - In Round 1, players introduce themselves and maintain their character roles.
  - In Round 2, players vote on who they think is the human.

## How It Works

1. **Character Assignment**: The player is randomly assigned a historical character like Einstein, Genghis Khan, or Cleopatra.
2. **Round 1 (Introduction)**: Players introduce themselves in character, and AI-generated characters also respond in their specific personas.
3. **Round 2 (Voting)**: Players and AI vote on who they suspect is the human. AI votes based on reasoning provided by the language models.
4. **Results**: The game master counts the votes, and the player with the most votes loses (if they were suspected of being human).

### Setup

1. **Run Server on LM Studuio**:
   The game uses a local Large Language Model (LLM). Ensure you have the model running locally at `localhost:1234`.
   You can adjust this in the code by modifying the `client` initialization. The code here uses the Server function provided in LM Studio.

## Installation

To set up and run the game locally, follow these steps:

1. **Download the WhoIsTheHuman.py file**
2. Open the directory the game is downloaded to and type **cmd** in the directory field to open the comandline in this folder.
3. Install the required software:**

   ```bash
   pip install openai-whisper pyaudio webrtcvad pydub fuzzywuzzy torch openai tts
   ```
   
4. **Run the Game**:

   Start the game with the following command:
   
   ```bash
   python WhoIsTheHuman.py
   ```  

## How to Play

1. **Launch the game** and wait for the Ai's to introduce themself.
2. **Follow on-screen instructions** to introduce yourself in character.
3. **Interact via voice**: Your microphone will be used to detect your responses.
4. **Vote on the human** in the second round.
5. The game will announce the results via TTS, and the human player will be revealed.

## Technologies Used

- **Whisper**: For speech-to-text transcription.
- **WebRTC VAD**: For detecting voice activity in real-time.
- **PyAudio**: For capturing audio from the microphone.
- **FuzzyWuzzy**: For approximate string matching (used to find and match the role you voted for since speach recognition not always translates the name correctly).
- **Text-to-Speech**: The game uses the `TTS` library for different character voices.
- **OpenAI**: Used for generating responses from the game master and AI characters.

## Troubleshooting

- For local LLM issues, ensure that the server is up and running at the correct URL.

## Contributing

Feel free to fork this repository and submit pull requests if you'd like to contribute to improving the game. Issues and feature requests are welcome as well.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

