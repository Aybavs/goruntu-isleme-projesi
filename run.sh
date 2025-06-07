#!/bin/bash

kitty --title "Frontend" --detach bash -c "cd frontend && npm run dev"
kitty --title "Gateway" --detach bash -c "cd gateway && node server.js"
kitty --title "Emotion Service" --detach bash -c "source .venv/bin/activate && cd emotion-service && python3 emotion_server.py"
kitty --title "Vision Service" --detach bash -c "source .venv/bin/activate && cd vision-service && python3 vision_server.py"
kitty --title "Speech Service" --detach bash -c "source .venv/bin/activate && cd speech-service && python3 speech_server.py"
