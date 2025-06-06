@echo off

REM 1. Terminal: frontend dizinine geç, npm run dev çalıştır
start cmd /k "cd /d %~dp0\frontend && npm run dev"

REM 2. Terminal: gateway dizinine geç, node server.js çalıştır
start cmd /k "cd /d %~dp0\gateway\src && node index.js"

REM 3. Terminal: .venv aktive et, emotion-service dizinine geç, python emotion_server.py çalıştır
start cmd /k "cd /d %~dp0 && .venv\Scripts\activate && cd emotion-service && python emotion_server.py"

REM 4. Terminal: .venv aktive et, vision-service dizinine geç, python vision_server.py çalıştır
start cmd /k "cd /d %~dp0 && .venv\Scripts\activate && cd vision-service && python vision_server.py"

REM 5. Terminal: .venv aktive et, speech-service dizinine geç, python speech_server.py çalıştır
start cmd /k "cd /d %~dp0 && .venv\Scripts\activate && cd speech-service && python speech_server.py"