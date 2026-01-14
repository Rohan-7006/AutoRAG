import whisper
import json
import os

model = whisper.load_model("large-v2")

audios = os.listdir("Audio")

for audio in audios:
    title = audio.split(".")[0]
    result = model.transcribe(f"Audios/{audio}",language="hi",task="translate",word_timestamps=False)

    chunks = []
    for segment in result["segments"]:
        chunks.append({"Title" : title, "Start" : segment["start"], "End" : segment["end"],"Text" : segment["text"]})

    chunks_with_metadata = {"chunks" : chunks, "text" : result["text"]}

    with open(f"jsons/{audio}.json","w") as f:
        json.dump(chunks_with_metadata,f)