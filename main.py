# This is a sample Python script.
import string
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import Union

from gradio_client import Client
from pydantic import BaseModel
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()
class Item(BaseModel):
    q: str
    sex:str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/texttospeech/")
async  def text_to_speech(item: Item):

    link = "https://ntt123-vietnam-male-voice-tts.hf.space/"
    if(item.sex=='female'):
            link ="https://ntt123-vietnam-female-voice-tts.hf.space/"

    client = Client(link)
    result = client.predict(
        item.q,  # str in 'text' Textbox component
        api_name="/predict"
    )



    return FileResponse(result, filename='text.wav')


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
