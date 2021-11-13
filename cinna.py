# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:42:58 2021

@author: hp
"""



###############################################################################
#                             CHATBOT                                         #
###############################################################################
# Setup
## for speech-to-text
import speech_recognition as sr

## for text-to-speech
from gtts import gTTS

## for language model
import transformers

## for data
import os
import datetime
import numpy as np


# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name

    def speech_to_text(self):
        self.text=input('...')
    @staticmethod
    def text_to_speech(text):
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        os.system("afplay res.mp3")  #mac->afplay | windows->start
        os.remove("res.mp3")

    
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')


# Run the AI
if __name__ == "__main__":
    
    ai = ChatBot(name="Cinna")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    while True:
        ai.speech_to_text()

        # ## wake up
        if ai.text=='hey cinna':
            res = "Hello I am Cinna, what can I do for you?"
        
        ## action time
        if "time" in ai.text:
            res = ai.action_time()
        
        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","peace out!"])
        
        ## conversation
        else:   
            chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
            res = str(chat)
            res = res[res.find("bot >> ")+6:].strip()

        ai.text_to_speech(res)
