# Python app for HuggingFace Inferences
# Only API Access token from Huggingface.co
# libraries for AI inferences
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests
# Internal usage
import os
import datetime

yourHFtoken = "hf_xxxxxxxxxxxxxx"  #paste your token here

def imageToText(url):
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=yourHFtoken)
    model_Image2Text = "Salesforce/blip-image-captioning-base"
    # tasks from huggingface.co/tasks
    text = client.image_to_text(url,
                                model=model_Image2Text)
    print(text)
    return text

basetext = imageToText("./photo.jpeg")


def Text2Image(text):
  from huggingface_hub import InferenceClient
  client = InferenceClient(model="runwayml/stable-diffusion-v1-5", token=yourHFtoken)
  image = client.text_to_image(text)
  image.save("yourimage.png")

myiimage = Text2Image("An astronaut riding a horse on the moon.")


def  text2speech(text):
  import requests
  API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
  headers = {"Authorization": f"Bearer {yourHFtoken}"}

  payloads = {
      "inputs" : text
  }
  response = requests.post(API_URL, headers=headers, json=payloads)
  with open('audio.flac', 'wb') as file:
    file.write(response.content)

mytext = "So let's create a function for our text-to-speech generation with the requests method"
text2speech(mytext)


def generation(question):
  from langchain import HuggingFaceHub
  os.environ["HUGGINGFACEHUB_API_TOKEN"] = yourHFtoken
  repo_id = "MBZUAI/LaMini-Flan-T5-248M"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
  llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 64})
  from langchain import PromptTemplate, LLMChain
  template = """Question: {question}

  Answer: Let's think step by step."""
  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  print(llm_chain.run(question))

question = "Who won the FIFA World Cup in the year 1994? "
generation(question)