import os
from langchain.chat_models import init_chat_model


os.environ["OPENAI_API_KEY"] = os.environ["OpenAI_Key"]

model = init_chat_model("gpt-4o-mini", model_provider="openai")
result = model.invoke("I am Captain Jack Sparrow. Savy.")
print(result)
