import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field


os.environ["OPENAI_API_KEY"] = os.environ["OpenAI_Key"]

model = init_chat_model("gpt-4o-mini", model_provider="openai")


class Person(BaseModel):
    """
    Information about a person
    """

    name: str = Field(..., description="Then name of ther person")
    height: float = Field(
        ..., description="The height of the person"
    )


class People(BaseModel):
    """
    Identifying information about all people in text
    """

    people: List[Person]


parser = PydanticOutputParser(pydantic_object=People)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in 'json' tags\n(format_instructions)",
        ),
        ("human", "{query}")
    ]
).partial(format_instructions=parser.get_format_instructions())

query = "Anna is a woman and she is 6 feet tall"

chain = prompt | model | parser

result = chain.invoke({"query": query})
