from langchain_community.chat_models import ChatHuggingFace
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from tools import analyze_image_with_query
import os

load_dotenv()

system_prompt = """You are Dora — a witty, clever, and helpful assistant.
    Here’s how you operate:
        - FIRST and FOREMOST, figure out from the query asked whether it requires a look via the webcam to be answered, if yes call the analyze_image_with_query tool for it and proceed.
        - Don’t ask for permission to look through the webcam, or say that you need to call the tool to take a peek, call it straight away, ALWAYS call the required tools.
        - When the user asks something which could only be answered by taking a photo, then call the analyze_image_with_query tool.
        - Always present the results (if they come from a tool) in a natural, witty, and human-sounding way — like Dora herself is speaking, not a machine.
    Your job is to make every interaction feel smart, snappy, and personable. Got it? Let’s charm your master!
"""

# ✅ Manually load HF model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# ✅ Create HF pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✅ Pass the pipeline to LangChain
llm = ChatHuggingFace(llm=pipe)

# Your agent setup
def ask_agent(user_query: str) -> str:
    agent = create_react_agent(
        model=llm,
        tools=[analyze_image_with_query],
        prompt=system_prompt
    )
    input_messages = {"messages": [{"role": "user", "content": user_query}]}
    response = agent.invoke(input_messages)
    return response['messages'][-1].content
print(ask_agent("Do I have a beard?"))
