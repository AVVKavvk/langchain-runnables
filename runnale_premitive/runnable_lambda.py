from langchain_deepseek import ChatDeepSeek
from langchain.schema.runnable import RunnableLambda, RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatDeepSeek(model="deepseek-chat",max_tokens=100)

prompt = PromptTemplate(
  template="Give me a joke on {topic}",
  input_variables=["topic"],
)

parser = StrOutputParser()

joke_chain = prompt | model |parser

def word_count(text):
  return len(text.split(" "))

word_count_runnable = RunnableLambda(word_count)

parallel_chain = RunnableParallel({
  "joke": RunnablePassthrough(),
  "word_count": word_count_runnable
})

chain = joke_chain | parallel_chain

result = chain.invoke({
  "topic": "black holes"
})

print(result)