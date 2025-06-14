from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

model = ChatDeepSeek(model="deepseek-chat",max_tokens=100)

prompt1 = PromptTemplate(
  template="Give me a joke on {topic}",
  input_variables=["topic"],
)

prompt2 = PromptTemplate(
  template="Explain me this joke /n {joke}",
  input_variables=["joke"],
)

parser = StrOutputParser()
joke_chain = prompt1 | model | parser

# joke = joke_chain.invoke({
#   "topic": "black holes"
# })

# chain = RunnableParallel({
#   "joke": RunnablePassthrough(),
#   "explain": prompt2 | model | parser
# })

# result = chain.invoke({
#   "joke":joke
# })
parallel_chain = RunnableParallel({
  "joke": RunnablePassthrough(),
  "explain": prompt2 | model | parser
})

chain = joke_chain | parallel_chain

result = chain.invoke({
  "topic": "black holes"
})

print(result)