from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatDeepSeek(model="deepseek-chat",max_tokens=500)

template1 = PromptTemplate(
  template=" Give me details details on {topic}",
  input_variables=["topic"],
  validate_template=True
)

template2 = PromptTemplate(
  template="Give me 5 important points on {text}?",
  input_variables=["text"],
  validate_template=True
)
parser= StrOutputParser()

chain = template1 | model | parser | template2 |model | parser

result= chain.invoke({
  "topic":"Black Holes"})

print(result)

chain.get_graph().print_ascii()