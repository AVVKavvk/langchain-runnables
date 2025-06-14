from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from typing import Annotated,Literal

from dotenv import load_dotenv
load_dotenv()

model = ChatDeepSeek(model="deepseek-chat",max_tokens=300)

class Output(BaseModel):
  sentiment: Annotated[Literal["pos", "neg"], Field(description="Return sentiment of the review either negative, positive")]

parser = PydanticOutputParser(pydantic_object=Output)

prompt = PromptTemplate(
  template="Give me a sentiment of the review \n {feedback} \n {format_instructions}",
  input_variables=["feedback"],
  partial_variables={"format_instructions":parser.get_format_instructions()}
)

parserStr = StrOutputParser()

sentiment_chain = prompt | model | parser
# print(sentiment_chain.invoke({"feedback":"This product is bullshit"}).sentiment)

pos_prompt = PromptTemplate(
  template="Give me a positive response for this feedback \n {feedback}",
  input_variables=["feedback"],
)
neg_prompt = PromptTemplate(
  template="Give me a negative response for this feedback \n {feedback}",
  input_variables=["feedback"],
)

branch_chain = RunnableBranch(
  (lambda x: x.sentiment =="pos", pos_prompt | model | parserStr),
  (lambda x: x.sentiment == "neg", neg_prompt | model | parserStr),
  RunnableLambda(lambda x: "Could not found sentiment")
)

chain = sentiment_chain | branch_chain

result = chain.invoke({"feedback":"This product is bullshit"})
print(result)