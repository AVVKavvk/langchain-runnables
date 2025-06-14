from langchain_deepseek import ChatDeepSeek
from langchain.schema.runnable import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

model1 = ChatDeepSeek(model="deepseek-chat",max_tokens=1000)
model2 = ChatDeepSeek(model="deepseek-chat",max_tokens=1000)

prompt1= PromptTemplate(
  template="Give me a short notes on given text \n {text}",
  input_variables=["text"],
)

prompt2 = PromptTemplate(
  template="Give me 5 question and answers on given text \n {text}",
  input_variables=["text"],
)

prompt3 = PromptTemplate(
  template="Merge the following notes and quiz \n notes -> {notes} \n quiz -> {quiz}",
  input_variables=["notes","quiz"],
)

parser =StrOutputParser()

parallel_chain  = RunnableParallel({
  "notes": prompt1 | model1 | parser,
  "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 |model1 | parser

chain = parallel_chain | merge_chain

text="""
SVM (Support Vector Machine): A supervised learning algorithm used for classification and regression tasks.
Hyperplane: A decision boundary separating different classes in feature space and is represented by the equation wx + b = 0 in linear classification.
Support Vectors: The closest data points to the hyperplane, crucial for determining the hyperplane and margin in SVM.
Margin: The distance between the hyperplane and the support vectors. SVM aims to maximize this margin for better classification performance.
Kernel: A function that maps data to a higher-dimensional space enabling SVM to handle non-linearly separable data.
Hard Margin: A maximum-margin hyperplane that perfectly separates the data without misclassifications.
Soft Margin: Allows some misclassifications by introducing slack variables, balancing margin maximization and misclassification penalties when data is not perfectly separable.
C: A regularization term balancing margin maximization and misclassification penalties. A higher C value forces stricter penalty for misclassifications.
Hinge Loss: A loss function penalizing misclassified points or margin violations and is combined with regularization in SVM.
Dual Problem: Involves solving for Lagrange multipliers associated with support vectors, facilitating the kernel trick and efficient computation.
"""
result = chain.invoke({
  "text":text
})
print(result)

chain.get_graph().print_ascii()