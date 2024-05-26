# Question Answering

## Overview

Recall the overall workflow for retrieval augmented generation (RAG):

![overview.jpeg](overview.jpeg)

We discussed `Document Loading` and `Splitting` as well as `Storage` and `Retrieval`.

Let's load our vectorDB. 


```python
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
```

The code below was added to assign the openai LLM version filmed until it is deprecated, currently in Sept 2023. 
LLM responses can often vary, but the responses may be significantly different when using a different model version.


```python
import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)
```

    gpt-3.5-turbo



```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
```


```python
print(vectordb._collection.count())
```

    209



```python
question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)
```




    3




```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)
```

### RetrievalQA chain


```python
from langchain.chains import RetrievalQA
```


```python
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
```


```python
result = qa_chain({"query": question})
```


```python
result["result"]
```




    'The major topics for this class include machine learning, statistics, and algebra. Additionally, there will be discussions on extensions of the material covered in the main lectures.'



### Prompt


```python
from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

```


```python
# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
```


```python
question = "Is probability a class topic?"
```


```python
result = qa_chain({"query": question})
```


```python
result["result"]
```




    'Yes, probability is a class topic as the instructor assumes familiarity with basic probability and statistics. Thanks for asking!'




```python
result["source_documents"][0]
```




    Document(page_content="of this class will not be very program ming intensive, although we will do some \nprogramming, mostly in either MATLAB or Octa ve. I'll say a bit more about that later.  \nI also assume familiarity with basic proba bility and statistics. So most undergraduate \nstatistics class, like Stat 116 taught here at Stanford, will be more than enough. I'm gonna \nassume all of you know what ra ndom variables are, that all of you know what expectation \nis, what a variance or a random variable is. And in case of some of you, it's been a while \nsince you've seen some of this material. At some of the discussion sections, we'll actually \ngo over some of the prerequisites, sort of as  a refresher course under prerequisite class. \nI'll say a bit more about that later as well.  \nLastly, I also assume familiarity with basi c linear algebra. And again, most undergraduate \nlinear algebra courses are more than enough. So if you've taken courses like Math 51, \n103, Math 113 or CS205 at Stanford, that would be more than enough. Basically, I'm \ngonna assume that all of you know what matrix es and vectors are, that you know how to \nmultiply matrices and vectors and multiply matrix and matrices, that you know what a matrix inverse is. If you know what an eigenvect or of a matrix is, that'd be even better. \nBut if you don't quite know or if you're not qu ite sure, that's fine, too. We'll go over it in \nthe review sections.", metadata={'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 4})



### RetrievalQA chain types


```python
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
```


```python
result = qa_chain_mr({"query": question})
```


```python
result["result"]
```




    'Yes, probability is a class topic in the document. The instructor assumes familiarity with basic probability and statistics, including concepts like random variables, expectation, variance, and more. The course will involve some programming, mostly in MATLAB or Octave, and will also cover basic linear algebra.'



If you wish to experiment on the `LangSmith platform` (previously known as LangChain Plus):

 * Go to [LangSmith](https://www.langchain.com/langsmith) and sign up
 * Create an API key from your account's settings
 * Use this API key in the code below   
 * uncomment the code  
 Note, the endpoint in the video differs from the one below. Use the one below.


```python
#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
#os.environ["LANGCHAIN_API_KEY"] = "..." # replace dots with your api key
```


```python
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]
```




    'Yes, probability is a class topic in the document. The instructor assumes familiarity with basic probability and statistics, including concepts such as random variables, expectation, variance, and random variables. The document mentions that most undergraduate statistics classes, like Stat 116 at Stanford, will be more than enough preparation for the course.'




```python
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
result["result"]
```




    'The additional context provided does not directly impact the original answer about probability being a class topic in the course. The course will cover probability as a class topic, specifically in the context of deriving learning algorithms and addressing classification problems. The discussion sections may also touch on related topics like statistics or algebra as refreshers, and extensions to the main lecture material may be covered in these sections as well.'



### RetrievalQA limitations
 
QA fails to preserve conversational history.


```python
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
```


```python
question = "Is probability a class topic?"
result = qa_chain({"query": question})
result["result"]
```




    'Yes, probability is a class topic in the course being described. The instructor assumes familiarity with basic probability and statistics.'




```python
question = "why are those prerequesites needed?"
result = qa_chain({"query": question})
result["result"]
```




    'The prerequisites mentioned in the context are needed for the machine learning class because the instructor assumes that students already have a basic knowledge of computer science and computer skills. This foundational knowledge is essential for understanding the concepts and algorithms taught in the class.'



Note, The LLM response varies. Some responses **do** include a reference to probability which might be gleaned from referenced documents. The point is simply that the model does not have access to past questions or answers, this will be covered in the next section.


```python

```
