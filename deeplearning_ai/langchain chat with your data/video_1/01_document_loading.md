# Document Loading

## Note to students.
During periods of high load you may find the notebook unresponsive. It may appear to execute a cell, update the completion number in brackets [#] at the left of the cell but you may find the cell has not executed. This is particularly obvious on print statements when there is no output. If this happens, restart the kernel using the command under the Kernel tab.

## Retrieval augmented generation
 
In retrieval augmented generation (RAG), an LLM retrieves contextual documents from an external dataset as part of its execution. 

This is useful if we want to ask question about specific documents (e.g., our PDFs, a set of videos, etc). 

![overview.jpeg](overview.jpeg)


```python
#! pip install langchain
```


```python
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
```

## PDFs

Let's load a PDF [transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf) from Andrew Ng's famous CS229 course! These documents are the result of automated transcription so words and sentences are sometimes split unexpectedly.


```python
# The course will show the pip installs you would need to install packages on your own machine.
# These packages are already installed on this platform and should not be run again.
#! pip install pypdf 
```


```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
```

Each page is a `Document`.

A `Document` contains text (`page_content`) and `metadata`.


```python
len(pages)
```




    22




```python
page = pages[0]
```


```python
print(page.page_content[0:500])
```

    MachineLearning-Lecture01  
    Instructor (Andrew Ng):  Okay. Good morning. Welcome to CS229, the machine 
    learning class. So what I wanna do today is ju st spend a little time going over the logistics 
    of the class, and then we'll start to  talk a bit about machine learning.  
    By way of introduction, my name's  Andrew Ng and I'll be instru ctor for this class. And so 
    I personally work in machine learning, and I' ve worked on it for about 15 years now, and 
    I actually think that machine learning i



```python
page.metadata
```




    {'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 0}



## YouTube


```python
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
```


```python
# ! pip install yt_dlp
# ! pip install pydub
```

**Note**: This can take several minutes to complete.


```python
url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
```

    [youtube] Extracting URL: https://www.youtube.com/watch?v=jGwO_UgTS7I
    [youtube] jGwO_UgTS7I: Downloading webpage
    [youtube] jGwO_UgTS7I: Downloading ios player API JSON
    [youtube] jGwO_UgTS7I: Downloading android player API JSON


    WARNING: [youtube] YouTube said: ERROR - Precondition check failed.
    WARNING: [youtube] HTTP Error 400: Bad Request. Retrying (1/3)...


    [youtube] jGwO_UgTS7I: Downloading android player API JSON


    WARNING: [youtube] YouTube said: ERROR - Precondition check failed.
    WARNING: [youtube] HTTP Error 400: Bad Request. Retrying (2/3)...


    [youtube] jGwO_UgTS7I: Downloading android player API JSON


    WARNING: [youtube] YouTube said: ERROR - Precondition check failed.
    WARNING: [youtube] HTTP Error 400: Bad Request. Retrying (3/3)...


    [youtube] jGwO_UgTS7I: Downloading android player API JSON


    WARNING: [youtube] YouTube said: ERROR - Precondition check failed.
    WARNING: [youtube] Unable to download API page: HTTP Error 400: Bad Request (caused by <HTTPError 400: 'Bad Request'>); please report this issue on  https://github.com/yt-dlp/yt-dlp/issues?q= , filling out the appropriate issue template. Confirm you are on the latest version using  yt-dlp -U


    [youtube] jGwO_UgTS7I: Downloading player bc657243
    [youtube] jGwO_UgTS7I: Downloading m3u8 information
    [info] jGwO_UgTS7I: Downloading 1 format(s): 140
    [download] Destination: docs/youtube//Stanford CS229： Machine Learning Course, Lecture 1 - Andrew Ng (Autumn 2018).m4a
    [download] 100% of   69.76MiB in 00:00:01 at 39.72MiB/s  
    [FixupM4a] Correcting container of "docs/youtube//Stanford CS229： Machine Learning Course, Lecture 1 - Andrew Ng (Autumn 2018).m4a"
    [ExtractAudio] Not converting audio docs/youtube//Stanford CS229： Machine Learning Course, Lecture 1 - Andrew Ng (Autumn 2018).m4a; file is already in target format m4a
    Transcribing part 1!



```python
docs[0].page_content[0:500]
```

## URLs


```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
```


```python
docs = loader.load()
```


```python
print(docs[0].page_content[:500])
```

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    File not found · GitHub
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    Skip to content
    
    
    
    
    
    
    
    
    
    
    
    
    Navigation Menu
    
    Toggle navigation
    
    
    
    
    
    
    
    
    
    
              Sign in
            
    
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
            Product
            
    
    
    
    
    
    
    
    
    
    
    
    
    Actions
            Automate any workflow
          
    
    
    
    
    
    
    
    Packages
            Host and manage packages
          
    
    
    
    
    
    
    
    Security
            Find and fix vulnerabilities
          
    
    
    
    
    
    
    
    Codespaces
            Instant


## Notion

Follow steps [here](https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/notion) for an example Notion site such as [this one](https://yolospace.notion.site/Blendle-s-Employee-Handbook-e31bff7da17346ee99f531087d8b133f):

* Duplicate the page into your own Notion space and export as `Markdown / CSV`.
* Unzip it and save it as a folder that contains the markdown file for the Notion page.
 

![image.png](./img/image.png)


```python
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
```


```python
print(docs[0].page_content[0:200])
```


```python
docs[0].metadata
```


```python

```
