# quiz-me

Implementation for an app that allows you to upload a document and a set of questions.

The app then picks a random question to quiz you, and the answer can only come from the document you uploaded. 

This can be used by someone to prepare for exams or interviews. :D

## Install dependencies 

Install packages in the [requirements.txt](./requirements.txt): 

```
pip install -r requirements.txt
```


## Run it

To run the app, first change the [.env](./.env) file to use your OpenAI API key:

```
OPENAI_API_KEY=set_your_own_open_ai_key
```

You are all set.  Run: 

```
streamlit run app.py
```
