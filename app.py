from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai
import numpy as np
import random
import langchain
langchain.verbose = False


def compute_similarity(answer, gpt_response):
    resp = openai.Embedding.create(input=[answer, gpt_response],
                                   engine="text-similarity-davinci-001")

    similarity = np.dot(resp['data'][0]['embedding'], resp['data'][1]['embedding'])
    st.write("Score: {:.0%}".format(similarity))

    if similarity > 0.8:
        st.write(":white_check_mark: Well done!")
    elif 0.6 < similarity < 0.8:
        st.write(":sparkles: Not bad!")
    elif similarity < 0.6:
        st.write(":muscle: Check out the answer below!")


def process_questions(pdf_q):
    q_reader = PdfReader(pdf_q)
    q_text = ""
    for page in q_reader.pages:
        q_text += page.extract_text()
    questions = q_text.split('\n')
    questions = [q.strip() for q in questions]
    return questions


def process_answer_chunks(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def main():
    load_dotenv()
    st.set_page_config(page_title="Quiz me")
    st.header("Quiz me 💬")

    # upload file
    pdf_q = st.file_uploader("Upload your PDF that contains your questions", type="pdf")
    pdf = st.file_uploader("Upload your PDF that contains your answers", type="pdf")

    # extract the text
    if pdf is not None and pdf_q is not None:
        questions = process_questions(pdf_q)
        chunks = process_answer_chunks(pdf)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        if 'stage' not in st.session_state:
            st.session_state.stage = 0

        if 'question_selected' not in st.session_state:
            st.session_state.question_selected = None

        def set_state(i):
            st.session_state.stage = i

        st.button('Ask me a question :raising_hand:!', on_click=set_state, args=[1])
        if st.session_state.stage >= 1:
            if st.session_state.stage == 1:
                while True:
                    current_q = random.choice(questions)
                    if len(current_q.split(' ')) > 2 and (current_q[-1] == '?' or current_q[-1] == '.'):
                        st.session_state.question_selected = current_q
                        break
            st.write(st.session_state.question_selected)

            docs = knowledge_base.similarity_search(st.session_state.question_selected)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=st.session_state.question_selected)
                print(cb)

            with st.form(key='my_form'):
                user_answer = st.text_input("What is your answer?")
                st.form_submit_button(label='Submit', on_click=set_state, args=[2])

                if st.session_state.stage >= 2:
                    st.write("Grading...")
                    set_state(3)
                    compute_similarity(user_answer, response)

            with st.expander("See answer"):
                st.write(response)

            if st.session_state.stage >= 3:
                st.button('Start Over', on_click=set_state, args=[0])


if __name__ == '__main__':
    main()