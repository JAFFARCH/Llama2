import os
import sys
import re
import pandas as pd
import numpy as np
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import openai
from dotenv import load_dotenv

from create_vector_db import create_vector_db

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

custom_prompt_template="""Use the following ........
Context:{}
Question:{question}

only return helpful answer below"""

DATA_PATH="data/"
DB_FAISS_PATH = "vectorstores/db_fiass/"

current_dir = os.path.dirname(os.path.abspath(__file__))
chess_piece_dir = os.path.join(current_dir, "..", "chess_piece")
sys.path.append(chess_piece_dir)
# from king import hive_master_root, print_line_of_error
def hive_master_root(info='\pollen\pollen'):
    script_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(script_path)) # \pollen\pollen
def print_line_of_error(e='print_error_message'):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(e, exc_type, exc_tb.tb_lineno)
    return exc_type, exc_tb.tb_lineno


load_dotenv(os.path.join(hive_master_root(), '.env'))


class EmbeddingUtility:
    def __init__(self, embedding_model="text-embedding-ada-002"):
        self.embedding_model = embedding_model
        openai.api_key = os.environ.get('ozz_api_key')

    def get_embedding(self, text):
        text = text.replace("/n", " ")
        return openai.Embedding.create(input=[text], model=self.embedding_model)['data'][0]['embedding']

    def get_doc_embedding(self, text):
        return self.get_embedding(text)


class DocumentProcessor:
    MAX_SECTION_LEN = 1024
    SEPARATOR = "/n* "

    def __init__(self, api_params, utility: EmbeddingUtility):
        self.COMPLETIONS_API_PARAMS = api_params
        self.utility = utility

    def compute_matching_df_embeddings(self, matching_df):
        return {
            idx: self.utility.get_doc_embedding(r.description.replace("/n", " ")) for idx, r in matching_df.iterrows()
        }

    @staticmethod
    def vector_similarity(x, y):
        return np.dot(np.array(x), np.array(y))

    def order_document_sections_by_query_similarity(self, query, contexts):
        query_embedding = self.utility.get_embedding(query)
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)
        return document_similarities

    def construct_prompt(self, question, context_embeddings, df):
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(question, context_embeddings)
        chosen_sections = []
        chosen_sections_len = 0

        for _, section_index in most_relevant_document_sections:
            document_section = df.loc[section_index]
            chosen_sections_len += len(document_section.description) + 3
            if chosen_sections_len > self.MAX_SECTION_LEN:
                break
            chosen_sections.append(self.SEPARATOR + document_section.description.replace("/n", " "))

        header = """Answer the question as truthfully as possible only using the provided context. and if the answer is not contained within the text below, say "I don't know" /n/nContext:/n"""
        return header + "".join(chosen_sections) + "/n/n Q: " + question + "/n A:"

    def answer_query_with_context(self, query, df, context_embeddings, show_prompt=False):
        prompt = self.construct_prompt(query, context_embeddings, df)
        if show_prompt:
            print(prompt)
        response = openai.Completion.create(prompt=prompt, **self.COMPLETIONS_API_PARAMS)
        return response["choices"][0]["text"].strip(" /n")


def preprocessing(df):
    for i in df.index:
        desc = df.loc[i, 'description']
        if isinstance(desc, str):
            df.loc[i, 'description'] = desc.replace('/n', '')
            df.loc[i, 'description'] = re.sub(r'/(.*?/)', '', df.loc[i, 'description'])
            df.loc[i, 'description'] = re.sub('[/(/[].*?[/)/]]', '', df.loc[i, 'description'])
    return df


def closest_sentiment_match(dfs, query_sentiment):
    closest_index = None
    closest_difference = float('inf')
    analyzer = SentimentIntensityAnalyzer()

    for index, sub_df in enumerate(dfs):
        titles = sub_df['title'].tolist()
        aggregated_sentiment = 0

        for title in titles:
            aggregated_sentiment += analyzer.polarity_scores(title)['compound']

        average_sentiment = aggregated_sentiment / len(titles)
        difference = abs(average_sentiment - query_sentiment)

        if difference < closest_difference:
            closest_difference = difference
            closest_index = index
    return closest_index










def set_custome_prompt():
    """Prompt template"""

    prompt=PromptTemplate(template=custom_prompt_template,input_variable=['context','question'])

    return prompt

def retrieva_qa_chain(llm,prompt,db):
    qa_chain= RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt})

    return qa_chain

def load_llm():
    llm=CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.1
    )
    return llm


def qa_bot():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformer/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
    
    db=FAISS.load_local(DB_FAISS_PATH,embeddings)
    llm= load_llm()
    qa_prompt=set_custome_prompt()
    qa=retrieva_qa_chain(llm,qa_prompt,db)

    return qa

def final_result(query):
    qa_result=qa_bot()
    response=qa_result({'query':query})
    return response

@cl.on_chat_start 


def main(filename, chunk_rows=33):
    df = pd.read_csv(filename, engine='python', on_bad_lines='skip')
    # print(df)
    df.dropna(subset=['title', 'location'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = preprocessing(df)
    # print(df)

    
    # dfs_25 = [df.iloc[i:i+chunk_rows] for i in range(0, len(df), chunk_rows)]

    # analyzer = SentimentIntensityAnalyzer()
    # query = "tell me more about the jobs"
    # query_sentiment = analyzer.polarity_scores(query)['compound']
    # matched_index = closest_sentiment_match(dfs_25, query_sentiment)
    # print(matched_index)
    # Steps
    # return llama_ozz response
    # 1. read file ensure file format hanlde different files update create_vector_db

    df= create_vector_db(df)
    print(df)


    # 2. get response from llama_ozz

    

    # 3. deploy in streamlit llama_ozz_app.py
    
    # print("The closest matching index in dfs_25 is:", matched_index)
    # print(dfs_25[matched_index])
    # print(df.columns)

    # utility = EmbeddingUtility()
    # doc_processor = DocumentProcessor({
    #     "temperature": 0.0,
    #     "max_tokens": 300,
    #     "model": "text-davinci-003",
    # }, utility)

    # # # print(dfs_25[matched_index])

    # matching_df_embeddings = doc_processor.compute_matching_df_embeddings(dfs_25[matched_index])
    # resp = doc_processor.answer_query_with_context(query, dfs_25[matched_index], matching_df_embeddings)
    # print(resp)


if __name__ == "__main__":
    try:
        filename = "C:/Users/hp/Desktop/Viral Square/Stephan/largeDataset/pollen/ozz/fake_job_postings2.txt"  # Provide the correct file path here
        main(filename)
    except Exception as e:
        print(e)
        print_line_of_error()





# What kind of jobs are available for teachers abroad?
# What is the monthly salary for jobs in Asia for teachers?
# What qualifications are required for the English Teacher Abroad position?
# What responsibilities does the RMA Coordinator have in the Customer Success team?
# What are the preferred qualifications for the Junior Python Developer position?
# What are the values that Upstream believes in as mentioned in the job description?
# What are the main responsibilities of the Customer Service Associate at Novitex Enterprise Solutions?
# What are the key requirements for the Data Entry Clerk II position?
# What are the job responsibilities of an Engagement Executive at Upstream?
# How many years of work experience are required for the Engagement Executive position?