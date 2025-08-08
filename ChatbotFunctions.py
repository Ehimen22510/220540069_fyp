from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer
from dotenv import load_dotenv
import os

def retrieval(vector_store: PineconeVectorStore,question, knn):

    query = question
    Generalcontext = vector_store.similarity_search_with_score(query, k = knn, namespace = "General")
    ugcontext = vector_store.similarity_search_with_score(query, k = knn, namespace = "Undergraduate")
    pgcontext = vector_store.similarity_search_with_score(query, k = knn, namespace = "Postgraduate")

    allcontext = Generalcontext + ugcontext + pgcontext

    allcontext = [item for item in allcontext if item[1] > 0.5]
    allcontext = sorted(allcontext, key = lambda x: x[1],reverse = True)
    context = []

    if (len(allcontext) >= knn):
        context = [allcontext[i][0] for i in range(knn)]
    else:
        context = [allcontext[i][0] for i in range(len(allcontext))]

    return context




def createprompt():
    temp = "please answer the following question {question} using the context {context}. You are an assistant in Queen mary's university of london and you are answering a student, if there is no context then ask user to elaborate on their question without using context,relate it to queen mary's EECS department. write within 100 words and use a friendly demeanor"
    prompt = ChatPromptTemplate.from_template(temp)
    return prompt


def generation(question,  model : ChatOpenAI, context, prompt):
    chain = prompt | model | StrOutputParser()

    b = chain.invoke({"context":context, "question":question})

    return b

def full_pipeline(vectorstore: PineconeVectorStore,question, knn, model : ChatOpenAI):
    context = retrieval(vector_store=vectorstore, question=question,knn = knn)
    prompt = createprompt()
    response = generation(vector_store=vectorstore,question=question,model=model,context=context, prompt=prompt)
    return response