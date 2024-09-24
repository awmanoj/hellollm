from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"]
embeddings_model = OpenAIEmbeddings(model ='text-embedding-ada-002' )
embeddings = embeddings_model.embed_documents(
    [
        "Good morning!",
        "Oh, hello!",
        "I want to report an accident",
        "Sorry to hear that. May I ask your name?",
        "Sure, Mario Rossi."
    ]
)
print("Embed documents:")
print(f"Number of vector: {len(embeddings)}; Dimension of each vector: {len(embeddings[0])}")
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print("Embed query:")
print(f"Dimension of the vector: {len(embedded_query)}")
print(f"Sample of the first 5 elements of the vector: {embedded_query[:5]}")

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('dialogue.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0, separator = "\n",)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

query = "What is the reason for calling?"
docs = db.similarity_search(query)
print(docs[0].page_content)


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
query = "What was the reason of the call?"
qa.run(query)

