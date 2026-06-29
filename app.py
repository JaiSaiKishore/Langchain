# import os

# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#     model = "gpt-4o",
#     temperature = 0.7,
#     base_url = "https://models.inference.ai.azure.com"
# )

# response = llm.invoke("Explain langchain in simple words.")

# print(response)


import os
from langchain_openai import ChatOpenAI # OpenAIEmbeddings #embeddings-> to convert to vector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader #load documents and read text from files
from langchain_text_splitters import CharacterTextSplitter #split to chunks
from langchain_community.vectorstores import FAISS #stores vectors in vectordb, FAISS enable fast similarity search
from langchain_core.prompts import ChatPromptTemplate #reusable prompts
from langchain_core.output_parsers import StrOutputParser #model output to desired format
from dotenv import load_dotenv
load_dotenv()
# --------------------------------------------------------------
# 1. Set api key
# --------------------------------------------------------------

# os.environ["OPENAI_API_KEY"] = "key"

# --------------------------------------------------------------
# 2. Load the document
# --------------------------------------------------------------

loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

# --------------------------------------------------------------
# 3. Split the document into chunks
# --------------------------------------------------------------

textsplitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

docs = textsplitter.split_documents(documents)

# --------------------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------------------

# embeddings = OpenAIEmbeddings() openai provides embeddings but not github models
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------------------
# 5. Store document chunks in FAISS vector database
# --------------------------------------------------------------

vectorstore = FAISS.from_documents(docs, embeddings)

# --------------------------------------------------------------
# 6. create retriever
# --------------------------------------------------------------

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) #search_kwargs-> number of chunks to retrieve(top 3)

# --------------------------------------------------------------
# 7. create llm
# --------------------------------------------------------------

llm = ChatOpenAI(
    model = os.getenv("MODEL"),
    temperature = 0.7,
    base_url = "https://models.inference.ai.azure.com",
    api_key = os.getenv("OPENAI_API_KEY")
)

# --------------------------------------------------------------
# 8. Create prompt template
# --------------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. 
    Use only the provided context to answer the question.
    If the answer is not in the context, say:"I could not find that in the document"

context:
{context}
Question:
{question}
"""
)

# --------------------------------------------------------------
# 9. Start question-answer loop
# --------------------------------------------------------------

print("RAG app is ready. Type \"exit\" to quit")

while True:
    question = input("Ask a question: ")

    if(question.lower() == "exit"):
        print("Bye")
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build chain
    chain = prompt | llm | StrOutputParser()

    # Get response
    response = chain.invoke(
        {
            "context" : context,
            "question" : question
        }
    )
    print("\n Answer")
    print(response)
    print("\n"+"-"*60+"\n")
