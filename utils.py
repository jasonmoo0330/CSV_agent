from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from os import getenv


def qa_agent(memory, uploaded_files, question):
    model = ChatOpenAI(model = "deepseek-chat",
            openai_api_key='sk-a27705663bd2437b87dcabcba506f8ec',
            openai_api_base="https://api.deepseek.com",
            temperature=1)
    # print(uploaded_files)
    temp_file_path = "temp.pdf"
    file_content = uploaded_files.read()
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    embeddings_model = OpenAIEmbeddings(
        base_url = "https://api.aigc369.com/v1"
    )
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
