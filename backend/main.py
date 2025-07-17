


from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import uuid
import validators
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, UnstructuredURLLoader
import shutil
import requests
from bs4 import BeautifulSoup
from pytubefix import YouTube
from openai import OpenAI

app = FastAPI(title="RAG Q&A API with Summarization", description="Conversational RAG with PDF uploads, chat history, and URL/YouTube summarization")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for sessions and vector stores
session_store = {}
vector_stores = {}
chat_chains = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str
    openai_api_key: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class SummarizeRequest(BaseModel):
    url: str
    openai_api_key: str

class SummarizeResponse(BaseModel):
    summary: str
    url: str
    content_type: str

class SessionInfo(BaseModel):
    session_id: str
    documents_count: int
    has_vectorstore: bool

@app.get("/")
async def root():
    return {"message": "RAG Q&A API with Summarization is running"}

@app.post("/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
    openai_api_key: str = Form(...)
):
    """Upload PDF documents and create vector store"""
    try:
        if not openai_api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required")
        
        # Validate OpenAI API key by creating embeddings instance
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
        
        documents = []
        
        # Process uploaded files
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Load PDF content
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            documents.extend(docs)
            
            # Clean up temp file
            os.unlink(temp_file_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents could be processed")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name=f"collection_{session_id}"
        )
        
        # Store vector store and create RAG chain
        vector_stores[session_id] = vectorstore
        
        # Create LLM
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever()
        
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Answer prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Create conversational chain
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in session_store:
                session_store[session] = ChatMessageHistory()
            return session_store[session]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        chat_chains[session_id] = conversational_rag_chain
        
        return {
            "message": f"Successfully processed {len(files)} documents",
            "session_id": session_id,
            "documents_count": len(documents),
            "chunks_count": len(splits)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import os
import validators
import requests
from bs4 import BeautifulSoup
from pytubefix import YouTube
from openai import OpenAI

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_content(request: SummarizeRequest):
    """Summarize content from YouTube or website URL"""
    try:
        if not request.openai_api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required")
        
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Validate URL
        if not validators.url(request.url):
            raise HTTPException(status_code=400, detail="Please enter a valid URL. It can be a YouTube video URL or website URL")
        
        # Create OpenAI client
        try:
            client = OpenAI(api_key=request.openai_api_key)
            # Test API key
            client.models.list()
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
        
        # Extract content based on URL type
        content_type = "website"
        content = ""
        
        try:
            if "youtube.com" in request.url or "youtu.be" in request.url:
                content_type = "youtube"
                print("Processing YouTube video...")
                
                # Extract YouTube content
                yt = YouTube(request.url)
                audio = yt.streams.filter(only_audio=True).first()
                
                if not audio:
                    raise HTTPException(status_code=400, detail="No audio stream found for this video")
                
                filename = audio.download()
                
                # Transcribe audio
                with open(filename, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="text",
                        prompt="Provide clear transcription. Return empty if no speech detected."
                    )
                
                # Clean up file
                os.remove(filename)
                
                # Add video info
                content = f"Video: {yt.title}\nChannel: {yt.author}\n\nTranscript:\n{transcription}"
                
            else:
                content_type = "website"
                print("Processing website...")
                
                # Extract website content
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                }
                
                response = requests.get(request.url, headers=headers, timeout=30, verify=False)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                # Extract text
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
            
            if not content or len(content.strip()) < 50:
                raise HTTPException(status_code=400, detail="No sufficient content could be extracted from the URL")
            
            # Generate summary using OpenAI
            if len(content) > 12000:
                content = content[:12000] + "..."
            
            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise summaries in approximately 300 words."
                    },
                    {
                        "role": "user",
                        "content": f"Provide a summary of the following content:\n\n{content}"
                    }
                ],
                temperature=0,
                max_tokens=500
            )
            
            summary = summary_response.choices[0].message.content.strip()
            print("Summary generated successfully")
            
            return SummarizeResponse(
                summary=summary,
                url=request.url,
                content_type=content_type
            )
            
        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            if 'filename' in locals() and os.path.exists(filename):
                os.remove(filename)
            raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")
            
    except Exception as e:
        print(f"General error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
'''@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_content(request: SummarizeRequest):
    """Summarize content from YouTube or website URL"""
    try:
        if not request.openai_api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required")
        
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Validate URL
        if not validators.url(request.url):
            raise HTTPException(status_code=400, detail="Please enter a valid URL. It can be a YouTube video URL or website URL")
        
        # Create LLM instance
        try:
            llm = ChatOpenAI(
                openai_api_key=request.openai_api_key,
                model_name="gpt-4.1",
                temperature=0
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
        
        # Create prompt template
        prompt_template = """
        Provide a summary of the following content in 300 words:
        Content: {text}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        # Load content based on URL type
        content_type = "website"
        try:
            if "youtube.com" in request.url or "youtu.be" in request.url:
                loader = YoutubeLoader.from_youtube_url(request.url, add_video_info=True)
                content_type = "youtube"
                print("if --",content_type)
            else:
                loader = UnstructuredURLLoader(
                    urls=[request.url],
                    ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                )
                content_type = "website"
                print("else--",content_type)
            print("loader---",loader.load())
            docs = loader.load()
            print("docs-->",docs)
            if not docs:
                raise HTTPException(status_code=400, detail="No content could be extracted from the URL")
            
            # Create summarization chain
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            summary = chain.run(docs)
            print("summary-->",summary)
            
            return SummarizeResponse(
                summary=summary,
                url=request.url,
                content_type=content_type
            )
            
        except Exception as e:
            print("check-1")
            raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")
            
    except Exception as e:
        print("check-2")
        raise HTTPException(status_code=500, detail=str(e))'''

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the uploaded documents"""
    try:
        if request.session_id not in chat_chains:
            raise HTTPException(
                status_code=404, 
                detail="Session not found. Please upload documents first."
            )
        
        conversational_rag_chain = chat_chains[request.session_id]
        
        response = conversational_rag_chain.invoke(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        return ChatResponse(
            answer=response['answer'],
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a session"""
    has_vectorstore = session_id in vector_stores
    has_chat_chain = session_id in chat_chains
    
    documents_count = 0
    if has_vectorstore:
        # This is an approximation - Chroma doesn't directly expose document count
        documents_count = len(vector_stores[session_id].get()['documents']) if vector_stores[session_id].get() else 0
    
    return {
        "session_id": session_id,
        "has_vectorstore": has_vectorstore,
        "has_chat_chain": has_chat_chain,
        "documents_count": documents_count
    }

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id in session_store:
        messages = session_store[session_id].messages
        return {
            "session_id": session_id,
            "messages": [{"type": msg.type, "content": msg.content} for msg in messages]
        }
    return {"session_id": session_id, "messages": []}

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a session"""
    if session_id in session_store:
        del session_store[session_id]
    if session_id in vector_stores:
        del vector_stores[session_id]
    if session_id in chat_chains:
        del chat_chains[session_id]
    
    return {"message": f"Session {session_id} cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)









# from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# import os
# import tempfile
# import uuid
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# import shutil

# app = FastAPI(title="RAG Q&A API", description="Conversational RAG with PDF uploads and chat history")

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global storage for sessions and vector stores
# session_store = {}
# vector_stores = {}
# chat_chains = {}

# class ChatRequest(BaseModel):
#     message: str
#     session_id: str
#     openai_api_key: str

# class ChatResponse(BaseModel):
#     answer: str
#     session_id: str

# class SessionInfo(BaseModel):
#     session_id: str
#     documents_count: int
#     has_vectorstore: bool

# @app.get("/")
# async def root():
#     return {"message": "RAG Q&A API is running"}

# @app.post("/upload-documents")
# async def upload_documents(
#     files: List[UploadFile] = File(...),
#     session_id: str = Form(...),
#     openai_api_key: str = Form(...)
# ):
#     """Upload PDF documents and create vector store"""
#     try:
#         if not openai_api_key:
#             raise HTTPException(status_code=400, detail="OpenAI API key is required")
        
#         # Validate OpenAI API key by creating embeddings instance
#         try:
#             embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#         except Exception as e:
#             raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
        
#         documents = []
        
#         # Process uploaded files
#         for file in files:
#             if not file.filename.endswith('.pdf'):
#                 raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#                 content = await file.read()
#                 temp_file.write(content)
#                 temp_file_path = temp_file.name
            
#             # Load PDF content
#             loader = PyPDFLoader(temp_file_path)
#             docs = loader.load()
#             documents.extend(docs)
            
#             # Clean up temp file
#             os.unlink(temp_file_path)
        
#         if not documents:
#             raise HTTPException(status_code=400, detail="No documents could be processed")
        
#         # Split documents
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, 
#             chunk_overlap=200
#         )
#         splits = text_splitter.split_documents(documents)
        
#         # Create vector store
#         vectorstore = Chroma.from_documents(
#             documents=splits, 
#             embedding=embeddings,
#             collection_name=f"collection_{session_id}"
#         )
        
#         # Store vector store and create RAG chain
#         vector_stores[session_id] = vectorstore
        
#         # Create LLM
#         llm = ChatOpenAI(
#             openai_api_key=openai_api_key,
#             model_name="gpt-3.5-turbo",
#             temperature=0
#         )
        
#         # Create retriever
#         retriever = vectorstore.as_retriever()
        
#         # Contextualize question prompt
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it if needed and otherwise return it as is."
#         )
        
#         contextualize_q_prompt = ChatPromptTemplate.from_messages([
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ])
        
#         history_aware_retriever = create_history_aware_retriever(
#             llm, retriever, contextualize_q_prompt
#         )
        
#         # Answer prompt
#         system_prompt = (
#             "You are an assistant for question-answering tasks. "
#             "Use the following pieces of retrieved context to answer "
#             "the question. If you don't know the answer, say that you "
#             "don't know. Use three sentences maximum and keep the "
#             "answer concise."
#             "\n\n"
#             "{context}"
#         )
        
#         qa_prompt = ChatPromptTemplate.from_messages([
#             ("system", system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ])
        
#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
#         # Create conversational chain
#         def get_session_history(session: str) -> BaseChatMessageHistory:
#             if session not in session_store:
#                 session_store[session] = ChatMessageHistory()
#             return session_store[session]
        
#         conversational_rag_chain = RunnableWithMessageHistory(
#             rag_chain,
#             get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer"
#         )
        
#         chat_chains[session_id] = conversational_rag_chain
        
#         return {
#             "message": f"Successfully processed {len(files)} documents",
#             "session_id": session_id,
#             "documents_count": len(documents),
#             "chunks_count": len(splits)
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     """Chat with the uploaded documents"""
#     try:
#         if request.session_id not in chat_chains:
#             raise HTTPException(
#                 status_code=404, 
#                 detail="Session not found. Please upload documents first."
#             )
        
#         conversational_rag_chain = chat_chains[request.session_id]
        
#         response = conversational_rag_chain.invoke(
#             {"input": request.message},
#             config={"configurable": {"session_id": request.session_id}}
#         )
        
#         return ChatResponse(
#             answer=response['answer'],
#             session_id=request.session_id
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/sessions/{session_id}")
# async def get_session_info(session_id: str):
#     """Get information about a session"""
#     has_vectorstore = session_id in vector_stores
#     has_chat_chain = session_id in chat_chains
    
#     documents_count = 0
#     if has_vectorstore:
#         # This is an approximation - Chroma doesn't directly expose document count
#         documents_count = len(vector_stores[session_id].get()['documents']) if vector_stores[session_id].get() else 0
    
#     return {
#         "session_id": session_id,
#         "has_vectorstore": has_vectorstore,
#         "has_chat_chain": has_chat_chain,
#         "documents_count": documents_count
#     }

# @app.get("/sessions/{session_id}/history")
# async def get_chat_history(session_id: str):
#     """Get chat history for a session"""
#     if session_id in session_store:
#         messages = session_store[session_id].messages
#         return {
#             "session_id": session_id,
#             "messages": [{"type": msg.type, "content": msg.content} for msg in messages]
#         }
#     return {"session_id": session_id, "messages": []}

# @app.delete("/sessions/{session_id}")
# async def clear_session(session_id: str):
#     """Clear a session"""
#     if session_id in session_store:
#         del session_store[session_id]
#     if session_id in vector_stores:
#         del vector_stores[session_id]
#     if session_id in chat_chains:
#         del chat_chains[session_id]
    
#     return {"message": f"Session {session_id} cleared successfully"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)