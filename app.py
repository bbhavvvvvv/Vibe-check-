from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_llm():
    # It's best practice to set your key in a .env file as GROQ_API_KEY
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found in .env file.")
        
    return ChatGroq(
        temperature=0,
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

def setup_vector_db():
    data_dir = "./data/"
    db_path = "./chroma_db"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir} directory. Place your mental health PDFs here.")

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    if os.path.exists(db_path):
        print("Loading existing vector database...")
        return Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    print("Creating new vector database...")
    loader = DirectoryLoader(data_dir, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("Warning: No PDF documents found in ./data/")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    return vector_db

def setup_qa_chain(vector_db, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )

# Initialization
llm = initialize_llm()
vector_db = setup_vector_db()

# UI Helper for Premium Styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
    --chat-bubble-user: #80cbc4;
    --chat-bubble-bot: #ffffff;
    --text-color: #263238;
}

body {
    font-family: 'Outfit', sans-serif !important;
    background: var(--primary-gradient) !important;
}

.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important;
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    padding: 20px !important;
}

.chatbot-container {
    border-radius: 15px !important;
    overflow: hidden !important;
    border: none !important;
}

.user-msg {
    background-color: var(--chat-bubble-user) !important;
    color: white !important;
}

.bot-msg {
    background-color: var(--chat-bubble-bot) !important;
    border: 1px solid #e0e0e0 !important;
}
"""

# Custom theme based on Shiki
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%)",
    block_background_fill="rgba(255, 255, 255, 0.5)",
    block_border_width="0px",
)

# Gradio Interface Logic
def chat_fn(message, history):
    if vector_db is None:
        return "Please add mental health PDF documents to the './data' folder and restart the app."
    
    if not hasattr(chat_fn, "qa_chain"):
        chat_fn.qa_chain = setup_qa_chain(vector_db, llm)
    
    # Simple check for crisis keywords to trigger resources
    crisis_keywords = ["suicide", "hurt myself", "kill myself", "end it all"]
    triggered_crisis = any(word in message.lower() for word in crisis_keywords)
    
    response = chat_fn.qa_chain({"question": message})
    bot_message = response['answer']
    
    if triggered_crisis:
        bot_message += "\n\n--- \n**RESOURCES**: I'm hearing that you're in a lot of pain. Please reach out for help: \n- **National Suicide Prevention Lifeline**: 988 \n- **Crisis Text Line**: Text HOME to 741741"
        
    return bot_message

# UI Design with Blocks
with gr.Blocks(theme=theme, css=custom_css) as interface:
    with gr.Row():
        with gr.Column(scale=8):
            gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <h1 style="color: #004d40; font-size: 2.5rem; font-weight: 600; margin-bottom: 0;">✨ VibeCheck</h1>
                    <p style="color: #00796b; font-size: 1.1rem;">Your Real-Talk Mental Health Bestie</p>
                </div>
            """)
    
    with gr.Row():
        with gr.Column(scale=9):
            chat = gr.ChatInterface(
                fn=chat_fn,
                type="messages",
                examples=["Vibe check: I'm feeling low.", "How do I deal with academic burnout?", "Tips for social anxiety?"],
            )
        
        with gr.Column(scale=3):
            gr.Markdown("### 📱 The Inner Circle")
            gr.Markdown("""
            - **Main Character**: [@sarky.ayush](https://www.instagram.com/sarky.ayush/)
            - **Vibe Check**: DMs are open (sometimes)
            - **Grounding**: Talk to a real human
            
            *No cap, if things are actually hitting the fan, please reach out to real pros at 14416. We need you here.*
            """)
            
            gr.Markdown("### The Tea  💅")
            gr.Markdown("VibeCheck isn't just a bot—it's pulling facts from actual mental health docs so you get the real deal, not just generic toxic positivity.")

    gr.HTML("""
        <div style="text-align: center; font-size: 0.8rem; color: #546e7a; margin-top: 20px;">
            Built with ❤️ for your well-being.
        </div>
    """)

if __name__ == "__main__":
    interface.launch()
