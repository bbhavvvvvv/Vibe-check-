
# VibeCheck - Mental Health Bestie ✨

VibeCheck is your real-talk, RAG-based (Retrieval-Augmented Generation) mental health assistant built with **LangChain**, **Groq (Llama 3)**, and **ChromaDB**. It is designed to provide evidence-based support by retrieval-searching through professional mental health documentation (PDFs).

![Chatbot Preview](https://via.placeholder.com/800x400.png?text=VibeCheck+Interface+Preview) <!-- Replace with actual screenshot later if available -->

## ✨ Features
- **RAG Architecture**: Answers questions based on private PDF documents provided in the `data/` folder.
- **Conversational Memory**: Remembers prior parts of the conversation for a natural chat experience.
- **Crisis Detection**: Automatically provides emergency resources if high-distress keywords are detected.
- **Premium UI**: A calming, glassmorphism-inspired interface built with Gradio.

## 🛠️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "health chatbot"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```text
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Add Knowledge Base**:
   Place any mental health related PDFs in the `data/` folder.

5. **Run the App**:
   ```bash
   python app.py
   ```

## 📂 Project Structure
- `app.py`: Main application script (Gradio UI + LangChain logic).
- `data/`: Folder for source PDF documents.
- `chroma_db/`: Vector database storage (generated automatically).
- `requirements.txt`: Python dependencies.

## ⚠️ Disclaimer
This chatbot is for educational and supportive purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
