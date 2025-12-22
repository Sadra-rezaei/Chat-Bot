# Chat Bot
#### This chatbot answers questions related to Computer Engineering using a custom Q&A system.
This is a Q&A-based chatbot, **not** a Large Language Model (LLM).


---
## 🚀 Features

- Sentence embedding
- Question answering
- Customizable Q&A database

---
## 🛠️ Tech Stack

- **Language:** Python
- **Frameworks / Libraries:** Streamlit
	https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
- **Model / Tool:** all-MiniLM-L6-v2 (Sentence Transformer) 
	https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

---
## ▶️ Usage

Run the chatbot using Streamlit:

```bash
# example
streamlit run ChatBot.py
```

---
## 📖 Example

![Example Output](assets/Example.png)

---
## 🧠 How It Works

1. When you ask a question, the sentence is embedded first.
    
2. The `answer()` function compares your question with all embedded questions in the question bank and finds the most similar one.
    
3. The similarity score ranges from **0** (least similar) to **1** (most similar).  
    If the highest similarity score is below **0.3**, the function returns **"I don't know"**

---
## 🧪 Tests

This project includes two test files:

1. `EmbeddingTesting.py` 
	- Compares two sentences and shows how similar they are (from `0` to `1`).
2. `AnsweringTesting.py`
	- Similar to `ChatBot.py`, but without the Streamlit GUI.
	- Useful if you encounter issues with Streamlit

---
## ❓Is it just for Computer Engineering?

No! you can customize your own question bank.
just use this pattern for `questionBank.json` :

```json
{
  "questions": [
    {
      "id": "optional",
      "question": "your question",
      "answer": "the answer returned by the chatbot",
      "embedding": []
    }
  ]
}

```

---
## 👤 Author

Sadra Mir Mohammad Rezaei
Email: sadrarezaei4@gmail.com