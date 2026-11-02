
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from memory import get_all_memories


# Extract clean text from Gemini response
def get_response_text(response):

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts).strip()

    return str(content)


# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check your .env file."
    )


# Check Chroma database
if not os.path.isdir("chroma_db"):
    raise FileNotFoundError(
        "chroma_db folder not found. Run create_database.py first."
    )


# 1. Load Hugging Face embeddings
print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Load Chroma database
print("Loading PDF database...")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# 4. Connect Gemini
print("Connecting to Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    max_tokens=500
)


# 5. Conversation memory
chat_history = []


# Display saved memories
print("\nPreviously Saved Memories:")

user_memories = get_all_memories()

for key, value in user_memories:
    print(f"{key}: {value}")


print("\n" + "=" * 45)
print("       PDF RAG CHATBOT READY!")
print("=" * 45)
print("Ask questions about your PDF.")
print("Type 'exit' to stop.")


# 6. Chat loop
try:

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        try:

            # 7. Understand follow-up questions
            if chat_history:

                history_text = ""

                for chat in chat_history[-4:]:
                    history_text += (
                        f"User: {chat['question']}\n"
                        f"AI: {chat['answer']}\n"
                    )

                search_prompt = f"""
Convert the latest question into a standalone search query
using the conversation history.

Do not answer the question.
Return only the search query.

Conversation:
{history_text}

Latest question:
{question}

Search query:
"""

                search_response = llm.invoke(search_prompt)

                search_query = get_response_text(
                    search_response
                )

                if not search_query:
                    search_query = question

            else:
                search_query = question


            # 8. Retrieve relevant PDF chunks
            docs = retriever.invoke(search_query)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            if not context:
                context = "No relevant PDF information retrieved."


            # 9. Load permanent user memory
            memories = get_all_memories()

            memory_context = "\n".join(
                f"{key}: {value}"
                for key, value in memories
            )

            if not memory_context:
                memory_context = "No personal information saved yet."


            # 10. Prepare conversation history
            history = ""

            for chat in chat_history[-4:]:
                history += (
                    f"User: {chat['question']}\n"
                    f"AI: {chat['answer']}\n"
                )

            if not history:
                history = "No previous conversation."


            # 11. Create final prompt
            prompt = f"""
You are a helpful PDF-based AI assistant.

Instructions:

1. Use permanent user memory to answer personal questions.
2. Use PDF context to answer questions about the PDF.
3. Use conversation history to understand follow-up questions.
4. Never invent personal information or PDF facts.
5. If personal information is available in memory,
   use it instead of searching the PDF.
6. If information is unavailable in both memory and PDF,
   clearly say that it was not found.
7. Give clear and concise answers.

Permanent User Memory:
{memory_context}

Conversation History:
{history}

PDF Context:
{context}

Current Question:
{question}

Answer:
"""


            # 12. Generate answer
            response = llm.invoke(prompt)

            answer = get_response_text(response)

            if not answer:
                answer = "I could not generate a response. Please try again."


            # 13. Display answer
            print("\nAI:", answer)


            # 14. Store conversation
            chat_history.append({
                "question": question,
                "answer": answer
            })


        except Exception as e:

            print("\nSomething went wrong:", str(e))
            print("Please check your API connection and try again.")


except (KeyboardInterrupt, EOFError):
    print("\nChat stopped by user.")


print("\nChat ended! Thank you.")
