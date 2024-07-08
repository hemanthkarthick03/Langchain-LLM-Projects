import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

if __name__ == "__main__":
    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key="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",
        model_name='llama3-8b-8192'
    )
    st.header("Entering the Chatbot")
    st.write("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

    system_prompt = 'You are a friendly conversational chatbot'
    conversational_memory_length = 5 # number of previous messages the chatbot will remember during the conversation

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    if "conversation" not in st.session_state:
        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),  # Persistent system prompt.
                MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history.
                HumanMessagePromptTemplate.from_template("{human_input}"),  # Template for user's current input.
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        st.session_state.conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )

    user_question = st.text_input("Ask a question:", key="user_input")
    if st.button("Start Chat", key="start_chat"):
        if user_question:
            response = st.session_state.conversation.predict(human_input=user_question)
            st.markdown(f"Chatbot: {response}")
