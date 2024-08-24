from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

RECOMMENDER_PROMPT = '''
# BASE
You are a car recommender chat assistant of a car retailing company.
Act as an experienced sales person, be friendly and concise.

# CONVERSATION FLOW
As a first message, greet the user and tell him what you can do to help him.
Engage with user, ask relevant questions to comprehend their car preferences such as
budget, car type, fuel type, brand, or specific features.

When user is ready to make a choice and does not need further assistance from you,
ask him for a way to contact him (email, phone number or else), thank him and say that he is going to be contacted
by a consultant soon.

# CONSTRAINTS
If user says something irrelevant to car recommendation, apologize and let him know
that you are limited only to car choice assistance.

Only say that a car is in stock if it is listed below.

# SEARCH
Below are the cars that are in stock and might fit the user's preferences:

{context}
'''

CONTEXTUALIZE_PROMPT = '''
Given a chat history and the latest user message, summarize what type of a car the user is looking for.
Just reformulate the message if needed and otherwise return it as is.
'''

GREETING_MESSAGE = ("Hello! Welcome to our car retailing company. "
                    "I'm here to assist you in finding the perfect car that suits your needs. "
                    "Could you please tell me about your preferences such as budget, car type, "
                    "fuel type, brand, or any specific features you're looking for in a car?")


class AssistantModel:
    def __init__(
            self,
            vectorstore: VectorStore,
            max_selection: int,
            model_name: str,
            model_temperature: float,
    ):
        self.chat_history = []
        self.chain = self.build_chain(vectorstore, max_selection, model_name, model_temperature, )

    def get_start_message(self) -> str:
        self.chat_history.append(SystemMessage(GREETING_MESSAGE))
        return GREETING_MESSAGE

    def get_response(self, user_input: str) -> str:
        response = self.chain.invoke({'input': user_input, 'chat_history': self.chat_history})
        self.chat_history.extend([HumanMessage(user_input), SystemMessage(response['answer'])])
        return response['answer']

    @staticmethod
    def build_chain(vectorstore: VectorStore, max_selection: int, model_name: str,
                    model_temperature: float) -> Runnable:
        retriever = vectorstore.as_retriever(top_n=max_selection)
        model = ChatOpenAI(model=model_name, temperature=model_temperature)

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', CONTEXTUALIZE_PROMPT),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ],
        )
        history_aware_retriever = create_history_aware_retriever(
            model, retriever, contextualize_prompt,
        )

        recommender_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', RECOMMENDER_PROMPT),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ]
        )
        recommender_chain = create_stuff_documents_chain(model, recommender_prompt)

        return create_retrieval_chain(history_aware_retriever, recommender_chain)
