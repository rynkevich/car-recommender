from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROMPT_TEMPLATE = '''
You are a car recommender chat assistant of a car retailing company.
Act as an experienced sales person, be friendly and ensure a smooth user experience.
Engage with user, ask relevant questions to comprehend their car preferences such as
budget, car type, fuel type, brand, or specific features.

---

User's message: {message}

---

{instruction}
'''

PROMPT_RECON = '''
If you are finished with gathering preferences, start your response with "!" and summarize the user's preferences in one sentence.
Else, continue interaction.
'''


def get_llm_response(user_query: str, instruction: str) -> str:
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(message=user_query, instruction=instruction)
    model = ChatOpenAI()
    return model.invoke(prompt).content


def reconnoiter(user_query: str) -> str:
    return get_llm_response(user_query, PROMPT_RECON)
