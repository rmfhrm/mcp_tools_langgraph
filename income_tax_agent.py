# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# %% [markdown]
# ### 처음 데이터 생성 시에만 확인하기 //// 데이터 로딩

# %%
from langchain_chroma import Chroma

## 데이터 로딩 
vector_store = Chroma(
    collection_name='chroma-tax',
    embedding_function=embedding,
    persist_directory='./chroma-tax'
)

retriever = vector_store.as_retriever(search_kwargs={'k':3})

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

# %%
def retrieve(state: AgentState) -> AgentState:
    """ 
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.
    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state
    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.        
    """

    query = state['query']
    docs = retriever.invoke(query)
    return {'context': docs}

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')



# %% [markdown]
# ### 강사님 버전

# %%
from langchain import hub

generate_prompt = hub.pull('rlm/rag-prompt')

def generate(state: AgentState) -> AgentState:
    """ 
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state
    Returns:
        AgentState: 생성된 응답을 포함하는 state를 반환합니다.        
    """
    context = state['context']
    query = state['query']

    rag_chain = generate_prompt | llm

    response = rag_chain.invoke({'question': query, 'context': context})

    return {'answer': response}

# %%
from langchain import hub
from typing import Literal

doc_releveance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_doc_relevence(state: AgentState) -> Literal['relevant', 'irrelevant']:
    """ 
    주어진 state를 기반으로 문서의 관련성을 판단합니다.
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state
    Returns:
        Literal['relevant', 'irrelevant']: 문서가 관련성이 높으면 'relevant', 그렇치 않으면 'irrelevant' 를 반환합니다.       
    """

    query = state['query']
    context = state['context']

    doc_relevance_chain = doc_releveance_prompt | llm

    response = doc_relevance_chain.invoke({'question': query, 'documents': context})

    if response['Score'] == 1:
        return 'relevant'
    
    return 'irrelevant'

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사랑과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리 사전을 참고해서 사용자의 질문을 변경해 주세요.
사전: {dictionary}
질문: {{query}}
""")

def rewrite(state=AgentState) -> AgentState:
    """
    사용자의 질문을 사전에 고려하여 변경합니다.
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state
    Returns:
        AgentState: 변경된 질문을 포함하는 state를 반환합니다.   
    """
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    response = rewrite_chain.invoke({'query': query})

    return {'query': response}

# %%
from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})

    print('거짓말: ', response)

    return response

# %%
# %%


# %%
from langchain import hub

helpfulness_prompt = hub.pull('langchain-ai/rag-answer-helpfulness')

def check_helpfulness_grader(state: AgentState) -> Literal['helpful', 'unhelpful']:
    """
    사용자의 질문에 기반하여 생성된 답변의 유용성을 평가합니다.
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state
    Returns:
        Literal['helpful', 'unhelpful']: 답변이 유용하다고 판단되면 'helpful', 그렇지 않다면 'unhelpful'을 반환합니다.
    """
    query = state['query']
    answer = state['answer']

    helpfulness_chain = helpfulness_prompt | llm
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})

    if response['Score'] == 1:
        print('유용성: helpful')
        return 'helpful'
    
    print('유용성: unhelpful')
    return 'unhelpful'

# %%
def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수입니다.
    """
    return state

# %%

# %% [markdown]
# ### 혼자 해보기 그래프

# %%
builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node('rewrite', rewrite)
builder.add_node('check_helpfulness',check_helpfulness)

# %%
from langgraph.graph import START, END

builder.add_edge(START, "retrieve")   
builder.add_conditional_edges(
    "retrieve",
    check_doc_relevence,
    {
        "relevant": "generate",     # 문서 관련 있으면 → generate
        "irrelevant": END     # 문서 관련 없으면 → rewrite → END
    }
)
builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)
builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
builder.add_edge("rewrite", "retrieve")   # rewrite 후 다시 검색

# %%
graph = builder.compile()





