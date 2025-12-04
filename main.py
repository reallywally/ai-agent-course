import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from env import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class AgentState(TypedDict):
    user_query: str
    messages: list


def create_workflow():
    """
    Start -> analyze_query -> generate_response -> End
    """
    # 1. LLM
    model = ChatOpenAI(model="gpt-4o-mini")

    # 2. 노드 정리
    def analyze_query_node(state: AgentState) -> AgentState:
        user_query = state["user_query"]

        system_promt = SystemMessage(
            """
        당신은 전문 AI 입니다.
        """
        )

        return {
            "messages": [system_promt, HumanMessage(content=user_query)],
            "user_query": user_query,
        }

    def generate_response_node(state: AgentState) -> AgentState:

        messages = state["messages"]
        response = model.invoke(messages)

        return {"messages": [response], "user_query": state["user_query"]}

    # 3. 그래프 구성
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_query_node", analyze_query_node)
    workflow.add_node("generate_response_node", generate_response_node)

    workflow.add_edge(START, "analyze_query_node")
    workflow.add_edge("analyze_query_node", "generate_response_node")
    workflow.add_edge("generate_response_node", END)

    return workflow.compile()


class ChatBot:

    def __init__(self):
        self.workflow = create_workflow()

    def process_message(self, user_message: str) -> str:
        initial_state: AgentState = {"messages": [], "user_query": user_message}

        result = self.workflow.invoke(initial_state)

        messages = result["messages"]

        print(messages)
        ai_message = messages[0].content

        return ai_message


async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    if update.message is None or update.message.text is None:
        return

    user_message = update.message.text

    result = ChatBot().process_message(user_message)

    await update.message.reply_text(result)


app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT, handler))

app.run_polling()
