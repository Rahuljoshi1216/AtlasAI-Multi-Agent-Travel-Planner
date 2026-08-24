import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    
    
# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    
    
# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }    
    
        
# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, realistic, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="""
You are an expert travel itinerary planner.

Your task is to create a practical, realistic, and well-structured travel itinerary using the user's request along with the provided flight and hotel results.

Follow these rules:
- Base the itinerary strictly on the provided user query, flight results, and hotel results.
- Do not invent flight details, hotel details, prices, timings, or locations that are not provided.
- Consider the user's budget and optimize the trip for value and convenience.
- Organize the itinerary chronologically by day.
- Include travel dates, flight information, hotel stay, major activities, sightseeing, meals, and local transportation where the available information supports them.
- Keep the schedule realistic by considering travel time, check-in/check-out times, and reasonable activity durations.
- Avoid overloading a single day with too many activities.
- Clearly distinguish confirmed information from reasonable planning suggestions.
- If important information is missing, make a sensible assumption only when necessary and clearly mention it.
- Prioritize practicality, budget-awareness, convenience, and a good travel experience.
- Return only the final itinerary. Do not explain your reasoning or how you generated it.
"""),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    
    
# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel planning response for the user.

User Request:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Prepared Itinerary:
{state['itinerary']}

Create a polished, easy-to-read final response using the information above.

Structure the response using these sections:

## 1. Trip Summary
Briefly summarize the destination, travel dates, trip duration, travelers, and the overall travel plan based on the available information.

## 2. Flight Information
Present the available flight options clearly, including airline, departure/arrival locations, timings, duration, stops, and price if available.

If ticket pricing is not available from the flight data, explicitly state:
"Live ticket pricing is currently unavailable from the flight source."

Do not invent or estimate flight prices.

## 3. Hotel Suggestions
Present the available hotel options with useful information such as hotel name, location, rating, price, room information, and important amenities when available.

Do not invent hotel information or prices.

## 4. Day-by-Day Itinerary
Present the prepared itinerary in a chronological and practical format.

For each day, include:
- Morning
- Afternoon
- Evening

Only include activities supported by the itinerary or user request. Keep the schedule realistic and avoid packing too many activities into one day.

## 5. Estimated Budget
Provide a simple budget breakdown using only available pricing information.

Separate:
- Flights
- Hotel
- Activities
- Local transportation
- Food
- Other expenses

If a cost is unavailable, clearly mark it as "Not available" instead of inventing a value.

If enough information is available, provide a total estimated cost and clearly state what is included.

## 6. Final Recommendations
Give a few practical recommendations related to:
- Best flight/hotel option based on value and convenience
- Booking considerations
- Transportation
- Time management
- Important travel considerations

Important rules:
- Use the provided data as the source of truth.
- Never fabricate flights, hotels, prices, timings, ratings, bookings, or availability.
- Do not claim that anything has been booked or confirmed unless the provided data explicitly says so.
- Clearly distinguish between actual search results and planning suggestions.
- If information is missing, say that it is unavailable.
- Do not repeat the same information unnecessarily.
- Keep the response concise but sufficiently detailed for real travel planning.
- Use tables, bullet points, and headings where they improve readability.
- Make the final response professional, friendly, and easy to scan.
- Do not explain your reasoning or mention internal agents, state, prompts, or system processing.
"""

    response = llm.invoke([
        SystemMessage(content="""
You are the final response specialist for an AI travel planning and booking assistant.

Your job is to transform the user's request, flight results, hotel results, and prepared itinerary into one polished final travel response.

Follow these principles:

1. Accuracy First
Use the supplied data as the source of truth. Never fabricate travel information, prices, availability, ratings, timings, or booking confirmations.

2. Preserve Information
Do not unnecessarily remove useful details from the flight, hotel, or itinerary results.

3. Handle Missing Data
When information is unavailable, clearly say "Not available" or explain that the live source did not provide it. Never guess.

4. Practical Planning
Make the response useful for someone actually planning the trip. Highlight important timings, costs, locations, transportation considerations, and potential gaps.

5. Budget Awareness
Clearly separate known costs from unavailable costs. Never present an invented estimate as an actual price.

6. Readability
Use clear headings, bullet points, tables where appropriate, and chronological organization. Avoid large walls of text.

7. Consistency
Ensure that the final response is consistent with the prepared itinerary and the supplied flight/hotel results. Do not introduce conflicting information.

8. Booking Safety
Do not state or imply that a flight, hotel, activity, or ticket has been booked or confirmed unless the input explicitly confirms it.

9. User Focus
Answer the user's travel request directly. Do not discuss internal processing, agents, prompts, tools, APIs, or reasoning.

Return only the final user-facing travel response.
"""),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    
    
# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)



# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }