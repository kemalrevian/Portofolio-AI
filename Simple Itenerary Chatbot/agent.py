import os
import openai
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# =====================
# 1. ENV & CONFIG
# =====================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# =====================
# 2. MODEL
# =====================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
    )

# =====================
# 3. TOOLS
# =====================
@tool
def get_weather_data(location: List[float]) -> dict:
    """
    Get weather forecast data from Open-Meteo API.
    Args:
        location (List[float]): [latitude, longitude]
    Returns:
        dict: weather forecast data (hourly)
    """
    API_URL = "https://api.open-meteo.com/v1/forecast"
    PARAMS = [
        "temperature_2m",
        "weather_code",
        "surface_pressure",
        "precipitation_probability"
    ]

    response = requests.get(
        API_URL,
        params={
            "latitude": location[0],
            "longitude": location[1],
            "hourly": ",".join(PARAMS),
            "forecast_days": 3,
        },
    )

    return response.json()

tools = [get_weather_data]


# =====================
# 4. AGENT
# =====================
def iternary_agent():
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt= f"""    
    Kamu adalah seorang itinerary planner bot untuk website YukJalan.ai.\
    Personamu adalah seorang bot dengan bahasa gaul ala gen-Z jaksel.\
    Tugasmu adalah membuat itinerary satu hari yang cocok berdasarkan kondisi cuaca kota yang akan dituju.\
    Pertimbangkan pula preferensi tambahan dari user apabila ada.\
    Pastikan itinerary tersebut lengkap dengan aktivitas turis, kuliner, dan pertunjukkan atau wahana.\
    Sebelum memikirkan jawaban, kamu harus cek informasi cuaca menggunakan tool get_weather_data terlebih dahulu.\
    Gunakan data weather code di bawah untuk mengetahui kondisi cuaca yang didapat dari tool.\
    Jangan langsung memberikan jawaban sebelum melakukan action.\
    Pada Final Answer, berikan itinerary selama durasi liburan user apabila ada, jika tidak maka berikan sehari saja.\
    Tanyakan ke user apabila ada informasi tambahan yang diperlukan untuk melakukan eksekusi aksi.\
    Apabila user bertanya hal yang tidak sesuai konteks, maka tegur user dengan bahasa sopan.\
    Gunakan history untuk melakukan penyesuaian berdasarkan preferensi user.\

    **Weather Code**
    Informasi weather code adalah sebagai berikut:
    - Apabila weather code 0-1 = cerah
    - Apabila weather code 1-2 = mendung/berawan
    - Selain dari itu = cuaca buruk

    Gunakan pola Reasoning-Action berikut dalam menjawab
    Thought:
    Action:
    Action Input:
    Observation:
    ... ulangi jika perlu ...
    Final Answer:
    """
    )
    return agent
