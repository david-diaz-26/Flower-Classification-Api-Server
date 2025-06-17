from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess
import logging

# Logging
logging.basicConfig(
    filename="plant_llm.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("plant_llm")

# Initiate Fast Api
app = FastAPI()

class FlowerRequest(BaseModel):
    flower_name: str

@app.post("/flower-info")
async def flower_info(request: FlowerRequest):
    flower_name = request.flower_name.strip()
    
    full_prompt = (
        f"Write the name of the flower: {flower_name}\n\n"
        "Then write 3 sentences describing the history and discovery of this flower.\n\n"
        "Then list 3 bullet points with important facts about it.\n\n"
        "Format it exactly like this:\n"
        "Name: <flower name>\n\n"
        "History:\n"
        "1. sentence\n"
        "2. sentence\n"
        "3. sentence\n\n"
        "Important facts:\n"
        "- fact 1\n"
        "- fact 2\n"
        "- fact 3"
    )
    
    logger.info(f"Sending prompt to LLM:\n{full_prompt}")

    try:
        result = subprocess.run(
            ["/usr/local/bin/ollama", "run", "llama3.2", full_prompt],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"LLM ERROR: {e.stderr}")
        return {"error": e.stderr}

    raw_output = result.stdout.strip()
    logger.info(f"RAW LLM OUTPUT:\n{raw_output}")

    return {"output": raw_output}