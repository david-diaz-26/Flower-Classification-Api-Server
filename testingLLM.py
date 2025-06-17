import asyncio
from main import flower_info, FlowerRequest

async def test_flower_info():
    test_flower = FlowerRequest(flower_name="Rose") # Test flower, adjust 
    response = await flower_info(test_flower)
    print(response)

if __name__ == "__main__":
    asyncio.run(test_flower_info())



# HOW TO
# In terminal, and in the directory, run:  python testingLLM.py
