import timeit

import aiohttp
import asyncio


with open("message.txt", 'r') as f:
    data = eval(f.read())


async def send_request(sentence: str):
    url = "http://0.0.0.0:8080/process"

    async with aiohttp.ClientSession() as session:

        try:
            async with session.post(url, data=sentence) as response:
                success = response.status == 200
                return success
        except Exception as e:
            return False


async def main():
    start_time = asyncio.get_event_loop().time()
    concurrent_requests = len(data)
    tasks = [send_request(sentence=sentence[:50]) for sentence in data]
    results = await asyncio.gather(*tasks)
    total_request_time = asyncio.get_event_loop().time() - start_time
    successful_requests = sum(results)

    print(f"Успешных запросов: {successful_requests}")
    print(f"Суммарное время выполнения всех запросов: {total_request_time:.4f} секунд")
    print(f"rps: {(concurrent_requests / total_request_time):.4f}")


loop = asyncio.get_event_loop()

if __name__ == "__main__":
    number = 10
    t = timeit.timeit('loop.run_until_complete(main())', globals=globals(), number=number)
    print(f"Mean rps: {(100 * number / t):.4f} секунд")

