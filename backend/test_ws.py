import asyncio, websockets

async def test():
    try:
        async with asyncio.timeout(5):
            async with websockets.connect('ws://localhost:5000/ws/test') as ws:
                msg = await ws.recv()
                print('收到:', str(msg)[:100])
    except asyncio.TimeoutError:
        print('超时')
    except Exception as e:
        print('错误:', type(e).__name__, str(e)[:200])

asyncio.run(test())
