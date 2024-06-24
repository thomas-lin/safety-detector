import asyncio
import json
import os
from threading import Event, Thread
from typing import Dict

import nats
from dotenv import load_dotenv

from .predict import predict


async def run():
    predictThreads: Dict[str, (Thread, Event)] = {}
    nc = await nats.connect(f"nats://{os.getenv('NATS_SERVER')}")
    sub = await nc.subscribe("app.*.*")
    try:
        async for msg in sub.messages:
            print(f"Received a message on '{msg.subject} {msg.reply}': {msg.data.decode()}")
            data = json.loads(msg.data.decode())
            if "start" in msg.subject:
                if data["AC_NO"] in predictThreads:
                    continue
                stop_event = Event()
                thread = Thread(
                    target=predict, args=(data["AC_NO"], data["VIDEOURL"], stop_event), daemon=True
                )
                thread.start()
                predictThreads[data["AC_NO"]] = (thread, stop_event)
                print(f'thread_{data["AC_NO"]} start.')

            if "stop" in msg.subject:
                if data["AC_NO"] in predictThreads:
                    (thread, stop_event) = predictThreads[data["AC_NO"]]
                    stop_event.set()
                    thread.join()
                    del predictThreads[data["AC_NO"]]
                    print(f'thread_{data["AC_NO"]} stop.')

    except Exception as e:
        print(f"Error:{e}")
        pass

    # Terminate connection to NATS.
    await sub.unsubscribe()
    await nc.drain()


def main():
    load_dotenv()
    asyncio.run(run())
