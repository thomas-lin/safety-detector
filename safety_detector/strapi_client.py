import json
import os
from typing import TypedDict

from urllib3 import PoolManager, Timeout


def singleton(class_):
    _instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in _instances:
            _instances[class_] = class_(*args, **kwargs)
        return _instances[class_]

    return get_instance


class SafetyEvent(TypedDict):
    AC_NO: str
    Trace_id: str
    className: str
    conf: float


@singleton
class StrapiClient:
    def __init__(self):
        self._id = id(self)
        self.pool = PoolManager(timeout=Timeout(connect=10))

    def get_id(self):
        return self._id

    def createSafetyEvent(self, event: SafetyEvent):
        strapi_api = os.getenv('STRAPI_APP_SAFETY_COUNTER_API')
        body = json.dumps(event).encode("utf-8")
        self.pool.request(
            "POST",
            f"{strapi_api}",
            headers={"Content-Type": "application/json"},
            body=body,
        )

        pass
