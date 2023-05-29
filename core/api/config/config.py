import os
from aredis_om import get_redis_connection

# api
HOST: str = os.getenv('UVI_HOST')
UVI_PORT = int(os.getenv('UVI_PORT', default=8081))
# redis
REDIS_HOST: str = os.getenv('REDIS_HOST', default='redis')
aredis = get_redis_connection(
   url=f'redis://{REDIS_HOST}:6379/0',
   decode_responses=True
)
print(HOST)