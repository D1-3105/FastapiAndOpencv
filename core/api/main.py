import uvicorn
import os
import pathlib
from config.config import HOST, UVI_PORT
from aredis_om import Migrator
import controllers

base_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(base_dir)


@controllers.app.on_event('startup')
async def startup():
    await Migrator().run()


if __name__ == '__main__':
    uvicorn.run(controllers.app, host=HOST, port=UVI_PORT)
