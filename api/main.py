import controllers
import uvicorn

if __name__ == '__main__':
    uvicorn.run(controllers.app, host='localhost', port=8081)
