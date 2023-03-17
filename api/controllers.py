from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from image_processor import ImageProcessor, encode_cv2
from fastapi.middleware.cors import CORSMiddleware
from cv2 import imshow, waitKey

app = FastAPI(debug=True)

origins = {
    "http://localhost",
    "http://localhost:3000",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    path='/api/image/contours'
)
async def make_contours(image: UploadFile = File()):
    image = await image.read()
    proc = ImageProcessor(image)
    cnts, img = proc.make_contours()
    proc.img = img
    proc.put_text(f'1042 Konin Oleg Valerievich Found {len(cnts)} contours!')
    proc.put_sqrt()
    proc.put_dot()
    res = proc.img
    bytes_io = encode_cv2(res)
    return Response(content=bytes_io.read(), media_type="image/png", status_code=200)
