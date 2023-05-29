from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import Response
from core.image_processor import ImageProcessor, encode_cv2, DetectorFactory
from core.image_processor.shape_detector import process
from core.image_processor.figure_detector import FigureDetector
from fastapi.middleware.cors import CORSMiddleware
from core.model.model import BaseModel
from redis_model import create_hashed_image, obtain_hashed_image
from serializers import DeserializedImage
from core.image_processor.CNTR_filler import fill_contour

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


@app.post(
    path='/api/image/shape_detector'
)
async def shape_detect(image: UploadFile = File()):
    image = await image.read()
    detector = DetectorFactory().detector
    processed = process(detector, image)
    bytes_io = encode_cv2(processed)
    return Response(content=bytes_io.read(), media_type='image/png', status_code=200)


@app.post(
    path='/api/image/figure_detector'
)
async def detect_contours(image: UploadFile = File()):
    image = await image.read()
    detector = FigureDetector(image, BaseModel)
    detector.color_check()
    processed = detector.input_image.copy()
    for color_key in detector.colors.keys():
        if detector.contours[color_key] is None:
            continue
        results_and_caption_args = list(
            detector.find_figure(
                detector.contours[color_key],
                detector.masks[color_key]
            )
        )
        for res, cargs in results_and_caption_args:
            detector.add_caption(
                processed, color_key, res, *cargs
            )

    bytes_io = encode_cv2(processed)
    resp = Response(content=bytes_io.read(), media_type='image/png', status_code=200)
    cookie_key = await create_hashed_image(image, detector.contours.values())
    resp.set_cookie(key='IMG', value=cookie_key, max_age=30 * 60)
    resp.headers.append(key='IMG', value=cookie_key)
    return resp


@app.get(
    path='/api/image/obtain_contours'
)
async def obtain_contour(image_data: DeserializedImage = Depends(obtain_hashed_image)):
    cnts = len(image_data.contours)
    return {'cnts_sum': cnts}


@app.post(path='/api/image/fill_image')
async def remove_contour(
        cntr: int,
        image_data: DeserializedImage = Depends(obtain_hashed_image)
):
    image = image_data.image
    if len(image_data.contours) - 1 < cntr:
        raise HTTPException(detail='cntr is out of bound', status_code=400)
    contour = image_data.contours.pop(cntr)
    cv_image = fill_contour(image, contour)
    resp = Response(content=encode_cv2(cv_image).read(), media_type='image/png', status_code=200)
    return resp
