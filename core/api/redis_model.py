from aredis_om import HashModel, Field
from fastapi.requests import Request
import base64
from config.config import aredis
from uuid import uuid4
from serializers import DeserializedImage
import _pickle as cPickle


class HashedImage(HashModel):
    cookie: str = Field(primary_key=True)
    image: bytes = Field()
    contours: bytes

    class Meta:
        database = aredis

    @property
    def deserialized_image(self):
        return base64.b64decode(self.image)

    @property
    def deserialized_contours(self):
        enc_contours = base64.b64decode(self.contours)
        return cPickle.loads(enc_contours)


async def create_hashed_image(image: bytes, contours, is_flat=False):
    cookie = uuid4().hex
    data = []
    if not is_flat:
        for color_array in contours:
            for cnts in color_array:
                data.append(cnts)
        img = HashedImage(
            cookie=cookie,
            image=base64.b64encode(image),
            contours=base64.b64encode(cPickle.dumps(data))
        )
    else:
        img = HashedImage(
            cookie=cookie,
            image=base64.b64encode(image),
            contours=base64.b64encode(cPickle.dumps(contours))
        )
    await img.save()
    await img.expire(30 * 60)
    return cookie


async def obtain_hashed_image(request: Request):
    cookie = request.cookies.get('IMG') or request.headers.get('IMG')
    image = await HashedImage.get(cookie)
    image_data = {
        'image': image.deserialized_image,
        'contours': image.deserialized_contours
    }
    return DeserializedImage(**image_data)
