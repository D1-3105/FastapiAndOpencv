from pydantic import BaseModel, validator
import base64


class DeserializedImage(BaseModel):
    image: bytes
    contours: list

    @validator('image', pre=True)
    def deserialize_image(cls, value: bytes | str):
        if isinstance(value, str):
            return base64.b64decode(value)
        return value
