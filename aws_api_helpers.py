import pydantic
import boto3 #type: ignore
from PIL import Image
import io
import hashlib

BUCKET_NAME = 'translango'
DEFAULT_EXPIRATION_TIME_FOR_PRESIGNED_URLS = 1 * 60 # 1 minute (or 60 seconds)
IMG_FORMAT = 'jpeg'

s3_resource = boto3.resource('s3') #type: ignore
translango_bucket = s3_resource.Bucket(BUCKET_NAME) #type: ignore

class S3Bucket(pydantic.BaseModel):
    name: str

    class Config:
        orm_mode = True

class S3Object(pydantic.BaseModel):
    bucket_name: str
    key: str = pydantic.Field(description="name of the object uploaded in s3")

    class Config:
        orm_mode = True

def upload_PIL_Image_to_s3(image: Image.Image) -> S3Object:
    name = hashlib.sha512(image.tobytes()) #type: ignore
    # print(f"hash of the image: {name.name=}, {name.hexdigest()=}, {len(name.hexdigest())=} {name.digest_size=}")
    name = name.hexdigest() + f".{IMG_FORMAT}"
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format=IMG_FORMAT)
    in_mem_file.seek(0)
    return S3Object.from_orm(translango_bucket.put_object(Key=name, Body=in_mem_file)) #type: ignore

# def upload_image_fileobj_to_s3(image: bytes) -> S3Object:
#     name = hashlib.sha512(image).hexdigest() + '.' + IMG_FORMAT
#     return S3Object.from_orm(translango_bucket.put_object(Key=name, Body=image)) #type: ignore


def create_presigned_url(object_name: str, expiration: int=DEFAULT_EXPIRATION_TIME_FOR_PRESIGNED_URLS) -> pydantic.FileUrl:
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """
    response: str = s3_resource.meta.client.generate_presigned_url('get_object', #type: ignore
                                                    Params={'Bucket': BUCKET_NAME,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)

    # The response contains the presigned URL
    return pydantic.FileUrl(url=response, scheme='https')