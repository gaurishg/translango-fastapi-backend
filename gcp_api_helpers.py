import pydantic
from typing import Optional, List, Literal, Any
from google.cloud import vision, translate_v2, storage  # type: ignore
import functools

from database import LanguageInDB, Language
from PIL import Image
import hashlib
import io

vision_client = vision.ImageAnnotatorClient()
translate_client = translate_v2.Client()
storage_client = storage.Client()
storage_bucket = storage_client.bucket("translango-images")
IMG_FORMAT = "jpeg"


class TextTranslateResponseToFrontend(pydantic.BaseModel):
    translatedText: str
    detectedSourceLanguage: Optional[str] = pydantic.Field(
        default=None, alias="source_lang"
    )


class CloudTranslateResponse(TextTranslateResponseToFrontend):
    input: str
    model: Optional[str] = pydantic.Field(default=None)

    class Config:
        orm_mode = True


@functools.lru_cache(maxsize=2048)
def text_translate(
    text: str, target_lang: str, source_lang: Optional[str] = None
) -> "CloudTranslateResponse":
    if target_lang == source_lang:
        return CloudTranslateResponse(translatedText=text, input=text)
    response = translate_client.translate(  # type: ignore
        values=text, target_language=target_lang, source_language=source_lang
    )
    return CloudTranslateResponse.parse_obj(response)


@functools.lru_cache(maxsize=2048)
def get_languages(target_lang: str = "en"):
    languages = translate_client.get_languages(target_language=target_lang)  # type: ignore
    return [Language(code=l["language"], name=l["name"]) for l in languages]  # type: ignore


class CloudVisionFromURIRequest(pydantic.BaseModel):
    class Feature(pydantic.BaseModel):
        maxResults: Optional[int] = pydantic.Field(default=None)
        type: Literal["OBJECT_LOCALIZATION"] = "OBJECT_LOCALIZATION"

        class Config:
            orm_mode = True

    class Image(pydantic.BaseModel):
        class ImageSource(pydantic.BaseModel):
            imageUri: pydantic.FileUrl

            class Config:
                orm_mode = True

        source: ImageSource

        class Config:
            orm_mode = True

    features: List[Feature] = [Feature()]
    image: Image

    class Config:
        orm_mode = True


class CloudVisionNormalizedVertex(pydantic.BaseModel):
    x: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    y: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        orm_mode = True


class CloudVisionBoundingPoly(pydantic.BaseModel):
    normalizedVertices: List[CloudVisionNormalizedVertex] = pydantic.Field(
        min_items=4, max_items=4
    )

    class Config:
        orm_mode = True


class CloudVisionLocalizedObjectAnnotation(pydantic.BaseModel):
    mid: str = pydantic.Field(
        description="a machine-generated identifier (MID) corresponding to a label's Google Knowledge Graph entry"
    )
    name: str = pydantic.Field(description="name of object in English")
    score: float = pydantic.Field(
        ge=0.0, le=1.0, description="confidence score of the model, range [0, 1]"
    )
    boundingPoly: CloudVisionBoundingPoly

    class Config:
        orm_mode = True

    class GCPAnnotation(pydantic.BaseModel):
        mid: str
        name: str
        score: float
        bounding_poly: Any

    @staticmethod
    def from_single_annotation(
        annotation: "CloudVisionLocalizedObjectAnnotation.GCPAnnotation",
    ) -> "CloudVisionLocalizedObjectAnnotation":
        obj: CloudVisionLocalizedObjectAnnotation = (
            CloudVisionLocalizedObjectAnnotation(
                mid=annotation.mid,
                name=annotation.name,
                score=annotation.score,
                boundingPoly=CloudVisionBoundingPoly(
                    normalizedVertices=[
                        CloudVisionNormalizedVertex.from_orm(v)
                        for v in annotation.bounding_poly.normalized_vertices
                    ]
                ),
            )
        )
        return obj


class Translation(pydantic.BaseModel):
    language: str
    translation: str

    class Config:
        orm_mode = True


class CloudVisionAnnotationsWithTranslations(CloudVisionLocalizedObjectAnnotation):
    translatedName: str
    translations: List[Translation] = pydantic.Field(default_factory=list)


class CloudVisionResponse(pydantic.BaseModel):
    detections: List[CloudVisionLocalizedObjectAnnotation]

    class Config:
        orm_mode = True

    @staticmethod
    def from_response(response) -> "CloudVisionResponse":  # type: ignore
        obj = CloudVisionResponse(
            detections=[
                CloudVisionLocalizedObjectAnnotation.from_single_annotation(annotation)  # type: ignore
                for annotation in response.localized_object_annotations  # type: ignore
            ]
        )
        return obj


class CloudVisionAndTranslation(pydantic.BaseModel):
    image_name: str = ""
    detections: List[CloudVisionAnnotationsWithTranslations] = pydantic.Field(
        default_factory=list
    )

    class Config:
        orm_mode = True


def object_detection_from_url(url: pydantic.FileUrl) -> CloudVisionResponse:
    image = vision.Image()
    image.source.image_uri = url
    response = vision_client.object_localization(image)  # type: ignore
    return CloudVisionResponse.from_response(response)  # type: ignore


class CloudVisionTextDetection(pydantic.BaseModel):
    class Config:
        orm_mode = True

    locale: str
    description: str
    bounding_poly: CloudVisionBoundingPoly


class CloudVisionTextDetectionWithSourceLang(CloudVisionTextDetection):
    source_lang: str


def add_translation_to_CloudVisionDetections(
    detections: CloudVisionResponse,
    target_languages: List[LanguageInDB] = pydantic.Field(min_items=1),
    source_language: LanguageInDB = LanguageInDB(code="en"),
) -> CloudVisionAndTranslation:
    obj = CloudVisionAndTranslation()
    for detection in detections.detections:
        obj.detections.append(
            CloudVisionAnnotationsWithTranslations(
                **detection.dict(),
                translations=[],
                translatedName=text_translate(
                    detection.name, source_language.code, "en"
                ).translatedText,
            )
        )
        for target_lang in target_languages:
            obj.detections[-1].translations.append(
                Translation(
                    language=target_lang.code,
                    translation=text_translate(
                        detection.name, target_lang.code
                    ).translatedText,
                )
            )
    return obj


def object_detection_with_translation_from_url(
    url: pydantic.FileUrl,
    target_languages: List[LanguageInDB] = pydantic.Field(min_items=1),
    source_language: LanguageInDB = LanguageInDB(code="en"),
) -> CloudVisionAndTranslation:
    return add_translation_to_CloudVisionDetections(
        object_detection_from_url(url=url),
        target_languages=target_languages,
        source_language=source_language,
    )


def upload_PIL_Image(image: Image.Image):
    name = hashlib.sha512(image.tobytes()).hexdigest() + "." + IMG_FORMAT
    # Check if object already exists
    blob = storage_bucket.blob(name)
    if blob.exists():
        return blob
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format=IMG_FORMAT)
    in_mem_file.seek(0)
    blob.upload_from_file(in_mem_file)
    return blob

def filename_to_gs_bucket_uri(name: str) -> pydantic.FileUrl:
    return pydantic.FileUrl(url=f'gs://translango-images/{name}', scheme='gs')


def main():
    image = Image.open("cats-and-dogs.jpg")
    blob = upload_PIL_Image(image=image)
    filename: str = blob.name  # type: ignore
    print(filename)
    print(
        object_detection_with_translation_from_url(
            pydantic.FileUrl(url=f"gs://translango-images/{filename}", scheme="gs"),
            target_languages=[LanguageInDB(code="hi"), LanguageInDB(code="ja")],
        )
    )


if __name__ == "__main__":
    main()
