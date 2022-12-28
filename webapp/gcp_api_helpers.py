import pydantic
from typing import Optional, List, Literal, Any
from google.cloud import vision, translate_v2  # type: ignore
import functools
import aws_api_helpers
from database import Language


class CloudTranslateResponse(pydantic.BaseModel):
    translatedText: str
    detectedSourceLanguage: Optional[str] = pydantic.Field(default=None)
    input: str
    model: Optional[str] = pydantic.Field(default=None)

    class Config:
        orm_mode = True


@functools.lru_cache(maxsize=2048)
def text_translate(
    text: str, target_lang: str, source_lang: Optional[str] = None
) -> "CloudTranslateResponse":
    client = translate_v2.Client()
    response = client.translate(  # type: ignore
        values=text, target_language=target_lang, source_language=source_lang
    )
    return CloudTranslateResponse.parse_obj(response)


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
    detections: List[
        CloudVisionAnnotationsWithTranslations
    ] = pydantic.Field(default_factory=list)

    class Config:
        orm_mode = True


def object_detection_from_url(url: pydantic.FileUrl) -> CloudVisionResponse:
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = url
    response = client.object_localization(image)  # type: ignore
    return CloudVisionResponse.from_response(response)  # type: ignore


@functools.lru_cache(maxsize=2048)
def object_detection_from_s3(image_name: str) -> CloudVisionResponse:
    presigned_url = aws_api_helpers.create_presigned_url(image_name)
    return object_detection_from_url(presigned_url)


def add_translation_to_CloudVisionDetections(
    detections: CloudVisionResponse,
    target_languages: List[Language] = pydantic.Field(min_items=1),
) -> CloudVisionAndTranslation:
    obj = CloudVisionAndTranslation.parse_obj(detections.dict())
    for detection in detections.detections:
        obj.detections.append(
            CloudVisionAnnotationsWithTranslations(**detection.dict(), translations=[])
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


def main():
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = (
        "https://cloud.google.com/vision/docs/images/bicycle_example.png"
    )
    vision_response = CloudVisionResponse.from_response(  # type: ignore
        client.object_localization(image)  # type: ignore
    )
    vision_response_with_translations = add_translation_to_CloudVisionDetections(
        vision_response, [Language(code="hi")]
    )
    print(vision_response_with_translations)
    print(text_translate("hello", "hi"))


if __name__ == "__main__":
    main()
