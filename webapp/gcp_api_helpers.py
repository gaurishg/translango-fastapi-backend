import pydantic
from typing import Optional, List, Literal, Any
from google.cloud import vision  # type: ignore


class CloudVisionFromURIRequest(pydantic.BaseModel):
    class Feature(pydantic.BaseModel):
        maxResults: Optional[int] = pydantic.Field(default=None)
        type: Literal["OBJECT_LOCALIZATION"] = "OBJECT_LOCALIZATION"

        class Config:
            orm_mode = True

    class Features(pydantic.BaseModel):
        __root__: List["CloudVisionFromURIRequest.Feature"]

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

    @staticmethod
    def from_single_annotation(annotation) -> "CloudVisionLocalizedObjectAnnotation":
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


class CloudVisionResponse(pydantic.BaseModel):
    __root__: List[CloudVisionLocalizedObjectAnnotation]

    class Config:
        orm_mode = True

    @staticmethod
    def from_response(response) -> "CloudVisionResponse":
        obj = CloudVisionResponse(
            __root__=[
                CloudVisionLocalizedObjectAnnotation.from_single_annotation(annotation)
                for annotation in response.localized_object_annotations
            ]
        )
        return obj


def object_detection_from_url(url: pydantic.FileUrl):
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = url
    response = client.object_localization(image)
    return response


def main():
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = (
        "https://cloud.google.com/vision/docs/images/bicycle_example.png"
    )
    response = CloudVisionResponse.from_response(client.object_localization(image))
    print(response.json())


if __name__ == "__main__":
    main()
