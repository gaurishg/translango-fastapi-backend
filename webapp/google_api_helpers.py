import pydantic
from typing import Optional, List, Literal


class CloudVisionFromURIRequest(pydantic.BaseModel):
    class Feature(pydantic.BaseModel):
        maxResults: Optional[int] = pydantic.Field(default=None)
        type: Literal["OBJECT_LOCALIZATION"] = "OBJECT_LOCALIZATION"

    class Features(pydantic.BaseModel):
        __root__: List["CloudVisionFromURIRequest.Feature"]

    class Image(pydantic.BaseModel):
        class ImageSource(pydantic.BaseModel):
            imageUri: pydantic.FileUrl

        source: ImageSource

    features: List[Feature] = [Feature()]
    image: Image

class CloudVisionNormalizedVertex(pydantic.BaseModel):
    x: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    y: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)

class CloudVisionBoundingPoly(pydantic.BaseModel):
    normalizedVertices: List[CloudVisionNormalizedVertex] = pydantic.Field(min_items=4, max_items=4)

class CloudVisionLocalizedObjectAnnotation(pydantic.BaseModel):
    mid: str = pydantic.Field(description="a machine-generated identifier (MID) corresponding to a label's Google Knowledge Graph entry")
    name: str = pydantic.Field(description="name of object in English")
    score: float = pydantic.Field(ge=0.0, le=1.0, description="confidence score of the model, range [0, 1]")
    boundingPoly: CloudVisionBoundingPoly

class CloudVisionResponse(pydantic.BaseModel):
    localizedObjectAnnotations: List[CloudVisionLocalizedObjectAnnotation]


print(
    CloudVisionFromURIRequest(
        image=CloudVisionFromURIRequest.Image(
            source=CloudVisionFromURIRequest.Image.ImageSource(
                imageUri=pydantic.FileUrl(
                    url="https://cloud.google.com/vision/docs/images/bicycle_example.png",
                    scheme="https",
                )
            )
        )
    ).json()
)
