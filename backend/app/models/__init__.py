"""Database models."""

from app.models.annotation import Annotation, AnnotationType, CreatedBy
from app.models.annotation_video import AnnotationVideo
from app.models.detection import PlayerDetection
from app.models.game import Game
from app.models.game_roster import GameRoster, TeamSide
from app.models.jersey_number import JerseyNumber
from app.models.player import Player
from app.models.play import Play, PlayType
from app.models.video import ProcessingStatus, Video

__all__ = [
    "Annotation",
    "AnnotationType",
    "AnnotationVideo",
    "CreatedBy",
    "Game",
    "GameRoster",
    "JerseyNumber",
    "Player",
    "PlayerDetection",
    "Play",
    "PlayType",
    "ProcessingStatus",
    "TeamSide",
    "Video",
]
