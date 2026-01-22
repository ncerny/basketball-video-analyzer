"""Database models."""

from app.models.annotation import Annotation, AnnotationType, CreatedBy
from app.models.annotation_video import AnnotationVideo
from app.models.detection import PlayerDetection
from app.models.game import Game
from app.models.game_roster import GameRoster
from app.models.jersey_number import JerseyNumber
from app.models.player import Player
from app.models.play import Play, PlayType
from app.models.processing_batch import BatchStatus, ProcessingBatch
from app.models.processing_job import JobStatus, JobType, ProcessingJob
from app.models.video import ProcessingStatus, Video
from app.models.game_roster import TeamSide

__all__ = [
    "Annotation",
    "AnnotationType",
    "AnnotationVideo",
    "BatchStatus",
    "CreatedBy",
    "Game",
    "GameRoster",
    "JerseyNumber",
    "JobStatus",
    "JobType",
    "Player",
    "PlayerDetection",
    "Play",
    "PlayType",
    "ProcessingBatch",
    "ProcessingJob",
    "ProcessingStatus",
    "TeamSide",
    "Video",
]
