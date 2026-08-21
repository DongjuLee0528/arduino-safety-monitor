from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

from mpu.config import (
    MAX_ACTIVE_TRACKS,
    TRACK_CENTROID_DISTANCE_RATIO,
    TRACK_IOU_THRESHOLD,
    TRACK_TTL_SECONDS,
)


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    union = aw * ah + bw * bh - iw * ih
    return (iw * ih) / union if union > 0 else 0.0


def centroid_distance(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return math.hypot((ax + aw / 2.0) - (bx + bw / 2.0), (ay + ah / 2.0) - (by + bh / 2.0))


@dataclass
class PersonTrack:
    track_id: int
    bbox: list[int]
    last_seen_frame: int
    last_seen_time: float
    consecutive_matches: int = 1
    missed_frames: int = 0
    consecutive_no_helmet: int = 0
    consecutive_helmet: int = 0
    classification_state: str = "unknown"
    incident_active: bool = False
    current_incident_id: str | None = None
    http_event_sent: bool = False
    incident_counted: bool = False
    buzzer_requested: bool = False
    last_incident_time: float = 0.0
    expired_active_incident: bool = False


class PersonTracker:
    def __init__(
        self,
        *,
        iou_threshold: float = TRACK_IOU_THRESHOLD,
        centroid_distance_ratio: float = TRACK_CENTROID_DISTANCE_RATIO,
        ttl_seconds: float = TRACK_TTL_SECONDS,
        max_tracks: int = MAX_ACTIVE_TRACKS,
        clock=time.monotonic,
    ):
        self.iou_threshold = float(iou_threshold)
        self.centroid_distance_ratio = float(centroid_distance_ratio)
        self.ttl_seconds = float(ttl_seconds)
        self.max_tracks = int(max_tracks)
        self.clock = clock
        if not math.isfinite(self.centroid_distance_ratio) or self.centroid_distance_ratio < 0:
            raise ValueError("centroid_distance_ratio must be finite and non-negative")
        if not math.isfinite(self.ttl_seconds) or self.ttl_seconds < 0:
            raise ValueError("ttl_seconds must be finite and non-negative")
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be positive")
        self.tracks: dict[int, PersonTrack] = {}
        self._next_track_id = 1
        self.frame_index = 0

    def update(self, bboxes: Iterable[list[int]], *, frame_shape) -> tuple[list[PersonTrack], list[PersonTrack]]:
        self.frame_index += 1
        now = float(self.clock())
        centroid_gate = self._centroid_gate(frame_shape)
        detections = [list(map(int, bbox)) for bbox in bboxes]
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()

        matches = []
        for track in sorted(self.tracks.values(), key=lambda t: t.track_id):
            for det_index, bbox in enumerate(detections):
                iou = bbox_iou(track.bbox, bbox)
                distance = centroid_distance(track.bbox, bbox)
                if iou >= self.iou_threshold or distance <= centroid_gate:
                    matches.append((-iou, distance, track.track_id, det_index))

        for _neg_iou, _distance, track_id, det_index in sorted(matches):
            if track_id in assigned_tracks or det_index in assigned_detections:
                continue
            track = self.tracks[track_id]
            track.bbox = detections[det_index]
            track.last_seen_frame = self.frame_index
            track.last_seen_time = now
            track.consecutive_matches += 1
            track.missed_frames = 0
            assigned_tracks.add(track_id)
            assigned_detections.add(det_index)

        for track in self.tracks.values():
            if track.track_id not in assigned_tracks:
                track.missed_frames += 1

        for det_index, bbox in enumerate(detections):
            if det_index not in assigned_detections:
                track = PersonTrack(
                    track_id=self._next_track_id,
                    bbox=bbox,
                    last_seen_frame=self.frame_index,
                    last_seen_time=now,
                )
                self.tracks[track.track_id] = track
                self._next_track_id += 1
                assigned_tracks.add(track.track_id)

        expired = self._prune_expired(now)
        expired.extend(self._enforce_max_tracks())
        matched = [
            self.tracks[track_id]
            for track_id in sorted(assigned_tracks)
            if track_id in self.tracks and self.tracks[track_id].missed_frames == 0
        ]
        return matched, expired

    def active_tracks(self) -> list[PersonTrack]:
        return [self.tracks[k] for k in sorted(self.tracks)]

    def reset(self) -> None:
        self.tracks.clear()
        self._next_track_id = 1
        self.frame_index = 0

    def _centroid_gate(self, frame_shape) -> float:
        if not isinstance(frame_shape, tuple) or len(frame_shape) < 2:
            raise ValueError("frame_shape must be a tuple containing height and width")
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("frame dimensions must be positive")
        return self.centroid_distance_ratio * math.hypot(float(width), float(height))

    def _prune_expired(self, now: float) -> list[PersonTrack]:
        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track.last_seen_time > self.ttl_seconds
        ]
        expired = []
        for track_id in sorted(expired_ids):
            track = self.tracks.pop(track_id)
            track.expired_active_incident = track.incident_active
            expired.append(track)
        return expired

    def _enforce_max_tracks(self) -> list[PersonTrack]:
        pruned = []
        while len(self.tracks) > self.max_tracks:
            track_id = min(
                self.tracks,
                key=lambda tid: (
                    self.tracks[tid].incident_active,
                    self.tracks[tid].missed_frames == 0,
                    self.tracks[tid].last_seen_frame,
                    tid,
                ),
            )
            track = self.tracks.pop(track_id)
            track.expired_active_incident = track.incident_active
            pruned.append(track)
        return pruned
