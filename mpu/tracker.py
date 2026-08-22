"""
Person Tracking Module

Provides lightweight multi-person tracking across consecutive video frames
using IoU (Intersection-over-Union) and centroid-distance matching.

Tracking strategy:
  1. Each detected person bbox is matched to the nearest existing PersonTrack
     by IoU >= iou_threshold OR centroid distance <= centroid_gate.
  2. Unmatched detections spawn new tracks; unmatched tracks accumulate
     missed_frames until they exceed ttl_seconds and are pruned.
  3. When total tracks exceed max_tracks the oldest, least-active track is
     evicted to protect edge-hardware memory.

Per-track state (consecutive_no_helmet / consecutive_helmet) is written by
the caller (HelmetDetectionSystem) and used to gate incident creation.

Usage:
    tracker = PersonTracker()
    matched, expired = tracker.update(bboxes, frame_shape=frame.shape)
    for track in matched:
        # track.bbox, track.track_id, track.incident_active, …
"""

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
    """
    Compute Intersection-over-Union between two [x, y, w, h] bounding boxes.

    Returns 0.0 when the boxes do not overlap or either box has zero area.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix  # Width of intersection; negative = no overlap
    ih = min(ay + ah, by + bh) - iy  # Height of intersection; negative = no overlap
    if iw <= 0 or ih <= 0:
        return 0.0
    union = aw * ah + bw * bh - iw * ih
    return (iw * ih) / union if union > 0 else 0.0


def centroid_distance(a, b) -> float:
    """
    Euclidean distance between the centroids of two [x, y, w, h] bounding boxes.

    Used as a secondary matching gate when IoU is zero (e.g. small/distant boxes
    that do not overlap but clearly track the same person).
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return math.hypot((ax + aw / 2.0) - (bx + bw / 2.0), (ay + ah / 2.0) - (by + bh / 2.0))


@dataclass
class PersonTrack:
    """
    State container for one tracked person across multiple video frames.

    Identity fields (set by PersonTracker):
        track_id:              Monotonically increasing unique ID assigned at creation.
        bbox:                  Most recent [x, y, w, h] bounding box in pixel coords.
        last_seen_frame:       frame_index value when this track was last matched.
        last_seen_time:        Monotonic clock value when this track was last matched.
        consecutive_matches:   Total frames where a detection was matched to this track.
        missed_frames:         Frames since the last successful match (pruned when too high).

    Helmet-classification fields (set by HelmetDetectionSystem):
        consecutive_no_helmet: Unbroken streak of "no_helmet" frames for this track.
        consecutive_helmet:    Unbroken streak of "helmet" frames for this track.
        classification_state:  Latest label: "helmet" | "no_helmet" | "unknown".

    Incident-tracking fields (set by HelmetDetectionSystem):
        incident_active:       True while a no-helmet incident is open for this track.
        current_incident_id:   Unique string ID of the open incident, or None.
        http_event_sent:       True once the HTTP alert has been dispatched.
        incident_counted:      True once this incident was counted in daily statistics.
        buzzer_requested:      True once the buzzer was triggered for this incident.
        last_incident_time:    time.time() when the most recent incident was opened.

    Expiry field (set by PersonTracker._prune_expired / _enforce_max_tracks):
        expired_active_incident: True if the track expired while incident_active was True,
                                 so callers can log or escalate unresolved incidents.
    """
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
    """
    Frame-to-frame multi-person tracker using IoU + centroid-distance matching.

    Matching logic (called once per frame via update()):
      For each existing track, every new detection is scored by IoU and centroid
      distance.  Candidate pairs are sorted by descending IoU then ascending
      distance; the greedy assignment loop picks the best pair first, so the
      highest-IoU match wins ties.  Unmatched detections become new tracks;
      unmatched tracks increment missed_frames.

    Pruning:
      - TTL pruning (_prune_expired): any track not matched within ttl_seconds
        is removed regardless of missed_frames.
      - Capacity pruning (_enforce_max_tracks): when len(tracks) > max_tracks
        the track that is least active (no open incident, most missed frames,
        oldest) is evicted.

    Args:
        iou_threshold:           Minimum IoU to consider a detection a match.
        centroid_distance_ratio: Gate as a fraction of the frame diagonal;
                                 e.g. 0.2 at 640×480 ≈ 160 px.
        ttl_seconds:             Seconds before an unmatched track is pruned.
        max_tracks:              Hard upper bound on simultaneous live tracks.
        clock:                   Monotonic clock callable; override in tests.
    """
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
        """
        Advance the tracker by one frame.

        Args:
            bboxes:      Iterable of [x, y, w, h] detections for this frame.
            frame_shape: Frame dimensions as (height, width[, channels]) — used
                         to compute the centroid distance gate.

        Returns:
            (matched_tracks, expired_tracks)
            matched_tracks: Tracks that were successfully matched this frame
                            (missed_frames == 0).
            expired_tracks: Tracks removed by TTL or capacity pruning.
                            Each expired track has expired_active_incident set.
        """
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
