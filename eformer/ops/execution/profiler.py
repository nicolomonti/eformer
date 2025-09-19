# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Any

import numpy as np
from jax.profiler import ProfileData

XSpace = Any


class ProfilingError(Exception):
    """Exception raised when profiling operations fail."""

    pass


class ProfileAnalyzer:
    """Enhanced profile analysis for autotune optimization."""

    def __init__(self, enable_caching: bool = True, cache_size: int = 100):
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self._event_cache = {} if enable_caching else None

    def clear_cache(self):
        """Clear the internal event cache."""
        if self._event_cache is not None:
            self._event_cache.clear()


def _get_stat_value(stat, metadata):
    if stat.ref_value != 0:
        return metadata[stat.ref_value].name
    for key in ["double", "int64", "uint64", "ref"]:
        if getattr(stat, key + "_value") != 0:
            return getattr(stat, key + "_value")
    for key in ["bytes", "str"]:
        if len(getattr(stat, key + "_value")) > 0:
            return getattr(stat, key + "_value")


def _parse_stats(stats, stat_metadata):
    if stat_metadata is not None:
        return {stat_metadata[stat.metadata_id].name: _get_stat_value(stat, stat_metadata) for stat in stats}
    return dict(stats)


def _parse_event(event, event_metadata, stat_metadata, prefix_filter: str = "", line_name: str = ""):
    # Note: prefix_filter used implicitly in filtering logic, line_name for context
    if event_metadata is not None:
        name = event_metadata[event.metadata_id].name
    else:
        name = event.name
    stats = _parse_stats(event.stats, stat_metadata)
    name = stats.get("hlo_module", name)  # hlo_module is GPU, name is TPU
    # if not name.startswith(prefix_filter):
    #  return None
    program_id = stats.get("program_id", stats.get("run_id"))  # program_id is GPU, run_id is TPU
    scope_range_id = stats.get("scope_range_id", "None")
    key = f"{name}({program_id}-{scope_range_id})"
    if hasattr(event, "duration_ps"):
        stats["start_ps"] = int(event.offset_ps)
        stats["end_ps"] = int(event.offset_ps) + int(event.duration_ps)
        stats["duration_ps"] = int(event.duration_ps)
    else:
        stats["start_ps"] = int(event.start_ns * 1000)
        stats["end_ps"] = int(event.start_ns * 1000) + int(event.duration_ns * 1000)
        stats["duration_ps"] = int(event.duration_ns * 1000)
    return dict(unified_name=key, fusion=name, line_name=line_name, **stats)


def parse_profile_from_bytes(profile_bytes: bytes) -> ProfileData:  # type:ignore
    """Parse profile data from bytes with enhanced error handling."""
    try:
        return ProfileData.from_serialized_xspace(profile_bytes)
    except Exception as e:
        raise ProfilingError(f"Failed to parse profile data: {e}") from e


def find_device_plane_ids(p: XSpace, device_str: str) -> list[int]:
    """Find device plane IDs with improved error handling."""
    try:
        plane_ids = [i for i, plane in enumerate(p.planes) if device_str.lower() in plane.name.lower()]
        if not plane_ids:
            available_devices = [plane.name for plane in p.planes]
            raise ProfilingError(f"No planes found for device '{device_str}'. Available devices: {available_devices}")
        return plane_ids
    except AttributeError as e:
        raise ProfilingError(f"Invalid profile structure: {e}") from e


def _find_children(own_name: str, start_ps: int, end_ps: int, events_sorted: list[dict[str, Any]]):
    """Find all events that are fully subsumed by the `start_ps` - `end_ps` range."""
    idx = np.searchsorted(np.sort(np.array([event["start_ps"] for event in events_sorted])), start_ps - 1)
    children = []
    while idx < len(events_sorted) and events_sorted[idx]["start_ps"] <= end_ps:
        ts, te = events_sorted[idx]["start_ps"], events_sorted[idx]["end_ps"]
        if ts >= start_ps and te <= end_ps and events_sorted[idx]["unified_name"] != own_name:
            children.append(events_sorted[idx])
        idx += 1
    return children


def _sum_events(events):
    """Sum the time of all events as right extreme - left extreme subtracting empty space."""
    if len(events) == 0:
        return 0
    if len(events) == 1:
        return events[0]["end_ps"] - events[0]["start_ps"]
    starts, ends = np.array([e["start_ps"] for e in events]), np.array([e["end_ps"] for e in events])
    min_start, max_end = int(np.min(starts)), int(np.max(ends))
    sorted_ends = np.sort(ends)
    empty_ends = np.where(
        ~np.any((sorted_ends[None, :-1] < ends[:, None]) & (sorted_ends[None, :-1] >= starts[:, None]), axis=0)
    )[0]
    sorted_starts = np.sort(starts)
    empty_space = sum(
        int(ends[end_idx]) - int(sorted_starts[np.searchsorted(sorted_starts, ends[end_idx])]) for end_idx in empty_ends
    )
    assert empty_space < (max_end - min_start)
    return max_end - min_start - empty_space


def get_events_from_plane(
    p,
    plane_idx: int,
    prefix_filter: str = "",
    event_filter_regex: str | None = None,
    min_duration_ns: float = 0.0,
    max_events: int | None = None,
) -> dict[str, float]:
    """Returns a dict of xla module names to their execution time in seconds.

    Args:
        p: Profile data object
        plane_idx: Index of the plane to analyze
        prefix_filter: Filter events by name prefix
        event_filter_regex: Regex filter for event names
        min_duration_ns: Minimum duration threshold in nanoseconds
        max_events: Maximum number of events to return (for performance)

    Returns:
        Dictionary mapping event names to execution times in seconds

    Raises:
        ProfilingError: If profile analysis fails
    """
    try:
        planes = list(p.planes)
        if plane_idx >= len(planes):
            raise ProfilingError(f"Plane index {plane_idx} out of range (0-{len(planes) - 1})")

        plane = planes[plane_idx]
        timed_events = {}

        # Get metadata with better error handling
        event_metadata = getattr(plane, "event_metadata", None)
        stat_metadata = getattr(plane, "stat_metadata", None)

        all_parsed_events = []
        total_events_processed = 0

        for line in plane.lines:
            if max_events and total_events_processed >= max_events:
                break

            for event in line.events:
                if max_events and total_events_processed >= max_events:
                    break

                parsed_event = _parse_event(event, event_metadata, stat_metadata, prefix_filter, line_name=line.name)

                if parsed_event is not None:
                    # Apply duration filtering
                    duration_ns = parsed_event.get("duration_ps", 0) / 1000
                    if duration_ns >= min_duration_ns:
                        all_parsed_events.append(parsed_event)
                        total_events_processed += 1

        if not all_parsed_events:
            return {}

        # Sort events for efficient child finding
        sorted_events = sorted(all_parsed_events, key=lambda x: x["start_ps"])

        filtered_events = []
        for event in all_parsed_events:
            event_name = event["unified_name"]

            # Apply prefix filter
            if not event_name.startswith(prefix_filter):
                continue

            # Find children events
            event["children"] = _find_children(event_name, event["start_ps"], event["end_ps"], sorted_events)

            # Apply regex filter if specified
            if event_filter_regex is not None:
                try:
                    pattern = re.compile(event_filter_regex)
                    new_children = [ch for ch in event["children"] if pattern.search(ch["unified_name"]) is not None]
                    event["children"] = new_children
                    event["children_duration"] = _sum_events(new_children)
                except re.error as e:
                    raise ProfilingError(f"Invalid regex pattern '{event_filter_regex}': {e}") from e

            filtered_events.append(event)

        # Convert to timing dictionary with better error handling
        timed_events = {}
        for event in filtered_events:
            try:
                duration_seconds = event.get("children_duration", (event["end_ps"] - event["start_ps"])) / 1e12

                # Sanity check for timing values
                if duration_seconds < 0:
                    continue

                timed_events[event["unified_name"]] = duration_seconds
            except (KeyError, TypeError, ZeroDivisionError):
                # Skip malformed events
                continue

        return timed_events

    except Exception as e:
        raise ProfilingError(f"Failed to extract events from plane {plane_idx}: {e}") from e
