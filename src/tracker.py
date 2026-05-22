"""
Centroid tracker with velocity and age tracking.
"""
import numpy as np


class CentroidTracker:
    def __init__(self, max_distance=80, max_missed=10):
        self.next_id = 0
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missed = max_missed
    
    def _new_track(self, detection):
        return {
            'centroid': detection['centroid'],
            'prev_centroid': None,
            'velocity': (0.0, 0.0),  # pixels per frame, (vx, vy)
            'missed': 0,
            'age': 1,                # number of frames this track has existed
            'detection': detection,
        }
    
    def _update_track(self, track, detection):
        prev = track['centroid']
        curr = detection['centroid']
        track['prev_centroid'] = prev
        track['centroid'] = curr
        track['velocity'] = (curr[0] - prev[0], curr[1] - prev[1])
        track['missed'] = 0
        track['age'] += 1
        track['detection'] = detection
    
    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = self._new_track(det)
                self.next_id += 1
            return dict(self.tracks)
        
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]['centroid'] for tid in track_ids]
        current_centroids = [d['centroid'] for d in detections]
        
        assigned_tracks = set()
        assigned_dets = set()
        
        if track_centroids and current_centroids:
            tc = np.array(track_centroids)
            dc = np.array(current_centroids)
            dists = np.linalg.norm(tc[:, None] - dc[None, :], axis=2).astype(float)
            
            while True:
                if dists.size == 0:
                    break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                min_val = dists[min_idx]
                if min_val > self.max_distance:
                    break
                ti, di = min_idx
                tid = track_ids[ti]
                if tid in assigned_tracks or di in assigned_dets:
                    dists[min_idx] = np.inf
                    continue
                self._update_track(self.tracks[tid], detections[di])
                assigned_tracks.add(tid)
                assigned_dets.add(di)
                dists[ti, :] = np.inf
                dists[:, di] = np.inf
        
        for di, det in enumerate(detections):
            if di not in assigned_dets:
                self.tracks[self.next_id] = self._new_track(det)
                self.next_id += 1
        
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]['missed'] += 1
        
        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if t['missed'] <= self.max_missed}
        
        return dict(self.tracks)