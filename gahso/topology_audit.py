"""Audit nominal links, visibility and independent-nearest conflicts over one orbit."""
from collections import Counter
from .network import distance, line_of_sight


def audit_topology(constellation, samples=17):
    rows = []
    cfg = constellation.config
    for i in range(samples):
        snap = constellation.snapshot(constellation.orbital_period_s * i / (samples - 1))
        degree = Counter(n for edge in snap.nominal_links for n in edge)
        blocked = Counter()
        nearest_conflicts = 0
        matching_not_individual_nearest = 0
        for (a, b), link in snap.nominal_links.items():
            pa, pb = snap.satellites[a].position_km, snap.satellites[b].position_km
            assert abs(distance(pa, pb) - link.distance_km) < 1e-7
            if (a, b) not in snap.links:
                blocked["coincident" if link.distance_km < 1e-6 else "earth_occlusion"] += 1
            if link.kind == "inter":
                for src, dst in ((a, b), (b, a)):
                    choices = [(distance(snap.satellites[src].position_km, s.position_km), n)
                               for n, s in snap.satellites.items() if s.plane == snap.satellites[dst].plane]
                    if link.distance_km > min(choices)[0] + 1e-7:
                        matching_not_individual_nearest += 1
        # Each directed adjacent-plane pair: independently nearest choices can collide.
        for plane in range(cfg.planes):
            targets = {(plane - 1) % cfg.planes, (plane + 1) % cfg.planes}
            if not cfg.close_plane_seam:
                targets = {p for p in targets if abs(p-plane) == 1}
            for target in targets:
                selected = [min((distance(s.position_km, t.position_km), j)
                                for j, t in snap.satellites.items() if t.plane == target)[1]
                            for s in snap.satellites.values() if s.plane == plane]
                nearest_conflicts += len(selected) - len(set(selected))
        for n, sat in snap.satellites.items():
            expected = 4 if cfg.close_plane_seam or 0 < sat.plane < cfg.planes-1 else 3
            assert degree[n] == expected
            assert len(snap.neighbors(n)) <= expected
            for neighbor in snap.neighbors(n):
                assert n in snap.neighbors(neighbor)
                if cfg.enforce_line_of_sight:
                    assert line_of_sight(sat.position_km, snap.satellites[neighbor].position_km, cfg.earth_radius_km)
        rows.append({"time_s": snap.time_s, "nominal_links": len(snap.nominal_links),
                     "actual_links": len(snap.links), "actual_degree_counts": dict(Counter(len(snap.neighbors(n)) for n in snap.satellites)),
                     "unavailable_reasons": dict(blocked), "independent_nearest_collisions": nearest_conflicts,
                     "matching_not_individual_nearest_count": matching_not_individual_nearest})
    return {"passed": True, "matching": "minimum_total_distance_one_to_one_per_adjacent_plane",
            "same_plane_ring": True, "plane_ring": cfg.close_plane_seam, "snapshots": rows}
