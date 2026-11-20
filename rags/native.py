"""Small, explicit native algorithms; no named research-backend aliases."""
from collections import Counter, defaultdict, deque
import math


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding dimensions differ; reindex the source")
    if any(not math.isfinite(x) for x in (*left, *right)):
        raise ValueError("Embeddings must be finite")
    norm = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / norm if norm else 0.0


def rrf(rankings: list[list[str]], limit: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(dict.fromkeys(ranking)):
            scores[identifier] = scores.get(identifier, 0.0) + 1 / (61 + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


def cluster_vectors(nodes: list[dict], group_size: int = 4) -> list[list[dict]]:
    """Deterministic greedy centroid clustering, with explicit bounded fan-in."""
    remaining = sorted(nodes, key=lambda n: n["id"])
    groups = []
    while remaining:
        seed = remaining.pop(0)
        candidates = sorted(remaining, key=lambda n: (-cosine(seed["embedding"], n["embedding"]), n["id"]))
        group = [seed, *candidates[:group_size - 1]]
        chosen = {node["id"] for node in group}
        remaining = [node for node in remaining if node["id"] not in chosen]
        groups.append(group)
    return groups


def graph_search(edges: list[dict], seeds: list[str], max_hops: int = 2) -> tuple[list[tuple[str, float]], dict]:
    """Undirected neighborhood retrieval with source evidence on every traversed edge."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    edge_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in edges:
        a, b = edge["source"].casefold(), edge["target"].casefold()
        neighbors[a].add(b)
        neighbors[b].add(a)
        edge_map[(a, b)].append(edge["evidence_id"])
        edge_map[(b, a)].append(edge["evidence_id"])
    # Asynchronous label propagation: actual graph communities, not merely
    # connected components or document co-occurrence labels.
    labels = {node: node for node in neighbors}
    for _ in range(12):
        changed = False
        for node in sorted(neighbors):
            counts = Counter(labels[n] for n in neighbors[node])
            if counts:
                winner = min(counts, key=lambda label: (-counts[label], label))
                if labels[node] != winner:
                    labels[node] = winner
                    changed = True
        if not changed:
            break
    groups: dict[str, list[str]] = defaultdict(list)
    for node, label in labels.items():
        groups[label].append(node)
    starts = [node for node in sorted(neighbors) if any(s and s in node for s in seeds)]
    if not starts:
        return [], {"communities": len(groups), "seeds": [], "visited": 0, "max_hops": max_hops}
    distance = {node: 0 for node in starts}
    queue = deque(starts)
    scores: dict[str, float] = {}
    while queue:
        node = queue.popleft()
        depth = distance[node]
        if depth >= max_hops:
            continue
        for other in sorted(neighbors[node]):
            for evidence_id in edge_map[(node, other)]:
                scores[evidence_id] = max(scores.get(evidence_id, 0.0), 1 / (depth + 1))
            if other not in distance:
                distance[other] = depth + 1
                queue.append(other)
    selected_groups = {labels[node] for node in starts}
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0])), {
        "communities": len(groups), "selected_communities": [sorted(groups[g]) for g in sorted(selected_groups)],
        "seeds": starts, "visited": len(distance), "max_hops": max_hops,
    }


def equal_value(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) is not type(right) and not (type(left) in (int, float) and type(right) in (int, float)):
        return False
    return left == right


def row_matches(row: dict, predicates: list[dict]) -> bool:
    for predicate in predicates:
        field, operator, expected = predicate["field"], predicate["op"], predicate.get("value")
        if field not in row:
            return False
        actual = row[field]
        if operator == "eq":
            matched = equal_value(actual, expected)
        elif operator == "ne":
            matched = not equal_value(actual, expected)
        elif operator == "in":
            matched = isinstance(expected, list) and any(equal_value(actual, item) for item in expected)
        elif operator == "contains":
            matched = isinstance(actual, str) and isinstance(expected, str) and expected in actual
        elif operator in {"gt", "gte", "lt", "lte"}:
            if type(actual) not in (int, float) or type(expected) not in (int, float):
                return False
            matched = {"gt": actual > expected, "gte": actual >= expected,
                       "lt": actual < expected, "lte": actual <= expected}[operator]
        else:
            raise ValueError("Unsupported structured predicate")
        if not matched:
            return False
    return True


def aggregate_rows(rows: list[dict], aggregate: dict | None) -> dict | None:
    if aggregate is None:
        return None
    operation = aggregate["operation"]
    if operation == "count":
        return {"operation": operation, "value": len(rows), "matched_rows": len(rows)}
    field = aggregate.get("field")
    if not field:
        raise ValueError("Numeric aggregation requires a field")
    values = [row.get(field) for row in rows]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Aggregation field must be present and numeric in every matched row")
    if not values:
        value = 0 if operation == "sum" else None
    elif operation == "sum":
        value = sum(values)
    elif operation == "mean":
        value = sum(values) / len(values)
    elif operation == "min":
        value = min(values)
    elif operation == "max":
        value = max(values)
    else:
        raise ValueError("Unsupported structured aggregation")
    return {"operation": operation, "field": field, "value": value, "matched_rows": len(rows)}
