from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List

import jsonschema

from .diagnostics import Diagnostic, Diagnostics

GRAPH_VERSION = "1.0.0"
PORT_KINDS = {"fluid", "thermal", "structural", "electrical", "control", "data"}
PORT_DIRECTIONS = {"in", "out", "bidir"}


def load_graph(path: Path) -> Dict[str, Any]:
    data = _load_data(path)
    if not isinstance(data, dict):
        raise ValueError("Graph must be a JSON object")
    return data


def validate_graph(graph: Dict[str, Any]) -> Diagnostics:
    diagnostics = Diagnostics()
    graph_version = graph.get("graph_version")
    if not graph_version:
        diagnostics.add(
            Diagnostic(
                code="E-GRAPH-VERSION",
                message="Missing graph_version",
                location="graph_version",
            )
        )
    elif _major(str(graph_version)) != _major(GRAPH_VERSION):
        diagnostics.add(
            Diagnostic(
                code="E-GRAPH-VERSION",
                message=f"Graph version mismatch: expected {GRAPH_VERSION}, got {graph_version}",
                location="graph_version",
            )
        )
    nodes: Dict[str, Any] = {}
    node_list = graph.get("nodes", []) or []
    for node in node_list:
        node_id = node.get("id")
        if not node_id:
            diagnostics.add(
                Diagnostic(
                    code="E-GRAPH-NODE-ID",
                    message="Node id missing",
                    location="nodes",
                )
            )
            continue
        if node_id in nodes:
            diagnostics.add(
                Diagnostic(
                    code="E-GRAPH-NODE-DUP",
                    message="Duplicate node id detected",
                    location="nodes",
                )
            )
            continue
        nodes[node_id] = node

    ports_by_node: Dict[str, Dict[str, Any]] = {}
    for node_id, node in nodes.items():
        ports: Dict[str, Any] = {}
        for port in node.get("ports", []) or []:
            port_id = port.get("id")
            if not port_id:
                diagnostics.add(
                    Diagnostic(
                        code="E-GRAPH-PORT-ID",
                        message=f"Port id missing for node {node_id}",
                        location=f"nodes.{node_id}.ports",
                    )
                )
                continue
            if port_id in ports:
                diagnostics.add(
                    Diagnostic(
                        code="E-GRAPH-PORT-DUP",
                        message=f"Duplicate port id '{port_id}' on node {node_id}",
                        location=f"nodes.{node_id}.ports",
                    )
                )
                continue
            kind = port.get("kind")
            if kind and kind not in PORT_KINDS:
                diagnostics.add(
                    Diagnostic(
                        code="E-GRAPH-PORT-KIND",
                        message=f"Unknown port kind '{kind}' for {node_id}.{port_id}",
                        location=f"nodes.{node_id}.ports.{port_id}",
                    )
                )
            direction = port.get("direction")
            if direction and direction not in PORT_DIRECTIONS:
                diagnostics.add(
                    Diagnostic(
                        code="E-GRAPH-PORT-DIR",
                        message=f"Unknown port direction '{direction}' for {node_id}.{port_id}",
                        location=f"nodes.{node_id}.ports.{port_id}",
                    )
                )
            ports[port_id] = port
        ports_by_node[node_id] = ports

    connected_ports: set[tuple[str, str]] = set()
    for edge in graph.get("edges", []) or []:
        from_ep = edge.get("from") or {}
        to_ep = edge.get("to") or {}
        from_node = from_ep.get("node")
        from_port = from_ep.get("port")
        to_node = to_ep.get("node")
        to_port = to_ep.get("port")
        if not from_node or not from_port or not to_node or not to_port:
            diagnostics.add(
                Diagnostic(
                    code="E-GRAPH-EDGE-PORT",
                    message="Edge missing from/to endpoint",
                    location="edges",
                )
            )
            continue
        if from_node not in ports_by_node or from_port not in ports_by_node[from_node]:
            diagnostics.add(
                Diagnostic(
                    code="E-GRAPH-EDGE-PORT",
                    message=f"Edge references unknown port: {from_node}.{from_port} -> {to_node}.{to_port}",
                    location="edges",
                )
            )
            continue
        if to_node not in ports_by_node or to_port not in ports_by_node[to_node]:
            diagnostics.add(
                Diagnostic(
                    code="E-GRAPH-EDGE-PORT",
                    message=f"Edge references unknown port: {from_node}.{from_port} -> {to_node}.{to_port}",
                    location="edges",
                )
            )
            continue
        src = ports_by_node[from_node][from_port]
        dst = ports_by_node[to_node][to_port]
        src_dir = src.get("direction")
        dst_dir = dst.get("direction")
        if src_dir not in {"out", "bidir"} or dst_dir not in {"in", "bidir"}:
            diagnostics.add(
                Diagnostic(
                    code="E-GRAPH-EDGE-DIR",
                    message=f"Edge direction mismatch: {from_node}.{from_port} -> {to_node}.{to_port}",
                    location="edges",
                )
            )
        src_kind = src.get("kind")
        dst_kind = dst.get("kind")
        edge_kind = edge.get("kind")
        if edge_kind:
            if edge_kind != src_kind or edge_kind != dst_kind:
                diagnostics.add(
                    Diagnostic(
                        code="E-GRAPH-EDGE-KIND",
                        message=f"Edge kind mismatch: {from_node}.{from_port} -> {to_node}.{to_port}",
                        location="edges",
                    )
                )
        elif src_kind != dst_kind:
            diagnostics.add(
                Diagnostic(
                    code="E-GRAPH-EDGE-KIND",
                    message=f"Edge kind mismatch: {from_node}.{from_port} -> {to_node}.{to_port}",
                    location="edges",
                )
            )
        connected_ports.add((from_node, from_port))
        connected_ports.add((to_node, to_port))

    for node_id, ports in ports_by_node.items():
        for port_id, port in ports.items():
            if (node_id, port_id) in connected_ports:
                continue
            if port.get("required"):
                diagnostics.add(
                    Diagnostic(
                        code="E-GRAPH-PORT-UNCONNECTED",
                        message=f"Required port {node_id}.{port_id} is not connected",
                        location=f"nodes.{node_id}.ports.{port_id}",
                    )
                )
            else:
                diagnostics.add(
                    Diagnostic(
                        code="W-GRAPH-PORT-UNCONNECTED",
                        message=f"Port {node_id}.{port_id} is not connected",
                        severity="WARN",
                        location=f"nodes.{node_id}.ports.{port_id}",
                    )
                )

    cycle = _detect_cycle(nodes, ports_by_node, graph.get("edges", []) or [])
    if cycle:
        diagnostics.add(
            Diagnostic(
                code="E-GRAPH-CYCLE",
                message="Cycle detected in graph",
                location=" -> ".join(cycle),
            )
        )

    return diagnostics


def validate_graph_schema(graph: Dict[str, Any], schema_path: Path) -> Diagnostics:
    diagnostics = Diagnostics()
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    schema["$id"] = schema_path.resolve().as_uri()
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(graph), key=str):
        diagnostics.add(
            Diagnostic(
                code="E-GRAPH-SCHEMA",
                message=error.message,
                location="/".join(str(x) for x in error.path),
            )
        )
    return diagnostics


def toposort(graph: Dict[str, Any]) -> List[str]:
    nodes: Dict[str, Any] = {}
    for node in graph.get("nodes", []) or []:
        node_id = node.get("id")
        if not node_id:
            continue
        nodes[node_id] = node
    edges = graph.get("edges", []) or []
    ports_by_node: Dict[str, Dict[str, Any]] = {}
    for node_id, node in nodes.items():
        ports_by_node[node_id] = {port.get("id"): port for port in node.get("ports", []) or []}

    adj = defaultdict(set)
    indegree = {node_id: 0 for node_id in nodes}

    for edge in edges:
        from_ep = edge.get("from") or {}
        to_ep = edge.get("to") or {}
        src_node = from_ep.get("node")
        dst_node = to_ep.get("node")
        src_port = from_ep.get("port")
        dst_port = to_ep.get("port")
        if not src_node or not dst_node or not src_port or not dst_port:
            continue
        if src_node not in ports_by_node or dst_node not in ports_by_node:
            continue
        if src_port not in ports_by_node[src_node] or dst_port not in ports_by_node[dst_node]:
            continue
        if src_node is None or dst_node is None:
            continue
        if dst_node not in adj[src_node]:
            adj[src_node].add(dst_node)
            indegree[dst_node] += 1

    queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
    order: List[str] = []

    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for neighbor in adj[node_id]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(nodes):
        raise ValueError("Graph has cycles")

    return order


def _detect_cycle(
    nodes: Dict[str, Any], ports_by_node: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]
) -> List[str]:
    adj = defaultdict(list)
    for edge in edges:
        from_ep = edge.get("from") or {}
        to_ep = edge.get("to") or {}
        src_node = from_ep.get("node")
        dst_node = to_ep.get("node")
        src_port = from_ep.get("port")
        dst_port = to_ep.get("port")
        if not src_node or not dst_node or not src_port or not dst_port:
            continue
        if src_node not in ports_by_node or dst_node not in ports_by_node:
            continue
        if src_port not in ports_by_node[src_node] or dst_port not in ports_by_node[dst_node]:
            continue
        adj[src_node].append(dst_node)

    visited = set()
    stack = set()
    path: List[str] = []

    def visit(node_id: str) -> bool:
        visited.add(node_id)
        stack.add(node_id)
        path.append(node_id)
        for neighbor in adj.get(node_id, []):
            if neighbor not in visited:
                if visit(neighbor):
                    return True
            elif neighbor in stack:
                path.append(neighbor)
                return True
        stack.remove(node_id)
        path.pop()
        return False

    for node_id in nodes:
        if node_id not in visited:
            if visit(node_id):
                return path

    return []


def _load_data(path: Path) -> Any:
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    if path.suffix == ".toml":
        try:
            import tomllib
        except ImportError:  # pragma: no cover - python <3.11
            import tomli as tomllib  # type: ignore

        with path.open("rb") as handle:
            return tomllib.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _major(version: str) -> str:
    return version.split(".")[0] if version else ""
