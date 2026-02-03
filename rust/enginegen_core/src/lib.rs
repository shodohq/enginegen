use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const ROUND_DIGITS: i32 = 12;
const GRAPH_VERSION: &str = "1.0.0";

#[derive(Serialize)]
struct Diagnostic {
    code: String,
    message: String,
    severity: String,
    location: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct EngineSpec {
    spec_version: Option<String>,
    name: Option<String>,
    description: Option<String>,
    created_at: Option<String>,
    tags: Option<Vec<String>>,
    requirements: Option<Value>,
    constraints: Option<Value>,
    manufacturing: Option<Value>,
    analysis_budget: Option<Value>,
    metadata: Option<Value>,
    extensions: Option<Value>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GraphPort {
    id: Option<String>,
    name: Option<String>,
    kind: Option<String>,
    direction: Option<String>,
    required: Option<bool>,
    schema: Option<Value>,
    metadata: Option<Value>,
    extensions: Option<Value>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GraphNode {
    id: Option<String>,
    kind: Option<String>,
    label: Option<String>,
    enabled: Option<bool>,
    params: Option<Value>,
    #[serde(default)]
    ports: Vec<GraphPort>,
    metadata: Option<Value>,
    extensions: Option<Value>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GraphEndpoint {
    node: Option<String>,
    port: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GraphEdge {
    #[serde(rename = "from")]
    from: Option<GraphEndpoint>,
    #[serde(rename = "to")]
    to: Option<GraphEndpoint>,
    kind: Option<String>,
    label: Option<String>,
    metadata: Option<Value>,
    extensions: Option<Value>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SystemGraph {
    graph_version: Option<String>,
    #[serde(default)]
    nodes: Vec<GraphNode>,
    #[serde(default)]
    edges: Vec<GraphEdge>,
    name: Option<String>,
    description: Option<String>,
    metadata: Option<Value>,
    extensions: Option<Value>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct IrOp {
    id: Option<String>,
    op: Option<String>,
    #[serde(default)]
    inputs: Vec<String>,
    args: Option<Value>,
    enabled: Option<bool>,
    tags: Option<Value>,
    debug: Option<Value>,
    source: Option<Value>,
    extensions: Option<Value>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GeometryIr {
    ir_version: Option<String>,
    dialect: Option<String>,
    units: Option<Map<String, Value>>,
    params: Option<Map<String, Value>>,
    #[serde(default)]
    ops: Vec<IrOp>,
    outputs: Option<Map<String, Value>>,
    annotations: Option<Value>,
    #[serde(default)]
    checks: Vec<Value>,
    domain: Option<Value>,
    metadata: Option<Value>,
    extensions: Option<Value>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ArtifactRecord {
    kind: Option<String>,
    path: Option<String>,
    content_hash: Option<String>,
    size_bytes: Option<i64>,
    mime_type: Option<String>,
    created_at: Option<String>,
    producer: Option<Value>,
    metadata: Option<Value>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ArtifactManifest {
    manifest_version: Option<String>,
    run_id: Option<String>,
    created_at: Option<String>,
    status: Option<String>,
    host: Option<Value>,
    inputs: Option<Value>,
    #[serde(default)]
    plugins: Vec<Value>,
    #[serde(default)]
    artifacts: Vec<ArtifactRecord>,
    metrics: Option<Value>,
    diagnostics: Option<Value>,
    environment: Option<Value>,
    notes: Option<String>,
    extensions: Option<Value>,
}

pub fn normalize_spec(input: &[u8]) -> Result<Vec<u8>, String> {
    canonicalize_json(input)
}

pub fn normalize_ir(input: &[u8]) -> Result<Vec<u8>, String> {
    canonicalize_json(input)
}

pub fn hash_canonical(input: &[u8]) -> Result<String, String> {
    let payload = canonicalize_json(input)?;
    let mut hasher = Sha256::new();
    hasher.update(payload);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

pub fn validate_graph(input: &[u8]) -> Result<Vec<u8>, String> {
    let graph: SystemGraph = serde_json::from_slice(input).map_err(|err| err.to_string())?;
    let mut diags: Vec<Diagnostic> = Vec::new();

    match graph.graph_version.as_deref() {
        Some(version) => {
            if major(version) != major(GRAPH_VERSION) {
                diags.push(Diagnostic {
                    code: "E-GRAPH-VERSION".to_string(),
                    message: format!(
                        "Graph version mismatch: expected {}, got {}",
                        GRAPH_VERSION, version
                    ),
                    severity: "ERROR".to_string(),
                    location: Some("graph_version".to_string()),
                });
            }
        }
        None => diags.push(Diagnostic {
            code: "E-GRAPH-VERSION".to_string(),
            message: "Missing graph_version".to_string(),
            severity: "ERROR".to_string(),
            location: Some("graph_version".to_string()),
        }),
    }

    let mut node_index: HashMap<String, usize> = HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        match node.id.as_deref() {
            Some(id) => {
                if node_index.contains_key(id) {
                    diags.push(Diagnostic {
                        code: "E-GRAPH-NODE-DUP".to_string(),
                        message: "Duplicate node id detected".to_string(),
                        severity: "ERROR".to_string(),
                        location: Some("nodes".to_string()),
                    });
                } else {
                    node_index.insert(id.to_string(), idx);
                }
            }
            None => diags.push(Diagnostic {
                code: "E-GRAPH-NODE-ID".to_string(),
                message: "Node id missing".to_string(),
                severity: "ERROR".to_string(),
                location: Some("nodes".to_string()),
            }),
        }
    }

    let port_kinds: HashSet<&str> =
        ["fluid", "thermal", "structural", "electrical", "control", "data"]
            .iter()
            .copied()
            .collect();
    let port_dirs: HashSet<&str> = ["in", "out", "bidir"].iter().copied().collect();

    let mut ports_by_node: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let Some(node_id) = node.id.as_deref() else { continue; };
        let mut port_ids: HashSet<String> = HashSet::new();
        for (port_idx, port) in node.ports.iter().enumerate() {
            match port.id.as_deref() {
                Some(port_id) => {
                    if !port_ids.insert(port_id.to_string()) {
                        diags.push(Diagnostic {
                            code: "E-GRAPH-PORT-DUP".to_string(),
                            message: format!("Duplicate port id '{}' on node {}", port_id, node_id),
                            severity: "ERROR".to_string(),
                            location: Some(format!("nodes.{}.ports", node_id)),
                        });
                    } else {
                        ports_by_node
                            .entry(node_id.to_string())
                            .or_default()
                            .insert(port_id.to_string(), port_idx);
                    }
                }
                None => diags.push(Diagnostic {
                    code: "E-GRAPH-PORT-ID".to_string(),
                    message: format!("Port id missing for node {}", node_id),
                    severity: "ERROR".to_string(),
                    location: Some(format!("nodes.{}.ports", node_id)),
                }),
            }

            if let Some(kind) = port.kind.as_deref() {
                if !port_kinds.contains(kind) {
                    diags.push(Diagnostic {
                        code: "E-GRAPH-PORT-KIND".to_string(),
                        message: format!("Unknown port kind '{}' for {}", kind, node_id),
                        severity: "ERROR".to_string(),
                        location: Some(format!("nodes.{}.ports", node_id)),
                    });
                }
            }
            if let Some(dir) = port.direction.as_deref() {
                if !port_dirs.contains(dir) {
                    diags.push(Diagnostic {
                        code: "E-GRAPH-PORT-DIR".to_string(),
                        message: format!("Unknown port direction '{}' for {}", dir, node_id),
                        severity: "ERROR".to_string(),
                        location: Some(format!("nodes.{}.ports", node_id)),
                    });
                }
            }
        }
    }

    let mut connected_ports: HashSet<(String, String)> = HashSet::new();
    for edge in &graph.edges {
        let Some(from_ep) = edge.from.as_ref() else {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-PORT".to_string(),
                message: "Edge missing from endpoint".to_string(),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
            continue;
        };
        let Some(to_ep) = edge.to.as_ref() else {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-PORT".to_string(),
                message: "Edge missing to endpoint".to_string(),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
            continue;
        };
        let (Some(from_node), Some(from_port)) = (from_ep.node.as_deref(), from_ep.port.as_deref()) else {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-PORT".to_string(),
                message: "Edge missing from endpoint".to_string(),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
            continue;
        };
        let (Some(to_node), Some(to_port)) = (to_ep.node.as_deref(), to_ep.port.as_deref()) else {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-PORT".to_string(),
                message: "Edge missing to endpoint".to_string(),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
            continue;
        };

        let Some(src_ports) = ports_by_node.get(from_node) else {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-PORT".to_string(),
                message: format!("Edge references unknown port: {}.{} -> {}.{}", from_node, from_port, to_node, to_port),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
            continue;
        };
        let Some(dst_ports) = ports_by_node.get(to_node) else {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-PORT".to_string(),
                message: format!("Edge references unknown port: {}.{} -> {}.{}", from_node, from_port, to_node, to_port),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
            continue;
        };
        let (Some(src_idx), Some(dst_idx)) = (src_ports.get(from_port), dst_ports.get(to_port)) else {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-PORT".to_string(),
                message: format!("Edge references unknown port: {}.{} -> {}.{}", from_node, from_port, to_node, to_port),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
            continue;
        };

        let src_node_idx = *node_index.get(from_node).unwrap_or(&0);
        let dst_node_idx = *node_index.get(to_node).unwrap_or(&0);
        let src = &graph.nodes[src_node_idx].ports[*src_idx];
        let dst = &graph.nodes[dst_node_idx].ports[*dst_idx];
        let src_dir = src.direction.as_deref().unwrap_or("");
        let dst_dir = dst.direction.as_deref().unwrap_or("");
        if (src_dir != "out" && src_dir != "bidir") || (dst_dir != "in" && dst_dir != "bidir") {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-DIR".to_string(),
                message: format!("Edge direction mismatch: {}.{} -> {}.{}", from_node, from_port, to_node, to_port),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
        }
        if let Some(edge_kind) = edge.kind.as_deref() {
            if src.kind.as_deref() != Some(edge_kind) || dst.kind.as_deref() != Some(edge_kind) {
                diags.push(Diagnostic {
                    code: "E-GRAPH-EDGE-KIND".to_string(),
                    message: format!("Edge kind mismatch: {}.{} -> {}.{}", from_node, from_port, to_node, to_port),
                    severity: "ERROR".to_string(),
                    location: Some("edges".to_string()),
                });
            }
        } else if src.kind != dst.kind {
            diags.push(Diagnostic {
                code: "E-GRAPH-EDGE-KIND".to_string(),
                message: format!("Edge kind mismatch: {}.{} -> {}.{}", from_node, from_port, to_node, to_port),
                severity: "ERROR".to_string(),
                location: Some("edges".to_string()),
            });
        }
        connected_ports.insert((from_node.to_string(), from_port.to_string()));
        connected_ports.insert((to_node.to_string(), to_port.to_string()));
    }

    for (node_id, ports) in &ports_by_node {
        let node_idx = *node_index.get(node_id).unwrap_or(&0);
        for (port_id, port_idx) in ports {
            let port = &graph.nodes[node_idx].ports[*port_idx];
            if port.required.unwrap_or(false)
                && !connected_ports.contains(&(node_id.to_string(), port_id.to_string()))
            {
                diags.push(Diagnostic {
                    code: "E-GRAPH-PORT-UNCONNECTED".to_string(),
                    message: format!("Required port {}.{} is not connected", node_id, port_id),
                    severity: "ERROR".to_string(),
                    location: Some(format!("nodes.{}.ports.{}", node_id, port_id)),
                });
            }
        }
    }

    if detect_cycle(&graph, &node_index, &ports_by_node) {
        diags.push(Diagnostic {
            code: "E-GRAPH-CYCLE".to_string(),
            message: "Cycle detected in graph".to_string(),
            severity: "ERROR".to_string(),
            location: Some("nodes".to_string()),
        });
    }

    serde_json::to_vec(&diags).map_err(|err| err.to_string())
}

pub fn schema_graph() -> Result<Vec<u8>, String> {
    let schema = schemars::schema_for!(SystemGraph);
    serde_json::to_vec(&schema).map_err(|err| err.to_string())
}

pub fn schema_ir() -> Result<Vec<u8>, String> {
    let schema = schemars::schema_for!(GeometryIr);
    serde_json::to_vec(&schema).map_err(|err| err.to_string())
}

pub fn schema_manifest() -> Result<Vec<u8>, String> {
    let schema = schemars::schema_for!(ArtifactManifest);
    serde_json::to_vec(&schema).map_err(|err| err.to_string())
}

pub fn schema_spec() -> Result<Vec<u8>, String> {
    let schema = schemars::schema_for!(EngineSpec);
    serde_json::to_vec(&schema).map_err(|err| err.to_string())
}

fn detect_cycle(
    graph: &SystemGraph,
    node_index: &HashMap<String, usize>,
    ports_by_node: &HashMap<String, HashMap<String, usize>>,
) -> bool {
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for edge in &graph.edges {
        let (Some(from_ep), Some(to_ep)) = (edge.from.as_ref(), edge.to.as_ref()) else {
            continue;
        };
        let (Some(src_node), Some(src_port)) = (from_ep.node.as_deref(), from_ep.port.as_deref())
        else {
            continue;
        };
        let (Some(dst_node), Some(dst_port)) = (to_ep.node.as_deref(), to_ep.port.as_deref())
        else {
            continue;
        };
        let Some(src_ports) = ports_by_node.get(src_node) else { continue; };
        let Some(dst_ports) = ports_by_node.get(dst_node) else { continue; };
        if !src_ports.contains_key(src_port) || !dst_ports.contains_key(dst_port) {
            continue;
        }
        if !node_index.contains_key(src_node) || !node_index.contains_key(dst_node) {
            continue;
        }
        adjacency
            .entry(src_node.to_string())
            .or_default()
            .push(dst_node.to_string());
    }

    let mut visited: HashSet<String> = HashSet::new();
    let mut stack: HashSet<String> = HashSet::new();

    for node in &graph.nodes {
        let Some(node_id) = node.id.as_deref() else { continue; };
        if !visited.contains(node_id) {
            if visit(node_id, &adjacency, &mut visited, &mut stack) {
                return true;
            }
        }
    }
    false
}

fn visit(
    node_id: &str,
    adjacency: &HashMap<String, Vec<String>>,
    visited: &mut HashSet<String>,
    stack: &mut HashSet<String>,
) -> bool {
    visited.insert(node_id.to_string());
    stack.insert(node_id.to_string());
    if let Some(neighbors) = adjacency.get(node_id) {
        for neighbor in neighbors {
            if !visited.contains(neighbor) {
                if visit(neighbor, adjacency, visited, stack) {
                    return true;
                }
            } else if stack.contains(neighbor) {
                return true;
            }
        }
    }
    stack.remove(node_id);
    false
}

fn major(version: &str) -> &str {
    version.split('.').next().unwrap_or("")
}

fn canonicalize_json(input: &[u8]) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_slice(input).map_err(|err| err.to_string())?;
    let canonical = canonicalize_value(&value);
    serde_json::to_vec(&canonical).map_err(|err| err.to_string())
}

fn canonicalize_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut new_map = Map::new();
            for key in keys {
                if let Some(child) = map.get(key) {
                    new_map.insert(key.clone(), canonicalize_value(child));
                }
            }
            Value::Object(new_map)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize_value).collect()),
        Value::Number(num) => {
            if num.is_i64() || num.is_u64() {
                return Value::Number(num.clone());
            }
            if let Some(f) = num.as_f64() {
                let rounded = round_to_digits(f);
                if let Some(new_num) = serde_json::Number::from_f64(rounded) {
                    return Value::Number(new_num);
                }
            }
            Value::Number(num.clone())
        }
        _ => value.clone(),
    }
}

fn round_to_digits(value: f64) -> f64 {
    let factor = 10f64.powi(ROUND_DIGITS);
    let rounded = (value * factor).round() / factor;
    if rounded == -0.0 {
        0.0
    } else {
        rounded
    }
}
