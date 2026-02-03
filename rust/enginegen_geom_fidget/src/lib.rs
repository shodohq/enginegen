use fidget::context::{Context, Tree};
use fidget::mesh::{Octree, Settings};
use fidget::vm::VmShape;
use fidget::jit::JitShape;
use fidget_core::render::{Bounds, ThreadPool};
use fidget_shapes::types::{Vec2, Vec3};
use fidget_shapes::{
    Blend, Box as FBox, Circle, Difference, ExtrudeZ, Intersection, Inverse, LoftZ, Move,
    ReflectX, ReflectY, ReflectZ, RevolveY, RotateX, RotateY, RotateZ, Scale, ScaleUniform,
    Sphere, Union, Rectangle,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::time::Instant;
use std::io::Write;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MeshConfig {
    pub depth: u8,
    pub max_tris: usize,
    pub parallel: bool,
    pub deterministic_order: bool,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            depth: 6,
            max_tris: 2_000_000,
            parallel: true,
            deterministic_order: true,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ExportConfig {
    pub stl: bool,
    pub obj: bool,
    pub ply: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            stl: true,
            obj: false,
            ply: false,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    pub bytecode: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { bytecode: false }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub evaluation_engine: String,
    pub domain_bbox: Option<[[f64; 3]; 2]>,
    pub mesh: MeshConfig,
    pub exports: ExportConfig,
    pub cache: CacheConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            evaluation_engine: "auto".to_string(),
            domain_bbox: None,
            mesh: MeshConfig::default(),
            exports: ExportConfig::default(),
            cache: CacheConfig::default(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub severity: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MeshResult {
    pub stl_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub obj_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ply_path: Option<String>,
    pub metrics: HashMap<String, Value>,
    pub diagnostics: Vec<Diagnostic>,
    pub engine: String,
}

#[derive(Debug, Deserialize, Clone)]
struct GeometryIr {
    ir_version: Option<String>,
    dialect: Option<String>,
    units: Option<HashMap<String, String>>,
    params: Option<HashMap<String, Value>>,
    ops: Option<Vec<IrOp>>,
    nodes: Option<Vec<IrOp>>,
    outputs: HashMap<String, String>,
    metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, Deserialize, Clone)]
struct IrOp {
    id: String,
    op: String,
    #[serde(default)]
    inputs: Vec<String>,
    args: Option<Value>,
    params: Option<Value>,
    enabled: Option<bool>,
}

impl GeometryIr {
    fn ops(&self) -> Vec<IrOp> {
        if let Some(ops) = self.ops.as_ref() {
            return ops.clone();
        }
        if let Some(nodes) = self.nodes.as_ref() {
            return nodes.clone();
        }
        Vec::new()
    }
}

#[derive(Debug, Serialize)]
pub struct StructuredError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_id: Option<String>,
}

impl StructuredError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            hint: None,
            op_id: None,
        }
    }

    fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    fn with_op(mut self, op_id: impl Into<String>) -> Self {
        self.op_id = Some(op_id.into());
        self
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            format!(
                "{{\"code\":\"{}\",\"message\":\"{}\"}}",
                self.code, self.message
            )
        })
    }
}

impl std::fmt::Display for StructuredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_json())
    }
}

impl std::error::Error for StructuredError {}

type GeomResult<T> = Result<T, StructuredError>;

pub fn compile_and_mesh(ir_json: &str, config_json: &str, out_dir: &Path) -> GeomResult<MeshResult> {
    let config: Config = serde_json::from_str(config_json)
        .map_err(|err| StructuredError::new("E-CONFIG-JSON", format!("invalid config json: {err}")))?;
    let bbox = config
        .domain_bbox
        .ok_or_else(|| StructuredError::new("E-DOMAIN-BBOX", "domain_bbox is required").with_hint("Provide domain_bbox in config or IR metadata"))?;

    let ir: GeometryIr = serde_json::from_str(ir_json)
        .map_err(|err| StructuredError::new("E-IR-JSON", format!("invalid geometry ir json: {err}")))?;

    validate_dialect(&ir)?;

    let unit_scale = length_unit_scale(&ir)?;
    let bbox = scale_bbox(bbox, unit_scale)?;

    let compiled = compile_ir(&ir, unit_scale)?;

    let (engine, mut diagnostics) = select_engine(&config.evaluation_engine)?;
    diagnostics.push(Diagnostic {
        code: "W-MESH-LIMITATIONS".to_string(),
        message: "fidget-mesh may miss thin features or produce self-intersections; validate results before manufacturing.".to_string(),
        severity: "WARN".to_string(),
        op_id: None,
    });
    if config.cache.bytecode {
        diagnostics.push(Diagnostic {
            code: "I-BYTECODE-CACHE-UNUSED".to_string(),
            message: "bytecode cache requested but not implemented; proceeding without cache.".to_string(),
            severity: "INFO".to_string(),
            op_id: None,
        });
    }

    let mut settings = Settings::default();
    settings.depth = config.mesh.depth;
    if config.mesh.parallel {
        settings.threads = Some(ThreadPool::Global);
    } else {
        settings.threads = None;
    }
    settings.bounds = bounds_from_bbox(bbox)?;

    let start = Instant::now();
    let (mesh, octree_nodes) = match engine {
        EngineChoice::Vm => {
            let shape = VmShape::new(&compiled.context, compiled.root)
                .map_err(|err| StructuredError::new("E-SHAPE", format!("failed to build vm shape: {err}")))?;
            let octree = Octree::build(&shape, &settings)
                .ok_or_else(|| StructuredError::new("E-MESH-OCTREE", "failed to build octree"))?;
            let mesh = octree.walk_dual();
            (mesh, octree_node_count(&octree))
        }
        EngineChoice::Jit => {
            let shape = JitShape::new(&compiled.context, compiled.root)
                .map_err(|err| StructuredError::new("E-SHAPE", format!("failed to build jit shape: {err}")))?;
            let octree = Octree::build(&shape, &settings)
                .ok_or_else(|| StructuredError::new("E-MESH-OCTREE", "failed to build octree"))?;
            let mesh = octree.walk_dual();
            (mesh, octree_node_count(&octree))
        }
    };
    let elapsed = start.elapsed().as_secs_f64();

    let tri_count = mesh.tris.len();
    if tri_count > config.mesh.max_tris {
        return Err(
            StructuredError::new(
                "E-MESH-LIMIT",
                format!("mesh triangle limit exceeded: {} > {}", tri_count, config.mesh.max_tris),
            )
            .with_hint("Reduce mesh depth or increase max_tris"),
        );
    }

    std::fs::create_dir_all(out_dir)
        .map_err(|err| StructuredError::new("E-IO", format!("failed to create output dir: {err}")))?;
    let stl_path = out_dir.join("geometry.mesh.stl");
    let (stl_bytes, stl_tris) = mesh_to_stl_bytes(&mesh, config.mesh.deterministic_order, &mut diagnostics)?;
    let mut handle = std::fs::File::create(&stl_path)
        .map_err(|err| StructuredError::new("E-IO", format!("failed to create stl: {err}")))?;
    handle
        .write_all(&stl_bytes)
        .map_err(|err| StructuredError::new("E-IO", format!("failed to write stl: {err}")))?;

    let mut obj_path = None;
    let mut ply_path = None;
    if config.exports.obj {
        let path = out_dir.join("geometry.mesh.obj");
        write_obj(&path, &stl_tris)?;
        obj_path = Some(path.to_string_lossy().to_string());
    }
    if config.exports.ply {
        let path = out_dir.join("geometry.mesh.ply");
        write_ply(&path, &stl_tris)?;
        ply_path = Some(path.to_string_lossy().to_string());
    }

    let mut metrics = HashMap::new();
    metrics.insert("mesh.triangles".to_string(), Value::from(tri_count as u64));
    metrics.insert("mesh.seconds".to_string(), Value::from(elapsed));
    if let Some(nodes) = octree_nodes {
        metrics.insert("mesh.octree_nodes".to_string(), Value::from(nodes as u64));
    }

    Ok(MeshResult {
        stl_path: stl_path.to_string_lossy().to_string(),
        obj_path,
        ply_path,
        metrics,
        diagnostics,
        engine: match engine {
            EngineChoice::Vm => "vm".to_string(),
            EngineChoice::Jit => "jit".to_string(),
        },
    })
}

struct CompiledShape {
    context: Context,
    root: fidget::context::Node,
}

fn validate_dialect(ir: &GeometryIr) -> GeomResult<()> {
    if let Some(dialect) = ir.dialect.as_deref() {
        if dialect.is_empty() {
            return Ok(());
        }
        if dialect != "enginegen.implicit.fidget.v1" && dialect != "builtin.geometry.fidget" {
            return Err(
                StructuredError::new("E-DIALECT", format!("unsupported dialect: {dialect}"))
                    .with_hint("Use enginegen.implicit.fidget.v1"),
            );
        }
    }
    Ok(())
}

fn compile_ir(ir: &GeometryIr, scale: f64) -> GeomResult<CompiledShape> {
    let ops = ir.ops();
    let params = ir.params.clone().unwrap_or_default();
    let enabled_ops: Vec<IrOp> = ops
        .into_iter()
        .filter(|op| op.enabled.unwrap_or(true))
        .collect();
    let mut op_map: HashMap<String, IrOp> = HashMap::new();
    for op in enabled_ops {
        op_map.insert(op.id.clone(), op);
    }

    let order = topo_sort(&op_map)?;
    let mut compiled: HashMap<String, Tree> = HashMap::new();
    for op_id in order {
        let node = op_map
            .get(&op_id)
            .ok_or_else(|| StructuredError::new("E-IR-OP", format!("missing op {op_id}")))?;
        let tree = compile_node(node, &compiled, &params, scale)?;
        compiled.insert(op_id, tree);
    }

    let out_id = if let Some(main) = ir.outputs.get("main") {
        main
    } else if let Some(solid) = ir.outputs.get("solid") {
        solid
    } else {
        return Err(StructuredError::new("E-IR-OUTPUTS", "outputs.main missing")
            .with_hint("Provide outputs.main (or outputs.solid for compatibility)"));
    };
    let tree = compiled
        .get(out_id)
        .ok_or_else(|| StructuredError::new("E-IR-OUTPUTS", format!("outputs refers to missing op {out_id}")))?
        .clone();

    let mut context = Context::new();
    let root = context.import(&tree);
    Ok(CompiledShape { context, root })
}

fn topo_sort(ops: &HashMap<String, IrOp>) -> GeomResult<Vec<String>> {
    let mut indegree: HashMap<String, usize> = HashMap::new();
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for (id, node) in ops {
        indegree.insert(id.clone(), 0);
        for input in &node.inputs {
            if ops.contains_key(input) {
                *indegree.entry(id.clone()).or_default() += 1;
                adj.entry(input.clone()).or_default().push(id.clone());
            }
        }
    }

    let mut queue: VecDeque<String> = indegree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut order = Vec::new();
    while let Some(id) = queue.pop_front() {
        order.push(id.clone());
        if let Some(neigh) = adj.get(&id) {
            for next in neigh {
                if let Some(entry) = indegree.get_mut(next) {
                    *entry -= 1;
                    if *entry == 0 {
                        queue.push_back(next.clone());
                    }
                }
            }
        }
    }
    if order.len() != ops.len() {
        return Err(StructuredError::new("E-IR-CYCLE", "cycle detected in IR ops"));
    }
    Ok(order)
}

fn compile_node(node: &IrOp, compiled: &HashMap<String, Tree>, params: &HashMap<String, Value>, scale: f64) -> GeomResult<Tree> {
    let op = normalize_op(&node.op);
    let args = node_args(node);
    match op {
        "prim.sphere" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "prim.sphere missing args", "Provide center and radius"))?;
            let center = parse_vec3(args, "center", node, scale)?;
            let radius = parse_number_or_param(args, "radius", params, node, scale)?;
            Ok(Tree::from(Sphere { center, radius }))
        }
        "prim.box" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "prim.box missing args", "Provide lower and upper"))?;
            let lower = parse_vec3(args, "lower", node, scale)?;
            let upper = parse_vec3(args, "upper", node, scale)?;
            Ok(Tree::from(FBox { lower, upper }))
        }
        "prim.circle" | "prim.circle2d" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "prim.circle missing args", "Provide center and radius"))?;
            let center = parse_vec2(args, "center", node, scale)?;
            let radius = parse_number_or_param(args, "radius", params, node, scale)?;
            Ok(Tree::from(Circle { center, radius }))
        }
        "prim.rectangle" | "prim.rect2d" | "prim.rect" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "prim.rectangle missing args", "Provide lower and upper"))?;
            let lower = parse_vec2(args, "lower", node, scale)?;
            let upper = parse_vec2(args, "upper", node, scale)?;
            Ok(Tree::from(Rectangle { lower, upper }))
        }
        "csg.union" => {
            let inputs = require_inputs(node, compiled, 1, None)?;
            Ok(Tree::from(Union { input: inputs }))
        }
        "csg.intersection" => {
            let inputs = require_inputs(node, compiled, 1, None)?;
            Ok(Tree::from(Intersection { input: inputs }))
        }
        "csg.difference" => {
            let mut inputs = require_inputs(node, compiled, 2, Some(2))?;
            let shape = inputs.remove(0);
            let cutout = inputs.remove(0);
            Ok(Tree::from(Difference { shape, cutout }))
        }
        "csg.inverse" => {
            let mut inputs = require_inputs(node, compiled, 1, Some(1))?;
            let shape = inputs.remove(0);
            Ok(Tree::from(Inverse { shape }))
        }
        "csg.blend_quadratic" | "csg.blend" | "ops.blend" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "blend missing args", "Provide radius"))?;
            let radius = parse_number_or_param(args, "radius", params, node, scale)?;
            let mut inputs = require_inputs(node, compiled, 2, Some(2))?;
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            Ok(Tree::from(Blend { a, b, radius }))
        }
        "xform.move" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.move missing args", "Provide offset"))?;
            let offset = parse_vec3(args, "offset", node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(Move { shape, offset }))
        }
        "xform.scale" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.scale missing args", "Provide scale"))?;
            let scale_vec = parse_vec3(args, "scale", node, 1.0)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(Scale { shape, scale: scale_vec }))
        }
        "xform.scale_uniform" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.scale_uniform missing args", "Provide scale"))?;
            let scale_val = parse_number_or_param(args, "scale", params, node, 1.0)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(ScaleUniform { shape, scale: scale_val }))
        }
        "xform.rotate_x" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.rotate_x missing args", "Provide angle_deg and center"))?;
            let angle = parse_number_or_param(args, "angle_deg", params, node, 1.0)?;
            let center = parse_vec3(args, "center", node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(RotateX { shape, angle, center }))
        }
        "xform.rotate_y" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.rotate_y missing args", "Provide angle_deg and center"))?;
            let angle = parse_number_or_param(args, "angle_deg", params, node, 1.0)?;
            let center = parse_vec3(args, "center", node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(RotateY { shape, angle, center }))
        }
        "xform.rotate_z" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.rotate_z missing args", "Provide angle_deg and center"))?;
            let angle = parse_number_or_param(args, "angle_deg", params, node, 1.0)?;
            let center = parse_vec3(args, "center", node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(RotateZ { shape, angle, center }))
        }
        "xform.reflect_x" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.reflect_x missing args", "Provide offset"))?;
            let offset = parse_number_or_param(args, "offset", params, node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(ReflectX { shape, offset }))
        }
        "xform.reflect_y" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.reflect_y missing args", "Provide offset"))?;
            let offset = parse_number_or_param(args, "offset", params, node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(ReflectY { shape, offset }))
        }
        "xform.reflect_z" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "xform.reflect_z missing args", "Provide offset"))?;
            let offset = parse_number_or_param(args, "offset", params, node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(ReflectZ { shape, offset }))
        }
        "solid.extrude_z" | "gen.extrude_z" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "extrude_z missing args", "Provide lower and upper"))?;
            let lower = parse_number_or_param(args, "lower", params, node, scale)?;
            let upper = parse_number_or_param(args, "upper", params, node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(ExtrudeZ { shape, lower, upper }))
        }
        "solid.revolve_y" | "gen.revolve_y" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "revolve_y missing args", "Provide offset"))?;
            let offset = parse_number_or_param(args, "offset", params, node, scale)?;
            let shape = require_single(node, compiled)?;
            Ok(Tree::from(RevolveY { shape, offset }))
        }
        "solid.loft_z" | "gen.loft_z" => {
            let args = args.ok_or_else(|| err_op(node, "E-OP-ARGS", "loft_z missing args", "Provide lower and upper"))?;
            let lower = parse_number_or_param(args, "lower", params, node, scale)?;
            let upper = parse_number_or_param(args, "upper", params, node, scale)?;
            let mut inputs = require_inputs(node, compiled, 2, Some(2))?;
            let a = inputs.remove(0);
            let b = inputs.remove(0);
            Ok(Tree::from(LoftZ { a, b, lower, upper }))
        }
        _ => Err(err_op(node, "E-OP-UNSUPPORTED", format!("unsupported op: {}", node.op), "Check dialect op list")),
    }
}

fn normalize_op(op: &str) -> &str {
    if let Some(rest) = op.strip_prefix("fidget.") {
        return rest;
    }
    op
}

fn node_args(node: &IrOp) -> Option<&Value> {
    node.args.as_ref().or(node.params.as_ref())
}

fn require_single(node: &IrOp, compiled: &HashMap<String, Tree>) -> GeomResult<Tree> {
    let mut inputs = require_inputs(node, compiled, 1, Some(1))?;
    Ok(inputs.remove(0))
}

fn require_inputs(
    node: &IrOp,
    compiled: &HashMap<String, Tree>,
    min: usize,
    max: Option<usize>,
) -> GeomResult<Vec<Tree>> {
    let count = node.inputs.len();
    if count < min {
        return Err(err_op(
            node,
            "E-OP-INPUTS",
            format!("{} expects at least {} inputs", node.op, min),
            "Provide required inputs",
        ));
    }
    if let Some(max) = max {
        if count > max {
            return Err(err_op(
                node,
                "E-OP-INPUTS",
                format!("{} expects at most {} inputs", node.op, max),
                "Remove extra inputs",
            ));
        }
    }
    let mut out = Vec::with_capacity(count);
    for input in &node.inputs {
        let tree = compiled
            .get(input)
            .ok_or_else(|| err_op(
                node,
                "E-OP-INPUTS",
                format!("{} references missing input {}", node.op, input),
                "Check input ids",
            ))?
            .clone();
        out.push(tree);
    }
    Ok(out)
}

fn parse_vec3(args: &Value, key: &str, node: &IrOp, scale: f64) -> GeomResult<Vec3> {
    let arr = args
        .get(key)
        .ok_or_else(|| err_op(node, "E-OP-ARGS", format!("missing {key}"), "Provide required args"))?
        .as_array()
        .ok_or_else(|| err_op(node, "E-OP-ARGS", format!("{key} must be array"), "Provide array values"))?;
    if arr.len() != 3 {
        return Err(err_op(node, "E-OP-ARGS", format!("{key} must have 3 elements"), "Provide 3 elements"));
    }
    Ok(Vec3::new(
        arr[0].as_f64().ok_or_else(|| err_op(node, "E-OP-ARGS", format!("{key}[0] must be number"), "Provide numeric values"))? * scale,
        arr[1].as_f64().ok_or_else(|| err_op(node, "E-OP-ARGS", format!("{key}[1] must be number"), "Provide numeric values"))? * scale,
        arr[2].as_f64().ok_or_else(|| err_op(node, "E-OP-ARGS", format!("{key}[2] must be number"), "Provide numeric values"))? * scale,
    ))
}

fn parse_vec2(args: &Value, key: &str, node: &IrOp, scale: f64) -> GeomResult<Vec2> {
    let arr = args
        .get(key)
        .ok_or_else(|| err_op(node, "E-OP-ARGS", format!("missing {key}"), "Provide required args"))?
        .as_array()
        .ok_or_else(|| err_op(node, "E-OP-ARGS", format!("{key} must be array"), "Provide array values"))?;
    if arr.len() != 2 {
        return Err(err_op(node, "E-OP-ARGS", format!("{key} must have 2 elements"), "Provide 2 elements"));
    }
    Ok(Vec2::new(
        arr[0].as_f64().ok_or_else(|| err_op(node, "E-OP-ARGS", format!("{key}[0] must be number"), "Provide numeric values"))? * scale,
        arr[1].as_f64().ok_or_else(|| err_op(node, "E-OP-ARGS", format!("{key}[1] must be number"), "Provide numeric values"))? * scale,
    ))
}

fn parse_number_or_param(args: &Value, key: &str, params: &HashMap<String, Value>, node: &IrOp, scale: f64) -> GeomResult<f64> {
    let value = args
        .get(key)
        .ok_or_else(|| err_op(node, "E-OP-ARGS", format!("missing {key}"), "Provide required args"))?;
    if let Some(num) = value.as_f64() {
        return Ok(num * scale);
    }
    if let Some(obj) = value.as_object() {
        if let Some(param_name) = obj.get("$param").and_then(|v| v.as_str()) {
            return param_value(params, param_name)
                .map(|v| v * scale)
                .ok_or_else(|| err_op(node, "E-OP-PARAM", format!("param {param_name} not found"), "Check params map"));
        }
        if let Some(num) = obj.get("value").and_then(|v| v.as_f64()) {
            return Ok(num * scale);
        }
    }
    Err(err_op(node, "E-OP-ARGS", format!("{key} must be number or {{$param:..}}"), "Provide numeric value or $param"))
}

fn param_value(params: &HashMap<String, Value>, name: &str) -> Option<f64> {
    let value = params.get(name)?;
    if let Some(num) = value.as_f64() {
        return Some(num);
    }
    if let Some(obj) = value.as_object() {
        if let Some(num) = obj.get("value").and_then(|v| v.as_f64()) {
            return Some(num);
        }
        if let Some(num) = obj.get("value").and_then(|v| v.as_str()) {
            return num.parse::<f64>().ok();
        }
    }
    None
}

fn bounds_from_bbox(bbox: [[f64; 3]; 2]) -> GeomResult<Bounds> {
    validate_bbox(bbox)?;
    let min = fidget_core::types::Vec3::new(
        bbox[0][0] as f32,
        bbox[0][1] as f32,
        bbox[0][2] as f32,
    );
    let max = fidget_core::types::Vec3::new(
        bbox[1][0] as f32,
        bbox[1][1] as f32,
        bbox[1][2] as f32,
    );
    Ok(Bounds::new(min, max))
}

#[derive(Debug, Clone, Copy)]
enum EngineChoice {
    Vm,
    Jit,
}

fn select_engine(mode: &str) -> GeomResult<(EngineChoice, Vec<Diagnostic>)> {
    let mut diags = Vec::new();
    let normalized = mode.to_lowercase();
    match normalized.as_str() {
        "vm" => Ok((EngineChoice::Vm, diags)),
        "jit" => {
            if jit_supported() {
                Ok((EngineChoice::Jit, diags))
            } else {
                return Err(
                    StructuredError::new("E-JIT-UNSUPPORTED", "jit requested but not supported on this platform")
                        .with_hint("Use evaluation_engine=vm or run on AVX2/NEON-capable hardware"),
                );
            }
        }
        "auto" => {
            if jit_supported() {
                diags.push(Diagnostic {
                    code: "I-JIT-ENABLED".to_string(),
                    message: "jit enabled by auto selection".to_string(),
                    severity: "INFO".to_string(),
                    op_id: None,
                });
                Ok((EngineChoice::Jit, diags))
            } else {
                diags.push(Diagnostic {
                    code: "I-VM-FALLBACK".to_string(),
                    message: "jit unavailable; using vm".to_string(),
                    severity: "INFO".to_string(),
                    op_id: None,
                });
                Ok((EngineChoice::Vm, diags))
            }
        }
        _ => Err(StructuredError::new("E-CONFIG", format!("unknown evaluation_engine: {mode}"))),
    }
}

fn jit_supported() -> bool {
    if cfg!(target_arch = "x86_64") {
        std::is_x86_feature_detected!("avx2")
            && (cfg!(target_os = "linux")
                || cfg!(target_os = "windows")
                || cfg!(target_os = "macos"))
    } else if cfg!(target_arch = "aarch64") {
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("neon")
                && (cfg!(target_os = "linux")
                    || cfg!(target_os = "windows")
                    || cfg!(target_os = "macos"))
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    } else {
        false
    }
}

fn err_op(node: &IrOp, code: &str, message: impl Into<String>, hint: impl Into<String>) -> StructuredError {
    StructuredError::new(code, message).with_hint(hint).with_op(node.id.clone())
}

fn length_unit_scale(ir: &GeometryIr) -> GeomResult<f64> {
    let units = ir.units.as_ref();
    let Some(units) = units else {
        return Ok(1.0);
    };
    let unit = units.get("length").map(|s| s.as_str()).unwrap_or("m");
    let normalized = unit.to_lowercase();
    let scale = match normalized.as_str() {
        "m" | "meter" | "meters" => 1.0,
        "cm" | "centimeter" | "centimeters" => 0.01,
        "mm" | "millimeter" | "millimeters" => 0.001,
        "um" | "micrometer" | "micrometers" => 0.000001,
        "nm" | "nanometer" | "nanometers" => 0.000000001,
        "in" | "inch" | "inches" => 0.0254,
        "ft" | "foot" | "feet" => 0.3048,
        other => {
            return Err(
                StructuredError::new("E-UNIT", format!("unsupported length unit: {other}"))
                    .with_hint("Use m, cm, mm, um, nm, in, or ft"),
            );
        }
    };
    Ok(scale)
}

fn scale_bbox(bbox: [[f64; 3]; 2], scale: f64) -> GeomResult<[[f64; 3]; 2]> {
    let mut out = bbox;
    for axis in 0..3 {
        out[0][axis] *= scale;
        out[1][axis] *= scale;
    }
    validate_bbox(out)?;
    Ok(out)
}

fn validate_bbox(bbox: [[f64; 3]; 2]) -> GeomResult<()> {
    for axis in 0..3 {
        if !bbox[0][axis].is_finite() || !bbox[1][axis].is_finite() {
            return Err(StructuredError::new("E-DOMAIN-BBOX", "domain_bbox contains non-finite values"));
        }
        if bbox[0][axis] >= bbox[1][axis] {
            return Err(
                StructuredError::new("E-DOMAIN-BBOX", "domain_bbox min must be < max")
                    .with_hint("Ensure each axis min is less than max"),
            );
        }
    }
    Ok(())
}

fn octree_node_count(octree: &Octree) -> Option<usize> {
    // Best-effort: fidget-mesh exposes nodes publicly in current versions.
    Some(octree.nodes.len())
}

#[derive(Clone, Copy)]
struct StlTri {
    normal: [f32; 3],
    verts: [[f32; 3]; 3],
    attr: u16,
}

fn mesh_to_stl_bytes(mesh: &fidget::mesh::Mesh, deterministic: bool, diagnostics: &mut Vec<Diagnostic>) -> GeomResult<(Vec<u8>, Vec<StlTri>)> {
    let mut buf = Vec::new();
    mesh.write_stl(&mut buf)
        .map_err(|err| StructuredError::new("E-MESH-EXPORT", format!("failed to encode stl: {err}")))?;
    let (header, mut tris) = parse_binary_stl(&buf)?;
    if deterministic {
        tris.sort_by_key(|tri| triangle_key(tri));
    }
    let out = write_binary_stl(&header, &tris);
    if deterministic {
        diagnostics.push(Diagnostic {
            code: "I-DETERMINISTIC-ORDER".to_string(),
            message: "deterministic_order applied to STL output".to_string(),
            severity: "INFO".to_string(),
            op_id: None,
        });
    }
    Ok((out, tris))
}

fn parse_binary_stl(bytes: &[u8]) -> GeomResult<([u8; 80], Vec<StlTri>)> {
    if bytes.len() < 84 {
        return Err(StructuredError::new("E-MESH-EXPORT", "stl output too short"));
    }
    let mut header = [0u8; 80];
    header.copy_from_slice(&bytes[0..80]);
    let tri_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    let expected = 84 + tri_count * 50;
    if expected != bytes.len() {
        return Err(StructuredError::new("E-MESH-EXPORT", "stl output has unexpected length"));
    }
    let mut tris = Vec::with_capacity(tri_count);
    let mut offset = 84;
    for _ in 0..tri_count {
        let mut read_f32 = |start: usize| -> f32 {
            f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
        };
        let normal = [read_f32(offset), read_f32(offset + 4), read_f32(offset + 8)];
        let v1 = [read_f32(offset + 12), read_f32(offset + 16), read_f32(offset + 20)];
        let v2 = [read_f32(offset + 24), read_f32(offset + 28), read_f32(offset + 32)];
        let v3 = [read_f32(offset + 36), read_f32(offset + 40), read_f32(offset + 44)];
        let attr = u16::from_le_bytes(bytes[offset + 48..offset + 50].try_into().unwrap());
        tris.push(StlTri {
            normal,
            verts: [v1, v2, v3],
            attr,
        });
        offset += 50;
    }
    Ok((header, tris))
}

fn write_binary_stl(header: &[u8; 80], tris: &[StlTri]) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + tris.len() * 50);
    out.extend_from_slice(header);
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for tri in tris {
        for val in tri.normal {
            out.extend_from_slice(&val.to_le_bytes());
        }
        for vert in tri.verts {
            for val in vert {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
        out.extend_from_slice(&tri.attr.to_le_bytes());
    }
    out
}

fn triangle_key(tri: &StlTri) -> [i32; 9] {
    let mut verts = [
        quantize_vec(tri.verts[0]),
        quantize_vec(tri.verts[1]),
        quantize_vec(tri.verts[2]),
    ];
    verts.sort();
    [
        verts[0][0], verts[0][1], verts[0][2],
        verts[1][0], verts[1][1], verts[1][2],
        verts[2][0], verts[2][1], verts[2][2],
    ]
}

fn quantize_vec(v: [f32; 3]) -> [i32; 3] {
    [
        (v[0] * 1_000_000.0).round() as i32,
        (v[1] * 1_000_000.0).round() as i32,
        (v[2] * 1_000_000.0).round() as i32,
    ]
}

fn write_obj(path: &Path, tris: &[StlTri]) -> GeomResult<()> {
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<[usize; 3]> = Vec::with_capacity(tris.len());
    let mut map: HashMap<[i32; 3], usize> = HashMap::new();
    for tri in tris {
        let mut face = [0usize; 3];
        for (i, vert) in tri.verts.iter().enumerate() {
            let key = quantize_vec(*vert);
            let idx = if let Some(idx) = map.get(&key) {
                *idx
            } else {
                let idx = verts.len();
                verts.push(*vert);
                map.insert(key, idx);
                idx
            };
            face[i] = idx + 1; // OBJ is 1-based
        }
        indices.push(face);
    }
    let mut file = std::fs::File::create(path)
        .map_err(|err| StructuredError::new("E-IO", format!("failed to create obj: {err}")))?;
    for v in &verts {
        writeln!(file, "v {} {} {}", v[0], v[1], v[2])
            .map_err(|err| StructuredError::new("E-IO", format!("failed to write obj: {err}")))?;
    }
    for f in &indices {
        writeln!(file, "f {} {} {}", f[0], f[1], f[2])
            .map_err(|err| StructuredError::new("E-IO", format!("failed to write obj: {err}")))?;
    }
    Ok(())
}

fn write_ply(path: &Path, tris: &[StlTri]) -> GeomResult<()> {
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<[usize; 3]> = Vec::with_capacity(tris.len());
    let mut map: HashMap<[i32; 3], usize> = HashMap::new();
    for tri in tris {
        let mut face = [0usize; 3];
        for (i, vert) in tri.verts.iter().enumerate() {
            let key = quantize_vec(*vert);
            let idx = if let Some(idx) = map.get(&key) {
                *idx
            } else {
                let idx = verts.len();
                verts.push(*vert);
                map.insert(key, idx);
                idx
            };
            face[i] = idx;
        }
        indices.push(face);
    }
    let mut file = std::fs::File::create(path)
        .map_err(|err| StructuredError::new("E-IO", format!("failed to create ply: {err}")))?;
    writeln!(file, "ply").map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    writeln!(file, "format ascii 1.0").map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    writeln!(file, "element vertex {}", verts.len())
        .map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    writeln!(file, "property float x").map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    writeln!(file, "property float y").map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    writeln!(file, "property float z").map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    writeln!(file, "element face {}", indices.len())
        .map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    writeln!(file, "property list uchar int vertex_indices")
        .map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    writeln!(file, "end_header").map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    for v in &verts {
        writeln!(file, "{} {} {}", v[0], v[1], v[2])
            .map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    }
    for f in &indices {
        writeln!(file, "3 {} {} {}", f[0], f[1], f[2])
            .map_err(|err| StructuredError::new("E-IO", format!("failed to write ply: {err}")))?;
    }
    Ok(())
}
