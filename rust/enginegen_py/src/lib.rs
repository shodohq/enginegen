use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use enginegen_core as core;
use enginegen_geom_fidget as geom_fidget;

#[pyfunction]
fn normalize_spec(py: Python<'_>, payload: &PyBytes) -> PyResult<PyObject> {
    let normalized = core::normalize_spec(payload.as_bytes())
        .map_err(|err| PyValueError::new_err(err))?;
    Ok(PyBytes::new(py, &normalized).into())
}

#[pyfunction]
fn normalize_ir(py: Python<'_>, payload: &PyBytes) -> PyResult<PyObject> {
    let normalized = core::normalize_ir(payload.as_bytes())
        .map_err(|err| PyValueError::new_err(err))?;
    Ok(PyBytes::new(py, &normalized).into())
}

#[pyfunction]
fn hash_canonical(payload: &PyBytes) -> PyResult<String> {
    core::hash_canonical(payload.as_bytes()).map_err(|err| PyValueError::new_err(err))
}

#[pyfunction]
fn validate_graph(py: Python<'_>, payload: &PyBytes) -> PyResult<PyObject> {
    let diags = core::validate_graph(payload.as_bytes())
        .map_err(|err| PyValueError::new_err(err))?;
    Ok(PyBytes::new(py, &diags).into())
}

#[pyfunction]
fn schema_graph(py: Python<'_>) -> PyResult<PyObject> {
    let schema = core::schema_graph().map_err(|err| PyValueError::new_err(err))?;
    Ok(PyBytes::new(py, &schema).into())
}

#[pyfunction]
fn schema_ir(py: Python<'_>) -> PyResult<PyObject> {
    let schema = core::schema_ir().map_err(|err| PyValueError::new_err(err))?;
    Ok(PyBytes::new(py, &schema).into())
}

#[pyfunction]
fn schema_manifest(py: Python<'_>) -> PyResult<PyObject> {
    let schema = core::schema_manifest().map_err(|err| PyValueError::new_err(err))?;
    Ok(PyBytes::new(py, &schema).into())
}

#[pyfunction]
fn schema_spec(py: Python<'_>) -> PyResult<PyObject> {
    let schema = core::schema_spec().map_err(|err| PyValueError::new_err(err))?;
    Ok(PyBytes::new(py, &schema).into())
}

#[pyfunction]
fn fidget_compile_and_mesh(py: Python<'_>, ir_json: &str, config_json: &str, out_dir: &str) -> PyResult<PyObject> {
    let result = geom_fidget::compile_and_mesh(ir_json, config_json, std::path::Path::new(out_dir))
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let payload = serde_json::to_vec(&result)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyBytes::new(py, &payload).into())
}

#[pymodule]
fn enginegen_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_spec, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ir, m)?)?;
    m.add_function(wrap_pyfunction!(hash_canonical, m)?)?;
    m.add_function(wrap_pyfunction!(validate_graph, m)?)?;
    m.add_function(wrap_pyfunction!(schema_graph, m)?)?;
    m.add_function(wrap_pyfunction!(schema_ir, m)?)?;
    m.add_function(wrap_pyfunction!(schema_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(schema_spec, m)?)?;
    m.add_function(wrap_pyfunction!(fidget_compile_and_mesh, m)?)?;
    Ok(())
}
