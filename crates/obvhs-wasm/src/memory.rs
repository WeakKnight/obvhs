use wasm_bindgen::prelude::*;

/// Return a reference to the WebAssembly linear memory.
/// This allows JS to create zero-copy views into WASM heap data.
#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}
