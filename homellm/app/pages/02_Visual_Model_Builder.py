"""Streamlit page for interactive visual model building (Node Editor) with Direct Save."""
from __future__ import annotations

import json
import threading
import socket
import yaml
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from homellm.models.blueprint import Blueprint

# Internationalization (i18n)
try:
    from homellm.i18n import t
except ImportError:
    # Fallback for direct run
    def t(key, **kwargs):
        return key
from homellm.models.blocks import BLOCK_REGISTRY
try:
    from homellm.app.ui_preferences import DEFAULT_THEME, init_user_preferences, apply_theme_css
except ImportError:
    from ..ui_preferences import DEFAULT_THEME, init_user_preferences, apply_theme_css

def _find_project_root(start: Path) -> Path:
    """Find repo root by walking up until we see docker-compose.yml or pyproject.toml."""
    cur = start.resolve()
    for _ in range(10):
        if (cur / "docker-compose.yml").exists() or (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: go to /app if this is a typical container layout /app/homellm/app/pages/...
    try:
        return start.resolve().parents[3]
    except Exception:
        return start.resolve().parent


PROJECT_ROOT = _find_project_root(Path(__file__).parent)
BLUEPRINTS_DIR = PROJECT_ROOT / "blueprints"
BLUEPRINTS_DIR.mkdir(exist_ok=True)
RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(exist_ok=True)
USER_PREFS_FILE = RUNS_DIR / "ui_preferences.json"

# Port for the sidecar save server
SIDECAR_PORT = 8502


# ==============================================================================
# DRAWFLOW TO BLUEPRINT CONVERTER
# ==============================================================================

def drawflow_to_blueprint(drawflow_data: Dict, training_config: Dict = None) -> Dict:
    """Converts Drawflow export format to HomeLLM Blueprint."""
    nodes = drawflow_data["drawflow"]["Home"]["data"]
    
    blocks = []
    id_map = {} 
    
    # First pass: IDs
    for nid, node in nodes.items():
        b_type = node["name"]
        clean_id = f"{b_type}_{nid}"
        id_map[nid] = clean_id
    
    # Second pass: Data
    inferred_hidden_size = 512 # Default
    inferred_vocab_size = 50257
    
    for nid, node in nodes.items():
        b_type = node["name"]
        params = node["data"].copy()
        params.pop("type", None)
        
        # Try to infer global dims from embedding layer
        if b_type == "token_embedding":
            if "hidden_size" in params:
                inferred_hidden_size = int(params["hidden_size"])
            if "vocab_size" in params:
                inferred_vocab_size = int(params["vocab_size"])
        
        inputs = []
        input_keys = sorted(node["inputs"].keys())
        for k in input_keys:
            conns = node["inputs"][k]["connections"]
            if conns:
                source_nid = str(conns[0]["node"])
                inputs.append(id_map[source_nid])
        
        block = {
            "id": id_map[nid],
            "type": b_type,
            "params": params,
            "inputs": inputs
        }
        blocks.append(block)
    
    result = {
        "model_type": "homellm_blueprint",
        "vocab_size": inferred_vocab_size, 
        "hidden_size": inferred_hidden_size,
        "max_position_embeddings": 2048,
        "auto_project": True,
        "blocks": blocks
    }
    
    # Add training config if provided
    if training_config:
        result["training"] = training_config
    
    return result


# ==============================================================================
# BACKGROUND SAVE SERVER
# ==============================================================================

class SaveRequestHandler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path == '/ping':
            self.send_response(200)
            self._send_cors()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/save_blueprint':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                payload = json.loads(post_data.decode('utf-8'))
                fname = payload.get("filename", "model.json")
                # Use JSON by default
                if not fname.endswith(".json") and not fname.endswith(".yaml"):
                    fname += ".json"
                
                raw_data = payload.get("data")
                training_config = payload.get("training", None)
                
                # Convert
                bp_data = drawflow_to_blueprint(raw_data, training_config)
                
                # Validate
                Blueprint.parse_obj(bp_data)
                
                # Save as JSON (or YAML if specified)
                save_path = BLUEPRINTS_DIR / fname
                with open(save_path, "w", encoding="utf-8") as f:
                    if fname.endswith(".yaml"):
                        yaml_str = yaml.dump(bp_data, sort_keys=False, allow_unicode=True)
                        f.write(yaml_str)
                    else:
                        json.dump(bp_data, f, indent=2, ensure_ascii=False)
                
                # Response
                self.send_response(200)
                self._send_cors()
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "path": str(save_path)}).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self._send_cors()
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        
        elif self.path == '/analyze_blueprint':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                payload = json.loads(post_data.decode('utf-8'))
                raw_data = payload.get("data")
                training_config = payload.get("training", None)
                
                # Convert
                bp_data = drawflow_to_blueprint(raw_data, training_config)
                
                # Validation & Build
                import torch
                from homellm.models.blueprint_model import BlueprintLMConfig, BlueprintForCausalLM
                # Blueprint is already imported globally
                
                bp = Blueprint.parse_obj(bp_data)
                
                config = BlueprintLMConfig(
                    vocab_size=bp.vocab_size,
                    hidden_size=bp.hidden_size,
                    max_position_embeddings=bp.max_position_embeddings,
                    auto_project=bp.auto_project,
                    blueprint=bp.dict()
                )
                
                # Build on CPU (safe & reasonably fast for analysis)
                model = BlueprintForCausalLM(config)
                
                # --- Shape Analysis (Dry Run) ---
                node_shapes = {}
                error_msg = None
                
                try:
                    def get_shape_hook(name):
                        def hook(module, inp, out):
                            if isinstance(out, torch.Tensor):
                                shape_str = str(list(out.shape))
                            elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                                shape_str = str(list(out[0].shape))
                            else:
                                shape_str = "N/A"
                            node_shapes[name] = shape_str
                        return hook

                    # Register hooks on graph blocks
                    # model.model is the BlueprintGraphModule
                    for name, module in model.model.named_children():
                        module.register_forward_hook(get_shape_hook(name))
                    
                    # Dummy forward pass
                    # Use small seq_len to be fast
                    seq_len = min(64, bp.max_position_embeddings)
                    dummy_input = torch.randint(0, bp.vocab_size, (1, seq_len))
                    
                    with torch.no_grad():
                        model(dummy_input)
                        
                except Exception as e:
                    error_msg = str(e)
                    node_shapes["error"] = error_msg
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                size_mb = total_params * 4 / (1024 * 1024) # fp32
                
                result = {
                    "status": "success",
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "size_mb": size_mb,
                    "layers": len(bp.blocks),
                    "hidden_size": bp.hidden_size,
                    "vocab_size": bp.vocab_size,
                    "node_shapes": node_shapes
                }
                
                self.send_response(200)
                self._send_cors()
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self._send_cors()
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        return # Silence logs

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) == 0

def start_server():
    if getattr(start_server, "server_running", False):
        return

    if is_port_in_use(SIDECAR_PORT):
        print(f"Port {SIDECAR_PORT} is already in use. Assuming Save Server is active.")
        start_server.server_running = True
        return

    try:
        server = HTTPServer(('0.0.0.0', SIDECAR_PORT), SaveRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        start_server.server_running = True
        print(f"Blueprint Save Server started on port {SIDECAR_PORT}")
    except Exception as e:
        print(f"Failed to start Save Server: {e}")

start_server()


# ==============================================================================
# HTML/JS Code for Drawflow Editor
# ==============================================================================

DRAWFLOW_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jerosoler/Drawflow/dist/drawflow.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.13.0/css/all.min.css">
  <script src="https://cdn.jsdelivr.net/gh/jerosoler/Drawflow/dist/drawflow.min.js"></script>
  <style>
    /* ... (CSS same as before) ... */
    :root {{
      --bg-color: #2b2b2b;
      --node-bg: #1e1e1e;
      --text-color: #ffffff;
      --border-color: #444;
    }}
    body {{ margin: 0px; padding: 0px; width: 100vw; height: 100vh; overflow: hidden; background: var(--bg-color); font-family: 'Roboto', sans-serif; }}
    
    /* Configured for Left Sidebar Layout */
    #drawflow {{ 
        position: absolute;
        top: 0; left: 240px; 
        width: calc(100% - 240px); 
        height: 100%; 
        background: var(--bg-color); 
        background-size: 25px 25px; 
        background-image: radial-gradient(#444 1px, transparent 1px); 
    }}
    
    .drawflow .drawflow-node {{
      background: var(--node-bg);
      border: 1px solid var(--border-color);
      color: var(--text-color);
      width: 200px;
      padding: 0px;
      border-radius: 8px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .drawflow .drawflow-node.selected {{
      border: 1px solid #4ea9ff;
      box-shadow: 0 0 10px rgba(78, 169, 255, 0.5);
    }}
    
    .node-header {{
      padding: 10px;
      border-bottom: 1px solid var(--border-color);
      border-radius: 8px 8px 0 0;
      font-weight: bold;
      display: flex;
      align-items: center;
    }}
    .node-header i {{ margin-right: 8px; }}
    
    .node-content {{ padding: 10px; font-size: 0.9em; }}
    .node-content input, .node-content select {{
      width: 90%;
      background: #333;
      border: 1px solid #555;
      color: white;
      margin-bottom: 5px;
      padding: 4px;
      border-radius: 4px;
    }}
    
    .drawflow .drawflow-node .input, .drawflow .drawflow-node .output {{
      width: 15px; height: 15px;
      background: #777;
      border: 1px solid #fff;
    }}
    .drawflow .drawflow-node .input:hover, .drawflow .drawflow-node .output:hover {{
      background: #4ea9ff;
    }}
    
    .header-core {{ background: #1e3a8a; }} 
    .header-layer {{ background: #4c1d95; }} 
    .header-op {{ background: #065f46; }} 
    .header-act {{ background: #be185d; }}
    .header-conv {{ background: #b45309; }}
    .header-pool {{ background: #0f766e; }}
    
    /* MAIN LEFT TOOLBAR */
    #toolbar {{
      position: absolute;
      top: 10px; left: 10px;
      background: #1e1e1e;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid #444;
      display: flex;
      flex-direction: column;
      gap: 5px;
      z-index: 10;
      max-height: 95vh;
      overflow-y: auto;
      width: 220px; /* Wider to fit config */
    }}
    
    .category-title {{
        font-weight:bold; margin-bottom:2px; margin-top:12px; color:#c084fc; font-size:0.75em; text-transform:uppercase; border-bottom: 1px solid #444; padding-bottom:2px;
    }}
    
    .tool-btn {{
      background: #333;
      color: white;
      border: 1px solid #555;
      padding: 6px 10px;
      cursor: pointer;
      text-align: left;
      border-radius: 4px;
      font-size: 0.85em;
      display: flex;
      align-items: center;
    }}
    .tool-btn:hover {{ background: #444; }}
    .tool-btn i {{ width: 18px; text-align: center; margin-right: 6px; }}
    
    /* Config Panel inside Toolbar */
    .config-section {{
        margin-top: 5px;
        background: #2a2a2a;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #555;
    }}
    .config-label {{
        font-size: 0.75em; color: #aaa; margin-bottom: 2px; display:block;
    }}
    .config-input, .config-select {{
        background: #111;
        border: 1px solid #444;
        color: white;
        padding: 4px;
        border-radius: 3px;
        width: 100%;
        box-sizing: border-box;
        font-size: 0.8em;
        margin-bottom: 6px;
    }}
    .action-btn {{
        width: 100%; padding: 6px; margin-top: 5px; border-radius: 4px; border:none; cursor:pointer; font-weight:bold; font-size: 0.85em;
    }}
    .btn-primary {{ background: #4ea9ff; color: white; }}
    .btn-secondary {{ background: #555; color: white; }}
    .btn-danger {{ background: #ef4444; color: white; }}
    .btn-success {{ background: #22c55e; color: white; }}

    #preset-bar {{
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid #444;
    }}
    
    /* Modal for Analysis */
    #modal-overlay {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.5); z-index: 999; display: none;
        align-items: center; justify-content: center;
    }}
    #modal-content {{
        background: #1e1e1e; border: 1px solid #444; border-radius: 8px;
        padding: 20px; width: 400px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    #modal-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; color: #4ea9ff; }}
    .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }}
    .stat-label {{ color: #aaa; }}
    .stat-val {{ font-weight: bold; font-family: monospace; }}

  </style>
</head>
<body>

<div id="modal-overlay" onclick="closeModal()">
    <div id="modal-content" onclick="event.stopPropagation()">
        <div id="modal-title">📊 Model Analysis</div>
        <div id="modal-body">Loading...</div>
        <button class="action-btn btn-secondary" onclick="closeModal()">Close</button>
    </div>
</div>

<div id="toolbar">
  
  <!-- PRESETS SECTION -->
  <div id="preset-bar">
    <div class="config-label">PRESETS</div>
    <select id="preset-select" class="config-select" onchange="loadPreset(this.value)">
        <option value="" disabled selected>Load Preset...</option>
        <option value="gpt2_mini">GPT-2 (Mini)</option>
        <option value="llama_mini">Llama (Mini)</option>
        <option value="moe_simple">Simple MoE</option>
    </select>
    <button class="action-btn btn-danger" onclick="editor.clear()">Clear All</button>
  </div>

  <!-- PROJECT SETTINGS SECTION -->
  <div class="category-title" style="margin-top:0">Project Settings</div>
  <div class="config-section">
      <label class="config-label">Filename</label>
      <input type="text" id="filename" class="config-input" value="model.json" />
      
      <button class="action-btn btn-primary" onclick="saveGraph()">💾 Save to Disk</button>
      <button class="action-btn btn-success" onclick="analyzeGraph()">📊 Analyze Model</button>
      <label style="font-size:0.8em; display:block; margin-top:8px; cursor:pointer; color:#ccc;">
          <input type="checkbox" id="viz-shapes" checked> Visualize Shapes on Graph
      </label>
      <button class="action-btn btn-secondary" onclick="copyJSON()">📋 Copy JSON</button>
      <div id="status-msg" style="font-size:0.75em; margin-top:5px; color:#ef4444; display:none;"></div>
      
      <label class="config-label" style="margin-top:5px">Server Host</label>
      <input type="text" id="host" class="config-input" value="localhost" />
  </div>

  <!-- TRAINING CONFIG SECTION -->
  <div class="category-title">Training Config</div>
  <div class="config-section">
      <label class="config-label">Optimizer</label>
      <select id="optimizer" class="config-select">
        <option value="adamw" selected>AdamW</option>
        <option value="adam">Adam</option>
        <option value="sgd">SGD</option>
        <option value="rmsprop">RMSprop</option>
      </select>
      
      <label class="config-label">Learning Rate</label>
      <input type="number" id="lr" class="config-input" value="0.001" step="0.0001" />
      
      <label class="config-label">Weight Decay</label>
      <input type="number" id="weight_decay" class="config-input" value="0.01" step="0.001" />
      
      <label class="config-label">Loss Function</label>
      <select id="loss_fn" class="config-select">
        <option value="cross_entropy" selected>CrossEntropyLoss</option>
        <option value="mse">MSELoss</option>
        <option value="l1">L1Loss</option>
      </select>
      
      <label class="config-label">Label Smoothing</label>
      <input type="number" id="label_smoothing" class="config-input" value="0.0" step="0.01" min="0" max="1" />
  </div>

  <!-- BLOCKS -->
  <div class="category-title">Core Blocks</div>
  <div class="tool-btn" onclick="addNode('token_embedding')"><i class="fas fa-font"></i> Embedding</div>
  <div class="tool-btn" onclick="addNode('positional_embedding')"><i class="fas fa-map-marker-alt"></i> Pos Embed</div>
  
  <div class="category-title">Layers</div>
  <div class="tool-btn" onclick="addNode('attention')"><i class="fas fa-eye"></i> Attention</div>
  <div class="tool-btn" onclick="addNode('causal_self_attention')"><i class="fas fa-bolt"></i> Flash Attn (RoPE)</div>
  <div class="tool-btn" onclick="addNode('mlp')"><i class="fas fa-brain"></i> MLP</div>
  <div class="tool-btn" onclick="addNode('swiglu')"><i class="fas fa-atom"></i> SwiGLU (Llama)</div>
  <div class="tool-btn" onclick="addNode('llama_block')"><i class="fas fa-cubes"></i> Llama Block</div>
  <div class="tool-btn" onclick="addNode('residual_mlp')"><i class="fas fa-recycle"></i> Res MLP</div>
  <div class="tool-btn" onclick="addNode('repeater')"><i class="fas fa-clone"></i> Repeater (Loop)</div>
  <div class="tool-btn" onclick="addNode('moe')"><i class="fas fa-network-wired"></i> MoE</div>

  <div class="category-title">Convolutions</div>
  <div class="tool-btn" onclick="addNode('causal_conv1d')"><i class="fas fa-wave-square"></i> Causal Conv1D</div>
  
  <div class="category-title">Ops & Tensor</div>
  <div class="tool-btn" onclick="addNode('add')"><i class="fas fa-plus"></i> Add</div>
  <div class="tool-btn" onclick="addNode('multiply')"><i class="fas fa-times"></i> Multiply</div>
  <div class="tool-btn" onclick="addNode('concat')"><i class="fas fa-object-group"></i> Concat</div>
  <div class="tool-btn" onclick="addNode('linear')"><i class="fas fa-ruler-horizontal"></i> Linear</div>
  
  <div class="category-title">Activations</div>
  <div class="tool-btn" onclick="addNode('activation')"><i class="fas fa-bolt"></i> Activation</div>
  
  <div class="category-title">Norm & Reg</div>
  <div class="tool-btn" onclick="addNode('rmsnorm')"><i class="fas fa-balance-scale"></i> RMSNorm</div>
  <div class="tool-btn" onclick="addNode('groupnorm')"><i class="fas fa-layer-group"></i> GroupNorm</div>
  <div class="tool-btn" onclick="addNode('dropout')"><i class="fas fa-dice-d6"></i> Dropout</div>

  <div class="category-title">Pooling</div>
  <div class="tool-btn" onclick="addNode('adaptive_avg_pool')"><i class="fas fa-compress"></i> Adapt Pool</div>
  
  <div class="category-title">Advanced</div>
  <div class="tool-btn" onclick="addNode('inline_code')"><i class="fas fa-code"></i> Inline Code</div>
  <div class="tool-btn" onclick="addNode('custom_op')"><i class="fas fa-plug"></i> Custom Op</div>
</div>

<div id="drawflow"></div>

<script>
  var id = document.getElementById("drawflow");
  const editor = new Drawflow(id);
  editor.reroute = true;
  editor.start();

  // Try to auto-detect host on load (referrer is the Streamlit parent URL)
  try {{
      if (document.referrer && document.referrer !== '') {{
          const u = new URL(document.referrer);
          if (u.hostname) {{
              document.getElementById('host').value = u.hostname;
          }}
      }}
  }} catch (e) {{ console.log("Auto-detect host from referrer failed", e); }}
  // Fallback (some iframes expose real location)
  try {{
      if (window.location.hostname && window.location.hostname !== '') {{
          document.getElementById('host').value = window.location.hostname;
      }}
  }} catch (e) {{ console.log("Auto-detect host from window.location failed", e); }}

  window.updateNodeData = function(nodeId, key, value) {{
    var data = editor.drawflow.drawflow.Home.data[nodeId].data;
    data[key] = value;
  }}

  function closeModal() {{
      document.getElementById('modal-overlay').style.display = 'none';
  }}

  // --- NODE TEMPLATES (same as before) ---
  function getHtml(type) {{
    let inputs = '<div class="input input_1"></div>';
    let outputs = '<div class="output output_1"></div>';
    let headerClass = 'header-op';
    let icon = 'fa-cube';
    let content = '';

    if (type === 'token_embedding') {{
      headerClass = 'header-core'; icon = 'fa-font'; inputs = '';
      content += `Vocab: <input type="number" value="50257" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'vocab_size', parseInt(this.value))"> Dim: <input type="number" value="512" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'hidden_size', parseInt(this.value))">`;
    }} else if (type === 'positional_embedding') {{
      headerClass = 'header-core'; icon = 'fa-map-marker-alt'; inputs = '';
      content += `Ctx: <input type="number" value="2048" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'max_position_embeddings', parseInt(this.value))"> Dim: <input type="number" value="512" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'hidden_size', parseInt(this.value))">`;
    }} else if (type === 'attention') {{
      headerClass = 'header-layer'; icon = 'fa-eye';
      content += `Heads: <input type="number" value="8" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'num_heads', parseInt(this.value))"> Drop: <input type="number" step="0.1" value="0.0" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'dropout', parseFloat(this.value))">`;
    }} else if (type === 'causal_self_attention') {{
      headerClass = 'header-layer'; icon = 'fa-bolt';
      content += `Heads: <input type="number" value="8" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'num_heads', parseInt(this.value))"><br><label><input type="checkbox" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'use_rope', this.checked)"> Use RoPE</label><br>Theta: <input type="number" value="10000" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'rope_theta', parseFloat(this.value))">`;
    }} else if (type === 'swiglu') {{
      headerClass = 'header-layer'; icon = 'fa-atom';
      content += `Inter: <input type="number" value="2048" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'intermediate_size', parseInt(this.value))"> Drop: <input type="number" step="0.1" value="0.0" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'dropout', parseFloat(this.value))">`;
    }} else if (type === 'llama_block') {{
      headerClass = 'header-layer'; icon = 'fa-cubes';
      content += `Heads: <input type="number" value="8" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'num_heads', parseInt(this.value))"><br>Inter: <input type="number" value="2048" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'intermediate_size', parseInt(this.value))">`;
    }} else if (type === 'repeater') {{
      headerClass = 'header-layer'; icon = 'fa-clone';
      content += `Repeats: <input type="number" value="1" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'num_repeats', parseInt(this.value))"><br>Block: <select onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'block_type', this.value)"><option value="residual_mlp">Res MLP</option><option value="attention">Attention</option><option value="causal_self_attention">Flash Attn</option><option value="swiglu">SwiGLU</option><option value="llama_block">Llama Block</option></select>`;
    }} else if (type === 'moe') {{
      headerClass = 'header-layer'; icon = 'fa-network-wired';
      content += `Experts: <input type="number" value="8" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'num_experts', parseInt(this.value))"> Select: <input type="number" value="2" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'num_select', parseInt(this.value))"><br>Type: <select onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'expert_type', this.value)"><option value="mlp">MLP</option><option value="swiglu">SwiGLU</option></select><br>Drop: <input type="number" step="0.1" value="0.0" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'dropout', parseFloat(this.value))">`;
    }} else if (type.includes('mlp')) {{
      headerClass = 'header-layer'; icon = 'fa-brain';
      content += `Inter: <input type="number" value="2048" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'intermediate_size', parseInt(this.value))"> Act: <select onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'activation', this.value)"><option value="silu">SiLU</option><option value="gelu">GELU</option><option value="relu">ReLU</option></select>`;
    }} else if (type === 'causal_conv1d') {{
        headerClass = 'header-conv'; icon = 'fa-wave-square';
        content += `Out Ch: <input type="number" value="512" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'out_channels', parseInt(this.value))"> K Size: <input type="number" value="3" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'kernel_size', parseInt(this.value))"> Dilat: <input type="number" value="1" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'dilation', parseInt(this.value))">`;
    }} else if (type === 'add') {{
      inputs = '<div class="input input_1"></div><div class="input input_2"></div>'; icon = 'fa-plus';
    }} else if (type === 'multiply') {{
      inputs = '<div class="input input_1"></div><div class="input input_2"></div>'; icon = 'fa-times';
    }} else if (type === 'concat') {{
      inputs = '<div class="input input_1"></div><div class="input input_2"></div><div class="input input_3"></div>'; icon = 'fa-object-group';
      content += `Dim: <input type="number" value="-1" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'dim', parseInt(this.value))">`;
    }} else if (type === 'linear') {{
       icon = 'fa-ruler-horizontal'; content += `Out: <input type="number" value="50257" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'out_features', parseInt(this.value))">`;
    }} else if (type === 'activation') {{
        headerClass = 'header-act'; icon = 'fa-bolt';
        content += `<select onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'type', this.value)"><option value="silu">SiLU</option><option value="gelu">GELU</option><option value="relu">ReLU</option><option value="tanh">Tanh</option><option value="sigmoid">Sigmoid</option></select>`;
    }} else if (type === 'rmsnorm') {{
       icon = 'fa-balance-scale'; content += `Eps: <input type="number" step="1e-6" value="1e-5" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'eps', parseFloat(this.value))">`;
    }} else if (type === 'groupnorm') {{
       icon = 'fa-layer-group'; content += `Groups: <input type="number" value="32" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'num_groups', parseInt(this.value))">`;
    }} else if (type === 'adaptive_avg_pool') {{
        headerClass = 'header-pool'; icon = 'fa-compress'; content += `Size: <input type="number" value="1" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'output_size', parseInt(this.value))">`;
    }} else if (type === 'inline_code') {{
       icon = 'fa-code'; content += `Code:<br><input type="text" value="x = x" onchange="updateNodeData(this.parentElement.parentElement.id.slice(5), 'code', this.value)">`;
    }}

    return `
      <div class="node-header ${{headerClass}}"><i class="fas ${{icon}}"></i> ${{type}}</div>
      <div class="node-content">${{content}}</div>
    `;
  }}

  function addNode(type, x, y) {{
    var num_inputs = (type.includes('embedding') ? 0 : 1);
    if (['add', 'multiply'].includes(type)) num_inputs = 2;
    if (type === 'concat') num_inputs = 3;

    var data = {{ type: type }};
    if (type === 'token_embedding') {{ data.vocab_size = 50257; data.hidden_size = 512; }}
    if (type === 'positional_embedding') {{ data.max_position_embeddings = 2048; data.hidden_size = 512; }}
    if (type === 'attention') {{ data.num_heads = 8; data.dropout = 0.0; }}
    if (type === 'causal_self_attention') {{ data.num_heads = 8; data.dropout = 0.0; data.use_rope = false; data.rope_theta = 10000.0; }}
    if (type === 'swiglu') {{ data.intermediate_size = 2048; data.dropout = 0.0; }}
    if (type === 'llama_block') {{ data.num_heads = 8; data.intermediate_size = 2048; data.dropout = 0.0; }}
    if (type === 'moe') {{ data.num_experts = 8; data.num_select = 2; data.dropout = 0.0; data.expert_type = 'mlp'; }}
    if (type === 'repeater') {{ data.num_repeats = 1; data.block_type = 'residual_mlp'; }}
    if (type.includes('mlp') && type !== 'swiglu' && type !== 'residual_mlp') {{ data.intermediate_size = 2048; data.activation = 'silu'; }}
    if (type === 'residual_mlp') {{ data.intermediate_size = 2048; data.activation = 'silu'; }}
    if (type === 'causal_conv1d') {{ data.out_channels = 512; data.kernel_size = 3; data.dilation = 1; }}
    if (type === 'rmsnorm') {{ data.eps = 1e-5; }}
    if (type === 'groupnorm') {{ data.num_groups = 32; }}
    if (type === 'linear') {{ data.out_features = 50257; }}
    if (type === 'concat') {{ data.dim = -1; }}
    if (type === 'activation') {{ data.type = 'silu'; }}
    if (type === 'adaptive_avg_pool') {{ data.output_size = 1; }}
    if (type === 'inline_code') {{ data.code = 'x = x'; }}
    
    // Default coords if not provided
    if (x === undefined) x = 100;
    if (y === undefined) y = 100;

    return editor.addNode(type, num_inputs, 1, x, y, type, data, getHtml(type));
  }}
  
  // Default Graph
  (function initGraph(){{
      var n1 = editor.addNode('token_embedding', 0, 1, 50, 100, 'token_embedding', {{type: 'token_embedding', vocab_size: 50257, hidden_size: 512}}, getHtml('token_embedding'));
      var n2 = editor.addNode('positional_embedding', 0, 1, 50, 250, 'positional_embedding', {{type: 'positional_embedding', max_position_embeddings: 2048, hidden_size: 512}}, getHtml('positional_embedding'));
      var n3 = editor.addNode('add', 2, 1, 350, 150, 'add', {{type: 'add'}}, getHtml('add'));
      var n4 = editor.addNode('attention', 1, 1, 600, 150, 'attention', {{type: 'attention', num_heads: 8, dropout: 0.0}}, getHtml('attention'));
      var n5 = editor.addNode('rmsnorm', 1, 1, 850, 150, 'rmsnorm', {{type: 'rmsnorm', eps: 1e-5}}, getHtml('rmsnorm'));
      
      editor.addConnection(n1, n3, "output_1", "input_1");
      editor.addConnection(n2, n3, "output_1", "input_2");
      editor.addConnection(n3, n4, "output_1", "input_1");
      editor.addConnection(n4, n5, "output_1", "input_1");
  }})();

  function getTrainingConfig() {{
    const optimizer = document.getElementById('optimizer').value;
    const lr = parseFloat(document.getElementById('lr').value);
    const weightDecay = parseFloat(document.getElementById('weight_decay').value);
    const lossFn = document.getElementById('loss_fn').value;
    const labelSmoothing = parseFloat(document.getElementById('label_smoothing').value);
    
    const config = {{
      optimizer: optimizer,
      lr: lr,
      weight_decay: weightDecay,
      loss_fn: lossFn
    }};
    
    // Label smoothing (for CrossEntropy)
    if (lossFn === 'cross_entropy' && labelSmoothing > 0) {{
      config.label_smoothing = labelSmoothing;
    }}
    
    return config;
  }}

  async function saveGraph() {{
    const data = editor.export();
    const fname = document.getElementById("filename").value;
    const host = document.getElementById("host").value;
    const training = getTrainingConfig();
    const btn = document.querySelector("button[onclick='saveGraph()']");
    const status = document.getElementById("status-msg");
    const originalText = btn.innerText;
    
    btn.innerText = "Saving...";
    btn.disabled = true;
    status.style.display = 'none';

    try {{
        const port = {SIDECAR_PORT};
        let scheme = "http:";
        try {{
            if (document.referrer && document.referrer !== "") {{
                const u = new URL(document.referrer);
                if (u.protocol === "http:" || u.protocol === "https:") scheme = u.protocol;
            }}
        }} catch (e) {{}}
        if (!(scheme === "http:" || scheme === "https:")) scheme = "http:";
        const url = scheme + "//" + host + ":" + port + "/save_blueprint";
        
        console.log("Saving to: " + url);
        console.log("Training config:", training);

        const response = await fetch(url, {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ filename: fname, data: data, training: training }})
        }});

        if (response.ok) {{
            const res = await response.json();
            alert("✅ Model saved successfully to " + res.path + "\\n\\nOptimizer: " + training.optimizer.toUpperCase() + "\\nLoss: " + training.loss_fn);
        }} else {{
            const err = await response.text();
            alert("❌ Save failed: " + err);
        }}
    }} catch (e) {{
        let msg = "❌ Connection Error: " + e;
        msg += "\\n\\nCheck host/port (" + document.getElementById("host").value + ":" + {SIDECAR_PORT} + ")";
        alert(msg);
        status.innerText = "⚠️ Network error. Use 'Copy JSON'.";
        status.style.display = 'inline';
    }} finally {{
        btn.innerText = originalText;
        btn.disabled = false;
    }}
  }}

  async function analyzeGraph() {{
    const data = editor.export();
    const host = document.getElementById("host").value;
    const training = getTrainingConfig();
    const btn = document.querySelector("button[onclick='analyzeGraph()']");
    const originalText = btn.innerText;
    
    btn.innerText = "Analyzing...";
    btn.disabled = true;
    
    // Show modal loading
    document.getElementById('modal-overlay').style.display = 'flex';
    document.getElementById('modal-body').innerHTML = '<div style="text-align:center; padding:20px;"><i class="fas fa-circle-notch fa-spin"></i> Building model & calculating stats...</div>';

    try {{
        const port = {SIDECAR_PORT};
        let scheme = "http:";
        try {{
            if (document.referrer && document.referrer !== "") {{
                const u = new URL(document.referrer);
                if (u.protocol === "http:" || u.protocol === "https:") scheme = u.protocol;
            }}
        }} catch (e) {{}}
        if (!(scheme === "http:" || scheme === "https:")) scheme = "http:";
        const url = scheme + "//" + host + ":" + port + "/analyze_blueprint";
        
        const response = await fetch(url, {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ data: data, training: training }})
        }});

        if (response.ok) {{
            const res = await response.json();
            const html = `
                <div class="stat-row"><span class="stat-label">Total Params</span><span class="stat-val">${{res.total_params.toLocaleString()}}</span></div>
                <div class="stat-row"><span class="stat-label">Trainable</span><span class="stat-val">${{res.trainable_params.toLocaleString()}}</span></div>
                <div class="stat-row"><span class="stat-label">Est. Size (FP32)</span><span class="stat-val">${{res.size_mb.toFixed(2)}} MB</span></div>
                <hr style="border-color:#444; margin:10px 0;">
                <div class="stat-row"><span class="stat-label">Layers/Blocks</span><span class="stat-val">${{res.layers}}</span></div>
                <div class="stat-row"><span class="stat-label">Hidden Dim</span><span class="stat-val">${{res.hidden_size}}</span></div>
                <div class="stat-row"><span class="stat-label">Vocab Size</span><span class="stat-val">${{res.vocab_size}}</span></div>
            `;
            document.getElementById('modal-body').innerHTML = html;
            
            if (res.node_shapes) {{
                displayShapes(res.node_shapes);
            }}

        }} else {{
            const err = await response.text();
            document.getElementById('modal-body').innerHTML = `<div style="color:#ef4444">❌ Analysis failed:<br>${{err}}</div>`;
        }}
    }} catch (e) {{
        document.getElementById('modal-body').innerHTML = `<div style="color:#ef4444">❌ Connection Error:<br>${{e}}</div>`;
    }} finally {{
        btn.innerText = originalText;
        btn.disabled = false;
    }}
  }}
  
  function displayShapes(shapeMap) {{
      document.querySelectorAll('.shape-badge').forEach(b => b.remove());

      const viz = document.getElementById('viz-shapes');
      if (!viz || !viz.checked || !shapeMap) return;

      Object.keys(shapeMap).forEach(key => {{
          if (key === 'error') return;
          const parts = key.split('_');
          const nid = parts[parts.length - 1];
          
          const nodeEl = document.getElementById('node-' + nid);
          if (nodeEl) {{
               const badge = document.createElement('div');
               badge.className = 'shape-badge';
               badge.innerText = shapeMap[key];
               badge.style.position = 'absolute';
               badge.style.top = '-22px';
               badge.style.right = '0px';
               badge.style.background = '#3b82f6';
               badge.style.color = '#fff';
               badge.style.padding = '2px 6px';
               badge.style.borderRadius = '4px';
               badge.style.fontSize = '11px';
               badge.style.fontFamily = 'monospace';
               badge.style.pointerEvents = 'none';
               badge.style.whiteSpace = 'nowrap';
               badge.style.zIndex = '100';
               badge.style.border = '1px solid #60a5fa';
               badge.style.boxShadow = '0 2px 4px rgba(0,0,0,0.5)';
               
               nodeEl.appendChild(badge);
          }}
      }});
  }}
  
  function loadPreset(presetName) {{
      editor.clear();
      
      if (presetName === 'gpt2_mini') {{
          // Embedding -> Pos -> Add -> [Attention -> MLP]x4 -> Norm
          const n1 = addNode('token_embedding', 50, 100);
          const n2 = addNode('positional_embedding', 50, 250);
          const n3 = addNode('add', 300, 175);
          
          // Block 1
          const n4 = addNode('attention', 550, 100);
          const n5 = addNode('residual_mlp', 800, 100);
          
          // Block 2
          const n6 = addNode('attention', 550, 300);
          const n7 = addNode('residual_mlp', 800, 300);
          
          const n8 = addNode('rmsnorm', 1050, 200);
          
          // Data
          updateNodeData(n1, 'hidden_size', 256);
          updateNodeData(n2, 'hidden_size', 256);
          updateNodeData(n4, 'num_heads', 4);
          updateNodeData(n6, 'num_heads', 4);
          
          // Connections
          editor.addConnection(n1, n3, "output_1", "input_1");
          editor.addConnection(n2, n3, "output_1", "input_2");
          editor.addConnection(n3, n4, "output_1", "input_1");
          editor.addConnection(n4, n5, "output_1", "input_1");
          editor.addConnection(n5, n6, "output_1", "input_1");
          editor.addConnection(n6, n7, "output_1", "input_1");
          editor.addConnection(n7, n8, "output_1", "input_1");
          
      }} else if (presetName === 'llama_mini') {{
          const n1 = addNode('token_embedding', 50, 200);
          
          // Repeater: Llama Block x 8
          const n2 = addNode('repeater', 400, 200);
          updateNodeData(n2, 'num_repeats', 8);
          updateNodeData(n2, 'block_type', 'llama_block');
          
          const n3 = addNode('rmsnorm', 700, 200);
          
          editor.addConnection(n1, n2, "output_1", "input_1");
          editor.addConnection(n2, n3, "output_1", "input_1");
          
      }} else if (presetName === 'moe_simple') {{
          const n1 = addNode('token_embedding', 50, 100);
          const n2 = addNode('positional_embedding', 50, 250);
          const n3 = addNode('add', 300, 175);
          
          const n4 = addNode('moe', 550, 175);
          const n5 = addNode('moe', 800, 175);
          
          const n6 = addNode('rmsnorm', 1050, 175);
          
          editor.addConnection(n1, n3, "output_1", "input_1");
          editor.addConnection(n2, n3, "output_1", "input_2");
          editor.addConnection(n3, n4, "output_1", "input_1");
          editor.addConnection(n4, n5, "output_1", "input_1");
          editor.addConnection(n5, n6, "output_1", "input_1");
      }}
  }}
  
  function copyJSON() {{
      const data = editor.export();
      const fname = document.getElementById("filename").value;
      const training = getTrainingConfig();
      
      const payload = {{
          filename: fname,
          data: data,
          training: training
      }};
      
      const jsonStr = JSON.stringify(payload, null, 2);
      
      // Fallback for clipboard API if not available in iframe
      const el = document.createElement('textarea');
      el.value = jsonStr;
      document.body.appendChild(el);
      el.select();
      document.execCommand('copy');
      document.body.removeChild(el);
      
      const status = document.getElementById("status-msg");
      status.innerText = "✅ JSON copied to clipboard!";
      status.style.display = 'inline';
      setTimeout(() => {{ status.style.display = 'none'; }}, 3000);
  }}
</script>
</body>
</html>
"""


# ==============================================================================
# MAIN PAGE
# ==============================================================================

def main():
    st.set_page_config(page_title="Visual Model Builder", page_icon="🧪", layout="wide")
    init_user_preferences(USER_PREFS_FILE)
    apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))
    
    st.title("🧪 Visual Model Builder")
    st.caption(t("visual_builder.subtitle"))

    components.html(DRAWFLOW_HTML, height=1100, scrolling=False)

    st.divider()
    st.caption(t("visual_builder.footer"))

if __name__ == "__main__":
    main()
