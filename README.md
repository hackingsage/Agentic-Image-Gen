
# 🧠 Agentic Image Generator (SDXL + Streamlit)

A **product-style, agentic image generation web app** built using **Streamlit** and **Stable Diffusion XL (SDXL)**.  
This project goes beyond “prompt → image” by introducing an **agent workflow** that plans, critiques, and refines prompts, and supports advanced tools like **ControlNet** and **Inpainting**.

---

## ✨ Features

### 🧠 Agentic Prompt Workflow
- Planner → Prompt Engineer → Critic → Refiner
- Produces **final refined prompts** and **negative prompts**
- Optional **Prompt Studio override** for manual prompt control

### 🖼️ Image Generation Modes
✅ **Text-to-Image** (SDXL)  
✅ **Image-to-Image** (Upload + transform)  
✅ **ControlNet** (Canny / Depth control)  
✅ **Inpainting** (Upload base + mask → edit only masked area)

### 🗂️ Persistent Outputs (Saved on Disk)
Every generation is saved to:
- `outputs/<run_id>/image_*.png`
- `outputs/<run_id>/meta.json`
- `outputs/generations.jsonl` (run index)

This makes the app function like a real generation tool with history.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Image Generation**: Stable Diffusion XL (Diffusers)
- **Agentic Logic**: rule-based multi-step agent loop (LLM upgrade possible)
- **Persistence**: local disk storage (outputs + JSON metadata)
- **Libraries**: PyTorch, Diffusers, Accelerate, Transformers

---

## 📸 Screenshots / Demo

![Demo](/docs/demo.gif)

---

## ⚡ Installation

### 1) Clone repo
```bash
git clone https://github.com/hackingsage/Agentic-Image-Gen.git
cd Agentic-Image-Gen
```

### 2) Create environment
```bash
python -m venv .venv
```

Activate:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux / macOS:
```bash
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧠 Recommended GPU Setup

This project works best with CUDA GPU.

### Optional: xFormers (speed + lower VRAM)
```bash
pip install xformers
```

---

## 🧪 How It Works (High Level)

### 1) User provides a goal
Example:
> "Cinematic astronaut portrait with rim light"

### 2) Agent loop runs
- Planner chooses style
- Prompt Engineer constructs SDXL prompt
- Critic adds improvements
- Refiner produces final prompt

### 3) Generation pipeline runs
Depends on selected mode:
- SDXL txt2img
- SDXL img2img
- SDXL + ControlNet (canny/depth)
- SDXL inpainting

### 4) Outputs saved
Saved with reproducible metadata including:
- prompts
- settings
- run_id
- runtime
- mode/tool settings

---

## 📂 Project Structure

```
agentic-image-generator/
├─ app.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ outputs/
│  ├─ generations.jsonl
│  └─ <run_id>/
│      ├─ meta.json
│      ├─ image_1.png
│      └─ ...
└─ src/
   ├─ ui.py
   ├─ storage.py
   ├─ agent_loop.py
   ├─ pipeline_sdxl.py
   ├─ pipeline_controlnet.py
   ├─ pipeline_inpaint.py
   ├─ utils.py
   ├─ presets.py
   └─ safety.py
```

---

## ✅ Notes / Common Issues

### Outputs not visible?
Outputs are saved under:
```
outputs/
```

If you cannot find them, ensure `src/storage.py` uses an **absolute project-root output path**, not a relative one.

### Closing browser tab stops generation?
Streamlit sessions can stop when WebSocket disconnects.
For long generations, keep the tab open during inference

---

## 🚀 Future Improvements

- True background generation worker (FastAPI + Celery/RQ)
- Multi-agent voting (generate multiple prompt candidates + choose best)
- LoRA selector + weight slider
- Image quality scoring (CLIP aesthetic scoring)
- Full “History Gallery” page with filters
- Cloud deployment with GPU inference

---

## 📜 Model / License Info

- Image model: **Stable Diffusion XL (SDXL 1.0)**
- This project uses open diffusion tooling from the HuggingFace ecosystem.

Please review SDXL and any additional model licenses before commercial deployment.

---

## 🙌 Author

Built by **Arul Tripathi**  
If you like this project, ⭐ the repo and feel free to connect!

---
