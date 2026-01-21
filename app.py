import time, json
import streamlit as st
from PIL import Image
import os
from src.ui import apply_theme, hero, side_header, side_card_start, side_card_end, side_tip
from src.utils import pil_to_bytes, zip_images
from src.safety import is_blocked_prompt
from src.agent_loop import run_agent_loop
from src.pipeline_sdxl import txt2img, img2img
from src.pipeline_controlnet import controlnet_generate
from src.pipeline_inpaint import inpaint
from src.storage import new_run_id, save_run, load_index, load_run_meta


# ----------------- PAGE -----------------
st.set_page_config(
    page_title="Agentic Image Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_theme()
hero()


# ----------------- HELPERS -----------------
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def apply_preset(preset: str):
    """
    Correct Streamlit way: update widget-backed keys via callback.
    """
    if preset == "fast":
        st.session_state.steps = 20
        st.session_state.guidance = 5.5
        st.session_state.num_images = 1
        st.session_state.width = 768
        st.session_state.height = 768
    elif preset == "best":
        st.session_state.steps = 35
        st.session_state.guidance = 6.5
        st.session_state.num_images = 2
        st.session_state.width = 1024
        st.session_state.height = 1024
    elif preset == "hq":
        st.session_state.steps = 50
        st.session_state.guidance = 7.5
        st.session_state.num_images = 4
        st.session_state.width = 1024
        st.session_state.height = 1024


def mode_card(mode: str):
    MODE_INFO = {
        "Text-to-Image": ("🪄", "Generate from prompt only", "No uploads needed. Best for fresh ideas."),
        "Image-to-Image": ("🖼️", "Transform an uploaded image", "Best for variations & restyling."),
        "ControlNet": ("🧩", "Lock composition with edges/depth", "Best when structure must match reference."),
        "Inpainting": ("🩹", "Edit only masked region", "Best for fixing faces/objects/background."),
    }
    icon, title, desc = MODE_INFO[mode]
    st.markdown(
        f"""
        <div class="side-card">
          <div style="font-size:16px; font-weight:850;">{icon} {title}</div>
          <div style="opacity:0.82; font-size:13px; margin-top:6px; line-height:1.3;">
            {desc}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def responsive_gallery(images, meta):
    """
    Gallery grid + select one preview (more product-y).
    """
    if not images:
        return

    mobile = st.session_state.get("mobile_mode", False)

    ss("selected_idx", 0)
    cols = 1 if mobile else 2

    st.markdown("### 🖼️ Gallery")
    grid = st.columns(cols, gap="medium")
    for i, im in enumerate(images):
        with grid[i % cols]:
            st.image(im, use_container_width=True)
            if st.button("Select", key=f"select_{meta.get('run_id','run')}_{i}", use_container_width=True):
                st.session_state["selected_idx"] = i

    sel = int(st.session_state.get("selected_idx", 0))
    sel = max(0, min(sel, len(images) - 1))

    st.markdown("### 🔍 Preview")
    st.image(images[sel], use_container_width=True)

    st.download_button(
        "⬇️ Download Selected",
        data=pil_to_bytes(images[sel]),
        file_name=f"{meta.get('run_id','run')}_selected.png",
        mime="image/png",
        use_container_width=True,
    )


# ----------------- STATE DEFAULTS -----------------
ss("latest_images", [])
ss("latest_meta", {})
ss("latest_control_preview", None)

ss("is_generating", False)

# prompt studio state
ss("studio_enabled", True)
ss("studio_use_manual", False)
ss("studio_manual_prompt", "")
ss("studio_manual_negative", "")

# generation settings
ss("steps", 30)
ss("guidance", 6.5)
ss("num_images", 1)
ss("seed", -1)
ss("width", 1024)
ss("height", 1024)

# tool settings
ss("img2img_strength", 0.65)
ss("cn_kind", "Canny")
ss("cn_strength", 0.80)
ss("inpaint_strength", 0.75)

# responsive UI
ss("mobile_mode", False)
ss("mode", "Text-to-Image")


# ----------------- SIDEBAR -----------------
with st.sidebar:
    # Layout control
    side_header("Layout", "📱")
    side_card_start()
    st.toggle("Compact layout", key="mobile_mode")
    side_tip("Recommended for mobile/small screens.")
    side_card_end()

    mobile = st.session_state.get("mobile_mode", False)

    # Sidebar routing
    if mobile:
        tab_choice = st.selectbox("Sidebar", ["⚡ Generate", "🧪 Studio", "🖼️ Gallery", "🗂️ History"])
        tabs = None
    else:
        tabs = st.tabs(["⚡ Generate", "🧪 Studio", "🖼️ History", "🗂️ Gallery"])
        tab_choice = None

    def render_gallery():
        side_header("Gallery", "🖼️")
        side_card_start()
        limit = st.slider("Load last N runs", 10, 200, 60, key="gallery_limit")
        cols = st.selectbox("Grid columns", [2, 3, 4], index=1, key="gallery_cols")
        side_tip("Click any run to load it on the main page.")
        side_card_end()

        index = load_index(limit=limit)
        if not index:
            st.info("No outputs saved yet. Generate something first ✅")
            return

        # optional search/filter
        query = st.text_input("Search (goal/style/mode/run_id)", "", key="gallery_query").strip().lower()

        filtered = []
        for row in index:
            blob = f"{row.get('goal','')} {row.get('style','')} {row.get('mode','')} {row.get('run_id','')}".lower()
            if not query or query in blob:
                filtered.append(row)

        st.caption(f"Showing **{len(filtered)}** / {len(index)} runs")

        grid = st.columns(cols)
        for i, row in enumerate(filtered):
            rid = row["run_id"]
            meta = load_run_meta(rid)
            if not meta:
                continue

            # pick thumbnail (first image)
            thumb_path = None
            if meta.get("image_paths"):
                thumb_path = meta["image_paths"][0]

            with grid[i % cols]:
                if thumb_path and os.path.exists(thumb_path):
                    st.image(thumb_path, use_container_width=True)
                else:
                    st.info("No image preview")

                st.markdown(f"**{meta.get('mode','')}**")
                st.caption(f"{meta.get('timestamp','')} · style `{meta.get('agent',{}).get('style','')}`")
                st.caption(str(meta.get("goal", ""))[:90])

                if st.button("Load", key=f"gallery_load_{rid}", use_container_width=True):
                    st.session_state["latest_meta"] = meta
                    st.success("Loaded ✅")
                    st.rerun()

    # ---------------- TAB: Generate ----------------
    def render_generate_sidebar():
        side_header("1) Mode", "🧠")
        side_card_start()
        mode = st.selectbox(
            "Mode",
            ["Text-to-Image", "Image-to-Image", "ControlNet", "Inpainting"],
            key="mode",
            label_visibility="collapsed",
        )
        side_card_end()
        mode_card(mode)

        side_header("2) Quality", "🎛️")
        side_card_start()
        st.slider("Steps", 10, 60, key="steps", disabled=st.session_state["is_generating"])
        st.slider("Guidance", 1.0, 15.0, key="guidance", disabled=st.session_state["is_generating"])
        st.selectbox("Batch", [1, 2, 4], key="num_images", disabled=st.session_state["is_generating"])
        st.number_input("Seed (-1 random)", key="seed", disabled=st.session_state["is_generating"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("⚡ Fast", use_container_width=True, on_click=apply_preset, args=("fast",), disabled=st.session_state["is_generating"])
        with c2:
            st.button("✨ Best", use_container_width=True, on_click=apply_preset, args=("best",), disabled=st.session_state["is_generating"])
        with c3:
            st.button("🧪 HQ", use_container_width=True, on_click=apply_preset, args=("hq",), disabled=st.session_state["is_generating"])
        side_tip("Presets adjust steps/guidance/batch/resolution.")
        side_card_end()

        side_header("3) Canvas", "📐")
        side_card_start()
        st.selectbox("Width", [768, 1024], key="width", disabled=st.session_state["is_generating"])
        st.selectbox("Height", [768, 1024], key="height", disabled=st.session_state["is_generating"])
        side_tip("1024×1024 gives best SDXL quality.")
        side_card_end()

        side_header("Tools", "🧰")
        side_card_start()
        if mode == "Image-to-Image":
            st.slider("Img2Img strength", 0.10, 0.95, key="img2img_strength", disabled=st.session_state["is_generating"])
            side_tip("0.2–0.4 preserve reference, 0.7+ changes a lot.")
        elif mode == "ControlNet":
            st.selectbox("Control type", ["Canny", "Depth"], key="cn_kind", disabled=st.session_state["is_generating"])
            st.slider("Control strength", 0.05, 1.0, key="cn_strength", disabled=st.session_state["is_generating"])
            side_tip("Canny=edges, Depth=structure.")
        elif mode == "Inpainting":
            st.slider("Inpaint strength", 0.10, 0.95, key="inpaint_strength", disabled=st.session_state["is_generating"])
            side_tip("White mask area will be edited.")
        side_card_end()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Primary CTA
        generate_btn = st.button(
            "✨ Generate",
            use_container_width=True,
            disabled=st.session_state["is_generating"]
        )

        st.markdown(
            "<div style='opacity:0.7; font-size:12px; margin-top:12px;'>Built by Arul • Agentic Pipeline</div>",
            unsafe_allow_html=True
        )

        return generate_btn

    # ---------------- TAB: Studio ----------------
    def render_studio_sidebar():
        side_header("Prompt Studio", "🧪")
        side_card_start()
        st.toggle("Enable Prompt Studio", key="studio_enabled")
        st.toggle("Override agent prompts manually", key="studio_use_manual")
        st.markdown("---")
        st.write("**Manual Prompt**")
        st.text_area(
            "manual_prompt",
            key="studio_manual_prompt",
            height=140,
            label_visibility="collapsed",
            placeholder="Write your own prompt here..."
        )
        st.write("**Manual Negative Prompt**")
        st.text_area(
            "manual_negative",
            key="studio_manual_negative",
            height=110,
            label_visibility="collapsed",
            placeholder="watermark, blurry, low quality, distorted..."
        )
        side_tip("For power users: manual control like a diffusion artist.")
        side_card_end()

    # ---------------- TAB: History ----------------
    def render_history_sidebar():
        side_header("Persistent History", "🗂️")
        side_card_start()
        limit = st.slider("Load last N runs", 5, 100, 30, key="history_limit")
        index = load_index(limit=limit)
        st.write(f"Found **{len(index)}** runs on disk.")
        side_card_end()

        if not index:
            st.info("No history yet. Generate something first ✅")
            return

        for row in index[:10]:
            rid = row.get("run_id")
            side_card_start()
            st.write(f"**{row.get('mode')}** · `{row.get('timestamp')}`")
            st.caption(f"Style: {row.get('style')}")
            st.caption(f"Goal: {str(row.get('goal',''))[:70]}")

            if st.button(f"Load {rid}", key=f"load_{rid}", use_container_width=True):
                meta = load_run_meta(rid)
                if meta:
                    st.session_state["latest_meta"] = meta
                    st.success("Loaded run ✅")
                else:
                    st.error("Could not load meta.json for that run.")
            side_card_end()

    # render correct sidebar
    if tabs:
        with tabs[0]:
            generate_btn = render_generate_sidebar()
        with tabs[1]:
            render_studio_sidebar()
        with tabs[2]:
            render_history_sidebar()
        with tabs[3]:
            render_gallery()
    else:
        if tab_choice == "⚡ Generate":
            generate_btn = render_generate_sidebar()
        elif tab_choice == "🧪 Studio":
            generate_btn = False
            render_studio_sidebar()
        elif tab_choice == "🖼️ Gallery":
            generate_btn = False
            render_gallery()
        else:
            generate_btn = False
            render_history_sidebar()

# ----------------- MAIN CONTENT (Responsive) -----------------
mobile = st.session_state.get("mobile_mode", False)
mode = st.session_state.get("mode", "Text-to-Image")

if mobile:
    left = st.container()
    right = st.container()
else:
    left, right = st.columns([1, 1], gap="large")


# ----------------- LEFT: Inputs -----------------
with left:
    st.subheader("✅ Step 1 — Describe your goal")
    goal = st.text_area(
        "Goal",
        height=140,
        placeholder="Example: Cinematic astronaut portrait, rim light, shallow depth of field",
        key="goal"
    )

    st.subheader("📤 Step 2 — Upload (only if needed)")
    ref_img = None
    mask_img = None

    if mode in ["Image-to-Image", "ControlNet"]:
        up = st.file_uploader("Reference Image", type=["png", "jpg", "jpeg"], key="ref_upload")
        if up:
            ref_img = Image.open(up).convert("RGB")
            st.image(ref_img, use_container_width=True)

    if mode == "Inpainting":
        up1 = st.file_uploader("Base image", type=["png", "jpg", "jpeg"], key="base_upload")
        up2 = st.file_uploader("Mask image (white=edit region)", type=["png", "jpg", "jpeg"], key="mask_upload")
        if up1:
            ref_img = Image.open(up1).convert("RGB")
            st.image(ref_img, caption="Base image", use_container_width=True)
        if up2:
            mask_img = Image.open(up2).convert("RGB")
            st.image(mask_img, caption="Mask image", use_container_width=True)


# ----------------- RIGHT: Agent + Output -----------------
with right:
    st.subheader("✨ Step 3 — Generate + Results")

    if generate_btn:
        if st.session_state["is_generating"]:
            st.warning("Generation already running…")
            st.stop()

        if is_blocked_prompt(goal):
            st.error("Blocked prompt. Please modify.")
            st.stop()

        st.session_state["is_generating"] = True
        progress = st.progress(0, text="Starting...")

        try:
            progress.progress(15, text="Agent planning prompt...")
            agent = run_agent_loop(goal)

            final_prompt = agent["final_prompt"]
            final_negative = agent["final_negative_prompt"]

            # Prompt studio override
            if st.session_state["studio_enabled"] and st.session_state["studio_use_manual"]:
                mp = st.session_state["studio_manual_prompt"].strip()
                mn = st.session_state["studio_manual_negative"].strip()
                if mp:
                    final_prompt = mp
                if mn:
                    final_negative = mn

            with st.expander("🧠 Agent details (optional)", expanded=False):
                st.write(f"**Style:** `{agent.get('style')}`")
                st.write("**Critic:**", agent.get("critique", {}).get("critique", ""))
                st.write("**Final Prompt**")
                st.code(final_prompt)
                st.write("**Final Negative Prompt**")
                st.code(final_negative)

            meta = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "goal": goal,
                "agent": agent,
                "settings": {
                    "steps": int(st.session_state["steps"]),
                    "guidance": float(st.session_state["guidance"]),
                    "seed": int(st.session_state["seed"]),
                    "num_images": int(st.session_state["num_images"]),
                    "width": int(st.session_state["width"]),
                    "height": int(st.session_state["height"]),
                }
            }

            progress.progress(55, text="Diffusion sampling (generating images)...")
            control_preview = None
            t0 = time.time()

            if mode == "Text-to-Image":
                images = txt2img(
                    final_prompt, final_negative,
                    meta["settings"]["width"], meta["settings"]["height"],
                    meta["settings"]["steps"], meta["settings"]["guidance"],
                    meta["settings"]["seed"], meta["settings"]["num_images"]
                )

            elif mode == "Image-to-Image":
                if ref_img is None:
                    st.error("Upload image for img2img.")
                    st.stop()

                strength = float(st.session_state["img2img_strength"])
                meta["settings"]["img2img_strength"] = strength

                images = img2img(
                    final_prompt, final_negative,
                    ref_img, strength,
                    meta["settings"]["steps"], meta["settings"]["guidance"],
                    meta["settings"]["seed"], meta["settings"]["num_images"]
                )

            elif mode == "ControlNet":
                if ref_img is None:
                    st.error("Upload image for ControlNet.")
                    st.stop()

                cn_kind = st.session_state["cn_kind"]
                cn_strength = float(st.session_state["cn_strength"])
                meta["settings"]["controlnet"] = {"kind": cn_kind, "strength": cn_strength}

                images, control_preview = controlnet_generate(
                    cn_kind,
                    final_prompt, final_negative,
                    ref_img,
                    meta["settings"]["width"], meta["settings"]["height"],
                    meta["settings"]["steps"], meta["settings"]["guidance"],
                    meta["settings"]["seed"], meta["settings"]["num_images"],
                    control_strength=cn_strength
                )

            else:  # Inpainting
                if ref_img is None or mask_img is None:
                    st.error("Upload base + mask image for inpainting.")
                    st.stop()

                strength = float(st.session_state["inpaint_strength"])
                meta["settings"]["inpaint_strength"] = strength

                images = inpaint(
                    final_prompt, final_negative,
                    ref_img, mask_img,
                    meta["settings"]["steps"], meta["settings"]["guidance"],
                    meta["settings"]["seed"],
                    strength=strength
                )

            meta["runtime_seconds"] = time.time() - t0

            progress.progress(85, text="Saving run...")
            run_id = new_run_id()
            meta["run_id"] = run_id
            run_dir = save_run(run_id, meta, images, control_preview=control_preview)

            progress.progress(100, text="Done ✅")

            st.success(f"Done in {meta['runtime_seconds']:.2f}s ✅")
            st.caption(f"Saved to disk: `{run_dir}`")

            st.session_state["latest_images"] = images
            st.session_state["latest_meta"] = meta
            st.session_state["latest_control_preview"] = control_preview

        finally:
            st.session_state["is_generating"] = False

    # ----------------- SHOW LATEST -----------------
    meta = st.session_state.get("latest_meta", {})
    images = st.session_state.get("latest_images", [])
    control_preview = st.session_state.get("latest_control_preview")

    if meta:
        st.markdown("### 🧾 Run Summary")
        st.markdown(
            f"- **Mode:** `{meta.get('mode')}`  \n"
            f"- **Style:** `{meta.get('agent', {}).get('style')}`  \n"
            f"- **Run ID:** `{meta.get('run_id')}`  \n"
            f"- **Runtime:** `{meta.get('runtime_seconds', 0):.2f}s`"
        )

    if control_preview is not None:
        st.markdown("### 🧩 Control Preview")
        st.image(control_preview, use_container_width=True)

    if images:
        responsive_gallery(images, meta)

        st.markdown("### ⬇️ Downloads")
        st.download_button(
            "⬇️ Download ALL (ZIP)",
            data=zip_images(images),
            file_name=f"{meta.get('run_id','run')}_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download Metadata JSON",
            data=json.dumps(meta, indent=2),
            file_name=f"{meta.get('run_id','run')}_meta.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Generate something to see outputs here ✅")
