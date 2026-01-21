import streamlit as st

def apply_theme():
    st.markdown(
        """
        <style>
        /* ===============================
           PAGE / BASE
           =============================== */

        .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }

        /* DO NOT hide header -> sidebar reopen button lives there */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Soft background glow */
        .stApp {
          background: radial-gradient(circle at 10% 20%, rgba(80,120,255,0.12), transparent 35%),
                      radial-gradient(circle at 90% 10%, rgba(255,80,200,0.10), transparent 40%),
                      radial-gradient(circle at 50% 90%, rgba(80,255,200,0.10), transparent 40%);
        }

        /* ===============================
           HERO
           =============================== */
        .hero {
            padding: 22px 22px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 35px rgba(0,0,0,0.25);
            margin-bottom: 1.2rem;
        }
        .hero h1 { margin: 0; font-size: 2.0rem; }
        .hero p { margin: 6px 0 0 0; opacity: 0.85; }

        /* ===============================
           BADGES
           =============================== */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(0,0,0,0.15);
            margin-right: 6px;
        }

        /* ===============================
           BUTTONS + INPUTS
           =============================== */
        div.stButton > button {
            border-radius: 16px;
            padding: 0.7rem 1rem;
            font-weight: 750;
        }

        textarea, input, select {
            border-radius: 14px !important;
        }

        img { border-radius: 18px; }

        /* ===============================
           SIDEBAR
           =============================== */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }

        @media (min-width: 769px) {
            section[data-testid="stSidebar"] {
                min-width: 320px;
                max-width: 360px;
            }
        }

        @media (max-width: 768px) {
            section[data-testid="stSidebar"] {
                min-width: unset !important;
                max-width: unset !important;
                width: auto !important;
            }
        }

        .side-title {
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            opacity: 0.85;
            margin: 0.2rem 0 0.7rem 0;
        }

        .side-card {
            padding: 12px 12px;
            border-radius: 18px;
            background: rgba(0,0,0,0.20);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 10px 25px rgba(0,0,0,0.12);
            margin-bottom: 10px;
        }

        .side-card:hover {
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: 0 14px 40px rgba(0,0,0,0.18);
            transform: translateY(-1px);
            transition: all 0.2s ease;
        }

        section[data-testid="stSidebar"] [data-testid="stSlider"] {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }

        .streamlit-expanderHeader {
            border-radius: 14px !important;
        }

        /* ===============================
           RESPONSIVE
           =============================== */
        @media (max-width: 1100px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1.2rem !important;
                padding-left: 0.8rem !important;
                padding-right: 0.8rem !important;
            }

            .hero {
                padding: 14px 14px !important;
                border-radius: 18px !important;
            }

            .hero h1 { font-size: 1.35rem !important; }
            .hero p { font-size: 0.92rem !important; }

            .badge {
                margin-top: 6px;
                margin-bottom: 4px;
                font-size: 11px !important;
            }

            div.stButton > button {
                padding: 0.75rem 1rem !important;
                border-radius: 18px !important;
                font-size: 1rem !important;
            }
        }

        @media (max-width: 420px) {
            .hero h1 { font-size: 1.2rem !important; }
            .hero p { font-size: 0.85rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def hero():
    st.markdown(
        """
        <div class="hero">
            <div style="display:flex; gap:10px; align-items:center;">
              <div style="font-size:28px;">🧠</div>
              <h1>Agentic Image Generator</h1>
            </div>
            <p>SDXL + Multi-step Prompt Agents + ControlNet + Inpainting • Streamlit Product UI</p>
            <div style="margin-top:10px;">
                <span class="badge">Planner</span>
                <span class="badge">Prompt Engineer</span>
                <span class="badge">Critic</span>
                <span class="badge">Refiner</span>
                <span class="badge">ControlNet</span>
                <span class="badge">Inpaint</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def side_header(title: str, icon: str = "⚙️"):
    st.markdown(f"<div class='side-title'>{icon} {title}</div>", unsafe_allow_html=True)


def side_card_start():
    st.markdown("<div class='side-card'>", unsafe_allow_html=True)


def side_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def side_tip(text: str):
    st.caption(f"💡 {text}")
