#!/usr/bin/env python3
"""Streamlit interface for an educational news-text classification system."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import streamlit as st

from advanced_models import advanced_component_availability
from detect_fake_news import classify_probability
from model_compat import load_pipeline as load_model_pipeline

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="NewsCheck",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --primary-color: #54f2e3;
        --ink: #edf7ff;
        --muted: #91a5bb;
        --cyan: #54f2e3;
        --blue: #62a8ff;
        --pink: #ff5eae;
        --panel: rgba(12, 23, 40, 0.82);
        --line: rgba(117, 169, 207, 0.17);
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", "Arial", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 78% 0%, rgba(27, 110, 157, 0.17), transparent 33rem),
            radial-gradient(ellipse at 0% 22%, rgba(113, 42, 122, 0.11), transparent 28rem),
            #070b14;
    }

    [data-testid="stAppViewContainer"]::before {
        position: fixed;
        z-index: 0;
        pointer-events: none;
        inset: 0;
        content: "";
        opacity: 0.13;
        background-image:
            linear-gradient(rgba(120, 177, 213, 0.12) 1px, transparent 1px),
            linear-gradient(90deg, rgba(120, 177, 213, 0.12) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: linear-gradient(to bottom, black, transparent 80%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMain"] {
        position: relative;
        z-index: 1;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2.6rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #080d17 100%);
    }

    [data-testid="stSidebar"] > div:first-child {
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: #adbed0;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--ink);
        letter-spacing: 0.02em;
    }

    .sidebar-brand {
        padding: 0.4rem 0 1rem;
    }

    .sidebar-brand-name {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: -0.04em;
    }

    .sidebar-brand-name span {
        color: var(--cyan);
    }

    .sidebar-tagline {
        margin-top: 0.25rem;
        color: var(--muted);
        font-size: 0.77rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        margin: 0.3rem 0 1.6rem;
        padding: clamp(1.7rem, 4vw, 3.2rem);
        border: 1px solid rgba(84, 242, 227, 0.22);
        border-radius: 22px;
        background:
            linear-gradient(112deg, rgba(15, 32, 49, 0.96), rgba(12, 17, 33, 0.93) 60%, rgba(42, 19, 48, 0.76));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.25), inset 0 1px rgba(255, 255, 255, 0.04);
    }

    .hero-shell::after {
        position: absolute;
        top: -8rem;
        right: -4rem;
        width: 23rem;
        height: 23rem;
        border: 1px solid rgba(84, 242, 227, 0.18);
        border-radius: 50%;
        box-shadow: 0 0 0 2.5rem rgba(84, 242, 227, 0.025), 0 0 0 5rem rgba(98, 168, 255, 0.025);
        content: "";
        pointer-events: none;
    }

    .hero-kicker, .section-kicker {
        color: var(--cyan);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        margin: 0.8rem 0 0.65rem;
        color: var(--ink);
        font-size: clamp(2.15rem, 5vw, 4rem);
        font-weight: 760;
        letter-spacing: -0.065em;
        line-height: 1.04;
    }

    .hero-title span {
        color: var(--cyan);
        text-shadow: 0 0 28px rgba(84, 242, 227, 0.28);
    }

    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 670px;
        margin: 0;
        color: #a9bbcf;
        font-size: 1rem;
        line-height: 1.7;
    }

    .hero-status {
        position: relative;
        z-index: 1;
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        margin-top: 1.35rem;
        padding: 0.45rem 0.72rem;
        border: 1px solid rgba(84, 242, 227, 0.2);
        border-radius: 999px;
        background: rgba(84, 242, 227, 0.06);
        color: #c4fff7;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--cyan);
        box-shadow: 0 0 12px var(--cyan);
    }

    .section-kicker {
        margin: 0.5rem 0 0.35rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #a9bbcf;
    }

    [data-testid="stTextArea"] textarea {
        min-height: 240px;
        border: 1px solid rgba(98, 168, 255, 0.26);
        border-radius: 14px;
        background: rgba(7, 13, 24, 0.9);
        color: var(--ink);
        line-height: 1.7;
        box-shadow: inset 0 0 24px rgba(45, 91, 131, 0.08);
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--cyan);
        box-shadow: 0 0 0 1px var(--cyan), 0 0 22px rgba(84, 242, 227, 0.09);
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #657b92;
    }

    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(18, 33, 52, 0.88), rgba(11, 19, 33, 0.9));
        box-shadow: inset 0 1px rgba(255, 255, 255, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #91a5bb;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 700;
    }

    [data-testid="stButton"] > button {
        min-height: 3.15rem;
        border: 1px solid rgba(84, 242, 227, 0.55);
        border-radius: 11px;
        background: linear-gradient(100deg, #1ec7ba, #388edc);
        color: #041217;
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 8px 28px rgba(39, 197, 190, 0.16);
        transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
    }

    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        border-color: #a0fff5;
        color: #031016;
        filter: brightness(1.08);
        box-shadow: 0 12px 34px rgba(39, 197, 190, 0.25);
    }

    [data-testid="stButton"] > button:focus {
        box-shadow: 0 0 0 2px #070b14, 0 0 0 4px var(--cyan);
    }

    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--cyan), var(--blue), var(--pink));
    }

    [data-testid="stAlert"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(15, 26, 42, 0.84);
    }

    [data-testid="stSlider"] [role="slider"] {
        border-color: var(--cyan);
        box-shadow: 0 0 12px rgba(84, 242, 227, 0.35);
    }

    [data-testid="stDivider"] {
        border-color: var(--line);
    }

    .tech-card {
        height: 100%;
        padding: 1.25rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(16, 29, 47, 0.9), rgba(10, 16, 29, 0.88));
    }

    .tech-index {
        color: var(--cyan);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }

    .tech-title {
        margin: 0.55rem 0;
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 700;
    }

    .tech-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .evidence-card {
        padding: 1.25rem;
        border: 1px solid rgba(84, 242, 227, 0.28);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(13, 29, 43, 0.96), rgba(13, 17, 32, 0.94));
        box-shadow: 0 16px 44px rgba(0, 0, 0, 0.2);
    }

    .evidence-label {
        color: var(--cyan);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .evidence-prediction {
        margin: 0.5rem 0;
        color: var(--ink);
        font-size: clamp(1.3rem, 3vw, 2rem);
        font-weight: 750;
    }

    .evidence-prediction.credible {
        color: var(--cyan);
    }

    .evidence-prediction.misleading {
        color: var(--pink);
    }

    .evidence-prediction.uncertain {
        color: #ffd479;
    }

    .evidence-claim {
        margin: 0.7rem 0 0;
        padding: 0.85rem 1rem;
        border-left: 2px solid var(--blue);
        border-radius: 0 8px 8px 0;
        background: rgba(5, 12, 23, 0.6);
        color: #bac9d9;
        line-height: 1.65;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .indicator-chip {
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
        padding: 0.4rem 0.65rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(98, 168, 255, 0.08);
        color: #c9dbed;
        font-size: 0.78rem;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.35rem 1rem 2rem;
        }

        .hero-shell {
            border-radius: 16px;
        }

        .hero-shell::after {
            right: -14rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }

    :root {
        color-scheme: light;
        --primary-color: #276b59;
        --ink: #24342f;
        --muted: #64746d;
        --cyan: #276b59;
        --blue: #4b7969;
        --pink: #a84c4c;
        --panel: #ffffff;
        --line: #e2e9e4;
    }

    [data-testid="stAppViewContainer"] {
        color: var(--ink);
        background: #f5f7f4;
    }

    [data-testid="stAppViewContainer"]::before {
        display: none;
    }

    [data-testid="stHeader"] {
        background: #f5f7f4;
    }

    .block-container {
        max-width: 940px;
        padding: 5rem 1.5rem 3rem;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
    }

    [data-testid="stSidebar"] > div:first-child {
        border-right: 1px solid #e7ece8;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #56675f;
    }

    h1, h2, h3,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #24342f;
        letter-spacing: -0.025em;
    }

    .sidebar-brand {
        padding: 0.2rem 0 0.5rem;
    }

    .sidebar-brand-name {
        color: #24342f;
        font-size: 1.45rem;
        letter-spacing: -0.04em;
    }

    .sidebar-brand-name span {
        color: #276b59;
    }

    .sidebar-tagline {
        color: #718078;
        font-size: 0.82rem;
        letter-spacing: 0;
        text-transform: none;
    }

    .hero-shell {
        margin: 0 0 1.5rem;
        padding: 0;
        border: 0;
        border-radius: 0;
        background: transparent;
        box-shadow: none;
    }

    .hero-shell::after,
    .hero-kicker,
    .hero-status {
        display: none;
    }

    .hero-title {
        margin: 0 0 0.65rem;
        color: #24342f;
        font-size: clamp(2rem, 5vw, 2.9rem);
        font-weight: 720;
        letter-spacing: -0.055em;
        line-height: 1.12;
    }

    .hero-title span {
        color: #276b59;
        text-shadow: none;
    }

    .hero-copy {
        max-width: 650px;
        color: #5d6d65;
        font-size: 1rem;
    }

    [data-testid="stTextArea"] textarea {
        min-height: 190px;
        border: 0;
        border-radius: 11px;
        background: #ffffff;
        color: #24342f;
        line-height: 1.65;
        box-shadow: none;
    }

    [data-testid="stTextAreaRootElement"] {
        overflow: hidden;
        border: 1px solid #d8e1da;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: none;
    }

    [data-testid="stTextAreaRootElement"]:focus-within {
        border-color: #43816d;
        box-shadow: 0 0 0 3px rgba(67, 129, 109, 0.14);
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #88958e;
    }

    [data-testid="stButton"] > button {
        min-height: 2.9rem;
        border: 1px solid #276b59;
        border-radius: 9px;
        background: #276b59;
        color: #ffffff;
        font-size: 1rem;
        font-weight: 650;
        letter-spacing: 0;
        text-transform: none;
        box-shadow: none;
        transition: background 120ms ease, border-color 120ms ease;
    }

    [data-testid="stButton"] > button:hover {
        transform: none;
        border-color: #1f5949;
        background: #1f5949;
        color: #ffffff;
        box-shadow: none;
    }

    [data-testid="stButton"] > button:focus {
        box-shadow: 0 0 0 3px rgba(67, 129, 109, 0.2);
    }

    [data-testid="stButton"] > button [data-testid="stMarkdownContainer"] p {
        color: #ffffff;
    }

    [data-testid="stMetric"] {
        padding: 0.8rem 1rem;
        border: 1px solid #e3e9e4;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: none;
    }

    [data-testid="stMetricLabel"] {
        color: #64746d;
        font-size: 0.78rem;
        letter-spacing: 0;
        text-transform: none;
    }

    [data-testid="stMetricValue"] {
        color: #24342f;
    }

    [data-testid="stProgress"] > div > div {
        background: #43816d;
    }

    [data-testid="stAlert"] {
        border: 1px solid #e3e9e4;
        border-radius: 10px;
        background: #ffffff;
    }

    [data-testid="stSlider"] [role="slider"] {
        border-color: #276b59;
        box-shadow: none;
    }

    [data-testid="stDivider"] {
        border-color: #e3e9e4;
    }

    .evidence-card {
        padding: 1.35rem;
        border: 1px solid #e0e8e2;
        border-left: 4px solid #789184;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 4px 16px rgba(30, 50, 40, 0.04);
    }

    .evidence-label {
        color: #64746d;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0;
        text-transform: none;
    }

    .evidence-prediction {
        margin: 0.5rem 0 0.3rem;
        color: #24342f;
        font-size: clamp(1.45rem, 3vw, 2rem);
    }

    .evidence-prediction.credible {
        color: #276b59;
    }

    .evidence-prediction.misleading {
        color: #a44343;
    }

    .evidence-prediction.uncertain {
        color: #8a641f;
    }

    .evidence-claim {
        margin: 1rem 0 0;
        padding: 0.8rem 0 0;
        border-top: 1px solid #edf0ed;
        border-left: 0;
        border-radius: 0;
        background: transparent;
        color: #4b5d54;
        line-height: 1.65;
    }

    .indicator-chip {
        border-color: #e1e8e3;
        background: #f4f7f4;
        color: #40584b;
    }

    .tech-card {
        border-color: #e3e9e4;
        border-radius: 10px;
        background: #ffffff;
    }

    .tech-index,
    .tech-title {
        color: #276b59;
    }

    [data-testid="stExpander"] {
        border-color: #e3e9e4;
        border-radius: 10px;
        background: #ffffff;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 5rem 1rem 2rem;
        }

        .hero-shell {
            border-radius: 0;
        }
    }

    :root {
        color-scheme: dark;
        --primary-color: #8d82ff;
        --ink: #f1f2ff;
        --muted: #a5a8c3;
        --cyan: #8de9e0;
        --blue: #91a5ff;
        --pink: #ff9eac;
        --panel: rgba(23, 27, 49, 0.64);
        --line: rgba(208, 214, 255, 0.15);
    }

    [data-testid="stAppViewContainer"] {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 14% 8%, rgba(112, 102, 213, 0.20), transparent 36rem),
            radial-gradient(ellipse at 92% 32%, rgba(57, 151, 166, 0.13), transparent 32rem),
            #0b0d18;
    }

    [data-testid="stAppViewContainer"]::before {
        position: fixed;
        z-index: 0;
        pointer-events: none;
        inset: 0;
        display: block;
        content: "";
        opacity: 0.12;
        background-image:
            linear-gradient(rgba(187, 198, 255, 0.12) 1px, transparent 1px),
            linear-gradient(90deg, rgba(187, 198, 255, 0.12) 1px, transparent 1px);
        background-size: 56px 56px;
        mask-image: linear-gradient(to bottom, black, transparent 70%);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 13, 24, 0.55);
        backdrop-filter: blur(16px);
    }

    .block-container {
        position: relative;
        z-index: 1;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 18, 34, 0.78);
        backdrop-filter: blur(24px) saturate(140%);
    }

    [data-testid="stSidebar"] > div:first-child {
        border-right: 1px solid rgba(208, 214, 255, 0.13);
        background: transparent;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #adb2cc;
    }

    h1, h2, h3,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f2ff;
    }

    .sidebar-brand-name {
        color: #f1f2ff;
    }

    .sidebar-brand-name span,
    .hero-title span,
    .tech-index,
    .tech-title {
        color: #8de9e0;
    }

    .sidebar-tagline {
        color: #9298b5;
    }

    .hero-shell {
        padding: clamp(1.35rem, 4vw, 2rem);
        border: 1px solid rgba(213, 219, 255, 0.16);
        border-radius: 20px;
        background:
            linear-gradient(135deg, rgba(42, 46, 78, 0.64), rgba(20, 25, 46, 0.52)),
            rgba(22, 26, 47, 0.56);
        box-shadow:
            0 20px 55px rgba(0, 0, 0, 0.22),
            inset 0 1px rgba(255, 255, 255, 0.11);
        backdrop-filter: blur(24px) saturate(150%);
        -webkit-backdrop-filter: blur(24px) saturate(150%);
    }

    .hero-title {
        color: #f5f5ff;
    }

    .hero-copy {
        color: #b6bad2;
    }

    [data-testid="stTextArea"] textarea {
        min-height: 190px;
        border: 0;
        border-radius: 11px;
        background: rgba(16, 19, 36, 0.25);
        color: #f1f2ff;
        box-shadow: none;
    }

    [data-testid="stTextAreaRootElement"] {
        overflow: hidden;
        border: 1px solid rgba(203, 211, 255, 0.19);
        border-radius: 14px;
        background: rgba(24, 28, 49, 0.54);
        box-shadow:
            inset 0 1px rgba(255, 255, 255, 0.06),
            0 12px 35px rgba(0, 0, 0, 0.12);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    [data-testid="stTextAreaRootElement"]:focus-within {
        border-color: rgba(141, 233, 224, 0.55);
        box-shadow:
            0 0 0 3px rgba(141, 233, 224, 0.10),
            0 14px 40px rgba(0, 0, 0, 0.16);
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #858ba8;
    }

    [data-testid="stButton"] > button {
        border: 1px solid rgba(164, 171, 255, 0.45);
        border-radius: 11px;
        background: linear-gradient(105deg, #7770dc, #5a91c9);
        box-shadow: 0 8px 24px rgba(94, 105, 203, 0.20);
        transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease;
    }

    [data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        border-color: rgba(206, 211, 255, 0.72);
        background: linear-gradient(105deg, #827be8, #69a1d7);
        box-shadow: 0 12px 30px rgba(94, 105, 203, 0.28);
    }

    [data-testid="stButton"] > button:focus {
        box-shadow: 0 0 0 3px rgba(141, 233, 224, 0.22);
    }

    [data-testid="stButton"] > button [data-testid="stMarkdownContainer"] p {
        color: #ffffff;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(205, 212, 255, 0.14);
        border-radius: 12px;
        background: rgba(34, 39, 65, 0.54);
        box-shadow: inset 0 1px rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    [data-testid="stMetricLabel"] {
        color: #a5a9c4;
    }

    [data-testid="stMetricValue"] {
        color: #f1f2ff;
    }

    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #8de9e0, #91a5ff);
    }

    [data-testid="stAlert"] {
        border: 1px solid rgba(205, 212, 255, 0.15);
        border-radius: 12px;
        background: rgba(31, 36, 59, 0.70);
        color: #e5e7fa;
        backdrop-filter: blur(16px);
    }

    [data-testid="stSlider"] [role="slider"] {
        border-color: #8de9e0;
    }

    [data-testid="stDivider"] {
        border-color: rgba(205, 212, 255, 0.14);
    }

    .evidence-card {
        border: 1px solid rgba(209, 216, 255, 0.19);
        border-left: 3px solid rgba(141, 233, 224, 0.76);
        border-radius: 15px;
        background:
            linear-gradient(135deg, rgba(43, 49, 79, 0.74), rgba(24, 30, 52, 0.60)),
            rgba(25, 30, 51, 0.62);
        box-shadow:
            0 16px 44px rgba(0, 0, 0, 0.18),
            inset 0 1px rgba(255, 255, 255, 0.09);
        backdrop-filter: blur(22px) saturate(140%);
        -webkit-backdrop-filter: blur(22px) saturate(140%);
    }

    .evidence-label {
        color: #afb4cf;
    }

    .evidence-prediction {
        color: #f4f4ff;
    }

    .evidence-prediction.credible {
        color: #8de9e0;
    }

    .evidence-prediction.misleading {
        color: #ff9eac;
    }

    .evidence-prediction.uncertain {
        color: #ffd58a;
    }

    .evidence-claim {
        border-top-color: rgba(210, 217, 255, 0.13);
        color: #c4c8dd;
    }

    .indicator-chip {
        border-color: rgba(205, 212, 255, 0.16);
        background: rgba(145, 165, 255, 0.09);
        color: #d2d6ed;
    }

    .tech-card {
        border-color: rgba(205, 212, 255, 0.14);
        background: rgba(34, 39, 65, 0.54);
        backdrop-filter: blur(16px);
    }

    [data-testid="stExpander"] {
        overflow: hidden;
        border-color: rgba(205, 212, 255, 0.14);
        border-radius: 12px;
        background: rgba(27, 32, 54, 0.54);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    @media (max-width: 700px) {
        .hero-shell {
            border-radius: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pipeline_path() -> Path:
    return project_root() / "outputs" / "pipeline.joblib"


def default_advanced_artifact_dir() -> Path:
    return project_root() / "outputs" / "advanced"


def default_advanced_metrics_path() -> Path:
    return default_advanced_artifact_dir() / "advanced_metrics.json"


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_pipeline(path: str):
    return load_model_pipeline(path)


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


# ============================================================
# HELPERS
# ============================================================

def is_short_input(text: str) -> bool:
    words = text.strip().split()
    return len(words) < 8 or len(text.strip()) < 50


# ============================================================
# MODEL PATH
# ============================================================

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    "--pipeline",
    default=str(default_pipeline_path()),
)

args, _ = parser.parse_known_args()

pipeline_path = Path(args.pipeline).resolve()
advanced_artifact_dir = default_advanced_artifact_dir()
advanced_components_available = advanced_component_availability(advanced_artifact_dir)
advanced_available = any(advanced_components_available.values())
advanced_metrics = load_metrics(default_advanced_metrics_path())


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">News<span>Check</span></div>
            <div class="sidebar-tagline">A guide to language patterns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_options = []
    advanced_component_options = {
        "Ensemble (recommended)": "ensemble",
        "Transformer": "transformer",
        "Attention fusion": "fusion",
        "Sentence embeddings": "embedding",
        "Baseline model": "baseline",
    }
    for label, component_name in advanced_component_options.items():
        if advanced_components_available.get(component_name):
            model_options.append(label)
    model_options.append("Baseline (raw)")
    default_model_index = (
        model_options.index("Ensemble (recommended)")
        if "Ensemble (recommended)" in model_options
        else model_options.index("Baseline model")
        if "Baseline model" in model_options
        else 0
    )
    with st.expander("Settings", expanded=False):
        st.caption("Most people can use the recommended defaults.")
        selected_model = st.selectbox(
            "Prediction model", model_options, index=default_model_index
        )
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="Adjust only if you need a stricter or more permissive decision.",
        )
        uncertainty_margin = st.slider(
            "Uncertain range",
            min_value=0.00,
            max_value=0.30,
            value=0.10,
            step=0.01,
            help="Scores close to the threshold are shown as uncertain.",
        )
        confidence_threshold = st.slider(
            "Minimum confidence",
            min_value=0.50,
            max_value=0.95,
            value=0.60,
            step=0.01,
            help="Lower-confidence predictions are shown as insufficient confidence.",
        )
        show_token_attributions = False
        if advanced_components_available.get("transformer"):
            show_token_attributions = st.checkbox(
                "Show word-level model explanation",
                value=False,
                help="Runs an extra local attribution pass for the Transformer.",
            )

    selected_component = advanced_component_options.get(selected_model, "legacy")
    use_advanced_model = selected_component != "legacy"

    if not advanced_available:
        st.caption("Using the baseline model. Advanced models have not been trained.")
    elif not advanced_components_available.get("transformer"):
        st.caption("Some advanced models are unavailable. Choose another model in Settings.")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <section class="hero-shell">
        <div class="hero-title">Check a news story<span>.</span></div>
        <p class="hero-copy">
            Paste a headline or article excerpt to see how its language compares
            with patterns learned by the model. This is not a fact-check.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CHECK
# ============================================================

if not use_advanced_model and not pipeline_path.exists():

    st.error(
        "The trained classification model is unavailable."
    )

    st.info(
        "Train the model first with "
        "`python src/train_model.py`."
    )

    st.stop()


pipeline = None
if not use_advanced_model:
    pipeline = load_pipeline(str(pipeline_path))


# ============================================================
# NEWS INPUT
# ============================================================

text = st.text_area(
    "Paste a headline or article text",
    height=210,
    placeholder=(
        "For a more useful language-pattern comparison, include a headline "
        "and a few sentences..."
    ),
)
word_count = len(text.strip().split()) if text.strip() else 0
st.caption(f"{word_count} words · Text stays on this device.")


# ============================================================
# ANALYZE
# ============================================================

analyze = st.button(
    "Analyze text",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not text.strip():

        st.warning(
            "Please enter some news text first."
        )

        st.stop()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:
        if use_advanced_model:
            from advanced_models import predict_advanced

            advanced_result = predict_advanced(
                advanced_artifact_dir,
                text,
                threshold=threshold,
                uncertainty_margin=uncertainty_margin,
                confidence_threshold=confidence_threshold,
                component=selected_component,
            )
            fake_probability = float(advanced_result["prob_fake"])
            prediction = str(advanced_result["label"])
            raw_fake_probability = float(advanced_result["raw_prob_fake"])
        else:
            if pipeline is None:
                raise RuntimeError("The baseline classifier was not loaded.")
            fake_probability = float(pipeline.predict_proba([text])[0, 1])
            raw_fake_probability = fake_probability
            if max(fake_probability, 1 - fake_probability) < confidence_threshold:
                prediction = "Insufficient Confidence"
            else:
                prediction = classify_probability(
                    fake_probability, threshold, uncertainty_margin
                )
            advanced_result = {
                "calibration_method": "Not calibrated (baseline raw probability)",
                "components": {"baseline_raw": fake_probability},
                "raw_components": {"baseline_raw": raw_fake_probability},
                "indicators": [],
                "decision_component": "baseline",
            }

    except Exception as error:

        st.error(
            "The system could not process this text."
        )

        st.exception(error)

        st.stop()


    real_probability = 1 - fake_probability


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Result")

    display_prediction = {
        "FAKE": "Likely Misleading",
        "REAL": "Likely Credible",
        "UNCERTAIN": "Uncertain",
        "Insufficient Confidence": "Insufficient Confidence",
    }.get(prediction, prediction)
    confidence = max(fake_probability, real_probability)
    escaped_text = html.escape(text)
    escaped_prediction = html.escape(display_prediction)
    calibration_method = html.escape(str(advanced_result["calibration_method"]))
    confidence_label = (
        "Raw model confidence"
        if "not calibrated" in str(advanced_result["calibration_method"]).casefold()
        or "uncalibrated" in str(advanced_result["calibration_method"]).casefold()
        else "Calibrated confidence"
    )
    prediction_class = {
        "Likely Credible": "credible",
        "Likely Misleading": "misleading",
        "Uncertain": "uncertain",
    }.get(display_prediction, "uncertain")
    st.markdown(
        f"""
        <section class="evidence-card">
            <div class="evidence-label">Model prediction · not a fact-check</div>
            <div class="evidence-prediction {prediction_class}">{escaped_prediction}</div>
            <div>{confidence_label}: <strong>{confidence:.1%}</strong></div>
            <div class="evidence-claim">{escaped_text}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The model compares writing patterns. It does not verify whether the story is true."
    )

    indicators = list(advanced_result.get("indicators", []))
    if not indicators and pipeline is not None:
        from advanced_models import baseline_feature_indicators

        indicators = baseline_feature_indicators(pipeline, text)

    component_probabilities = advanced_result.get("components", {})
    with st.expander("More about this result", expanded=False):
        st.markdown("**Model confidence**")
        st.progress(fake_probability, text=f"Misleading-pattern score: {fake_probability:.1%}")
        st.caption(
            f"Credible-pattern score: {real_probability:.1%} · "
            f"Calibration: {calibration_method}"
        )
        if component_probabilities:
            st.markdown("**Signals from the selected model**")
            component_columns = st.columns(len(component_probabilities))
            for column, (component_name, component_probability) in zip(
                component_columns, component_probabilities.items(), strict=True
            ):
                with column:
                    st.metric(
                        str(component_name).replace("_", " ").title(),
                        f"{float(component_probability):.1%}",
                    )
        if indicators:
            st.markdown("**Words that influenced the baseline model**")
            for item in indicators:
                st.markdown(
                    '<span class="indicator-chip">'
                    f"{html.escape(str(item['feature']))} · "
                    f"{html.escape(str(item['direction']))}</span>",
                    unsafe_allow_html=True,
                )
            st.caption("These are learned associations, not facts or proof.")
        else:
            st.caption(
                "Word-level indicators are not available for this selected model. "
                "Model scores are statistical signals, not verified evidence."
            )
        st.markdown("**Raw and calibrated scores**")
        score_columns = st.columns(2)
        score_columns[0].metric("Raw score", f"{raw_fake_probability:.1%}")
        score_columns[1].metric("Calibrated score", f"{fake_probability:.1%}")
        if "raw_components" in advanced_result:
            st.json(
                {
                    "Raw component scores": advanced_result["raw_components"],
                    "Calibrated component scores": component_probabilities,
                }
            )
        st.caption(f"Text length: {len(text.strip().split())} words · {len(text)} characters")

        if is_short_input(text):
            st.warning(
                "This text is quite short. A longer excerpt may give the model more "
                "language patterns to compare."
            )

        if prediction == "FAKE":
            interpretation = (
                "The model found language patterns more associated with misleading "
                "examples in its training data."
            )
        elif prediction == "REAL":
            interpretation = (
                "The model found language patterns more associated with credible "
                "examples in its training data."
            )
        elif prediction == "UNCERTAIN":
            interpretation = "The score is close to the selected decision threshold."
        else:
            interpretation = (
                "The score did not meet the minimum confidence setting. Consider "
                "adding more context."
            )
        st.write(interpretation)

        if show_token_attributions and use_advanced_model:
            from advanced_models import explain_transformer

            attribution_target = "FAKE" if fake_probability >= threshold else "REAL"
            attribution = explain_transformer(
                advanced_artifact_dir, text, target=attribution_target, steps=8
            )
            token_indicators = sorted(
                attribution["tokens"],
                key=lambda item: abs(item["attribution"]),
                reverse=True,
            )[:8]
            st.markdown("**Word-level model explanation · Integrated Gradients**")
            st.write(
                [
                    {
                        "token": item["token"],
                        "attribution": round(item["attribution"], 5),
                    }
                    for item in token_indicators
                ]
            )
            st.caption("These local model attributions are not verified evidence.")

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.info(
        "⚠️ Important: This application is a machine-learning "
        "classification system. It does not independently "
        "verify facts, sources, dates, people, or events. "
        "Always verify important claims using reliable sources."
    )


if advanced_metrics:
    with st.expander("Model performance and calibration", expanded=False):
        st.caption(
            "Evaluation scores show how the model performed on test data; "
            "they do not prove a story is true."
        )
        report_rows = []
        for model_name, model_report in advanced_metrics.get("holdout_test", {}).items():
            if model_report.get("status") != "available":
                continue
            has_comparison = False
            for probability_kind, metrics_key in (
                ("Raw", "raw_holdout_test"),
                ("Platt calibrated", "calibrated_holdout_test"),
            ):
                model_metrics = model_report.get(metrics_key)
                if not model_metrics:
                    continue
                has_comparison = True
                report_rows.append(
                    {
                        "Model": model_name,
                        "Probability": probability_kind,
                        "Accuracy": model_metrics.get("accuracy"),
                        "Precision": model_metrics.get("precision"),
                        "Recall": model_metrics.get("recall"),
                        "F1": model_metrics.get("f1"),
                        "ROC-AUC": model_metrics.get("roc_auc"),
                        "PR-AUC": model_metrics.get("pr_auc"),
                        "Brier": model_metrics.get("brier_score"),
                        "Calibration error": model_metrics.get("calibration_error"),
                    }
                )
                st.write(
                    f"{model_name} · {probability_kind} confusion matrix "
                    "(rows=true REAL/FAKE, columns=predicted REAL/FAKE)"
                )
                st.write(model_metrics.get("confusion_matrix"))
            if not has_comparison and model_report.get("holdout_test"):
                model_metrics = model_report["holdout_test"]
                report_rows.append(
                    {
                        "Model": model_name,
                        "Probability": "Previously saved score (retrain for calibration comparison)",
                        "Accuracy": model_metrics.get("accuracy"),
                        "Precision": model_metrics.get("precision"),
                        "Recall": model_metrics.get("recall"),
                        "F1": model_metrics.get("f1"),
                        "ROC-AUC": model_metrics.get("roc_auc"),
                        "PR-AUC": model_metrics.get("pr_auc"),
                        "Brier": model_metrics.get("brier_score"),
                        "Calibration error": model_metrics.get("calibration_error"),
                    }
                )
                st.write(model_metrics.get("confusion_matrix"))
        if report_rows:
            st.dataframe(report_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No trained advanced evaluation metrics are available yet.")

        calibration_curve_path = default_advanced_artifact_dir() / "calibration_curve.png"
        if calibration_curve_path.is_file():
            st.image(str(calibration_curve_path), caption="Holdout calibration curve")
        for component_name, component_status in advanced_metrics.get("components", {}).items():
            if not component_status.get("available"):
                st.caption(
                    f"{component_name.title()} not available: "
                    f"{component_status.get('reason', 'not trained')}"
                )


with st.expander("How this tool works", expanded=False):
    st.write(
        "NewsCheck compares writing patterns in your text with patterns learned "
        "from labeled examples. It does not check sources or confirm facts."
    )
    st.write(
        "The selected model runs locally. When available, Platt scaling adjusts "
        "probabilities using validation data. Predictions near the decision "
        "threshold may be marked uncertain."
    )
    st.write(
        "Confidence is not the same as truth. Verify important claims with "
        "reliable, independent sources."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NewsCheck / Educational machine-learning application / Not a factual verification service"
)
