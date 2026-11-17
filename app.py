"""
Gemini Text Classifier & Sentiment Analyzer — refreshed UI
Python 3.15 desktop app using CustomTkinter.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox

import customtkinter as ctk
from dotenv import load_dotenv
from google import genai
from google.genai import types

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class GeminiClassifierApp:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("Gemini Studio | Text Intelligence")
        self.root.geometry("1120x820")
        self.root.minsize(900, 700)
        self.root.configure(fg_color="#F4F7FC")

        self.events: queue.Queue = queue.Queue()
        self.history: list[dict] = []
        self.last_result: dict | None = None

        self._build_ui()
        self.root.after(100, self._process_events)

    def _build_ui(self) -> None:
        # Main scrollable page keeps the layout usable on smaller screens.
        page = ctk.CTkScrollableFrame(
            self.root, fg_color="#F4F7FC", corner_radius=0,
            scrollbar_button_color="#CBD5E1", scrollbar_button_hover_color="#94A3B8"
        )
        page.pack(fill="both", expand=True)
        page.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=34, pady=(28, 18))
        header.grid_columnconfigure(0, weight=1)

        brand_line = ctk.CTkFrame(header, fg_color="transparent")
        brand_line.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            brand_line, text="✦", text_color="#2563EB",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(
            brand_line, text="GEMINI STUDIO", text_color="#2563EB",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left")

        ctk.CTkLabel(
            header, text="Text Intelligence", text_color="#14213D",
            font=ctk.CTkFont(size=31, weight="bold")
        ).grid(row=1, column=0, sticky="w", pady=(9, 3))
        ctk.CTkLabel(
            header,
            text="Classify topics, understand sentiment, and extract meaning from your text.",
            text_color="#64748B", font=ctk.CTkFont(size=13)
        ).grid(row=2, column=0, sticky="w")

        # Settings card
        settings = self._card(page, row=1, padx=34, pady=(0, 14))
        settings.grid_columnconfigure((0, 1, 2), weight=1)
        self._section_heading(settings, "Analysis settings", "Choose your Gemini model and classification context.", 0, 3)

        ctk.CTkLabel(settings, text="GEMINI MODEL", text_color="#64748B",
                     font=ctk.CTkFont(size=10, weight="bold")).grid(row=1, column=0, sticky="w", padx=20, pady=(18, 6))
        self.model_var = ctk.StringVar(value=DEFAULT_MODEL)
        self.model_entry = ctk.CTkEntry(
            settings, textvariable=self.model_var, height=42, corner_radius=9,
            border_color="#D9E2F0", fg_color="#FBFCFE", text_color="#172033"
        )
        self.model_entry.grid(row=2, column=0, sticky="ew", padx=(20, 10), pady=(0, 18))

        ctk.CTkLabel(settings, text="CLASSIFICATION MODE", text_color="#64748B",
                     font=ctk.CTkFont(size=10, weight="bold")).grid(row=1, column=1, sticky="w", padx=10, pady=(18, 6))
        self.mode_var = ctk.StringVar(value="General")
        self.mode_menu = ctk.CTkOptionMenu(
            settings, variable=self.mode_var, values=[
                "General", "Reviews", "Customer support", "News & articles", "Social media", "Custom categories"
            ], height=42, corner_radius=9, fg_color="#2563EB", button_color="#1D4ED8",
            button_hover_color="#1E40AF", dropdown_fg_color="#FFFFFF",
            dropdown_text_color="#172033"
        )
        self.mode_menu.grid(row=2, column=1, sticky="ew", padx=10, pady=(0, 18))

        ctk.CTkLabel(settings, text="CUSTOM CATEGORIES (OPTIONAL)", text_color="#64748B",
                     font=ctk.CTkFont(size=10, weight="bold")).grid(row=1, column=2, sticky="w", padx=(10, 20), pady=(18, 6))
        self.categories_var = ctk.StringVar()
        self.categories_entry = ctk.CTkEntry(
            settings, textvariable=self.categories_var, height=42, corner_radius=9,
            placeholder_text="e.g. Billing, Delivery, Product quality",
            border_color="#D9E2F0", fg_color="#FBFCFE", text_color="#172033"
        )
        self.categories_entry.grid(row=2, column=2, sticky="ew", padx=(10, 20), pady=(0, 18))

        # Input card
        input_card = self._card(page, row=2, padx=34, pady=(0, 14))
        input_card.grid_columnconfigure(0, weight=1)
        self._section_heading(input_card, "Your text", "Paste or type the content you want Gemini to analyze.", 0, 1)
        self.input_text = ctk.CTkTextbox(
            input_card, height=185, corner_radius=10, border_width=1,
            border_color="#D9E2F0", fg_color="#FBFCFE",
            text_color="#172033", font=ctk.CTkFont(size=13),
            scrollbar_button_color="#CBD5E1"
        )
        self.input_text.grid(row=1, column=0, sticky="ew", padx=20, pady=(17, 10))
        self.input_text.insert(
            "1.0",
            "The delivery was fast and the product quality was excellent. I am very happy!"
        )

        input_footer = ctk.CTkFrame(input_card, fg_color="transparent")
        input_footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 18))
        input_footer.grid_columnconfigure(0, weight=1)
        self.char_count = ctk.CTkLabel(input_footer, text="0 characters", text_color="#94A3B8",
                                       font=ctk.CTkFont(size=11))
        self.char_count.grid(row=0, column=0, sticky="w")
        self.input_text.bind("<KeyRelease>", self._update_char_count)
        self.analyze_btn = ctk.CTkButton(
            input_footer, text="✦  Analyze text", height=42, width=160,
            corner_radius=9, fg_color="#2563EB", hover_color="#1D4ED8",
            font=ctk.CTkFont(size=13, weight="bold"), command=self.analyze
        )
        self.analyze_btn.grid(row=0, column=1, padx=(0, 9))
        ctk.CTkButton(
            input_footer, text="Clear", height=42, width=85, corner_radius=9,
            fg_color="#EEF2F8", hover_color="#E2E8F0", text_color="#334155",
            command=self.clear
        ).grid(row=0, column=2)

        # Results card
        result_card = self._card(page, row=3, padx=34, pady=(0, 30))
        result_card.grid_columnconfigure(0, weight=1)
        result_header = ctk.CTkFrame(result_card, fg_color="transparent")
        result_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 4))
        result_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(result_header, text="Analysis results", text_color="#14213D",
                     font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, sticky="w")
        self.status_label = ctk.CTkLabel(
            result_header, text="●  Ready", text_color="#64748B",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.status_label.grid(row=0, column=1, sticky="e")

        # Metric tiles
        metrics = ctk.CTkFrame(result_card, fg_color="transparent")
        metrics.grid(row=1, column=0, sticky="ew", padx=20, pady=(12, 12))
        for col in range(4):
            metrics.grid_columnconfigure(col, weight=1)
        self.metric_values = {}
        for col, (key, label, value) in enumerate([
            ("category", "CATEGORY", "—"),
            ("sentiment", "SENTIMENT", "—"),
            ("score", "SENTIMENT SCORE", "—"),
            ("confidence", "CONFIDENCE", "—"),
        ]):
            tile = ctk.CTkFrame(metrics, fg_color="#F6F8FC", corner_radius=10, border_width=1, border_color="#E8EDF5")
            tile.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 6, 6 if col < 3 else 0))
            ctk.CTkLabel(tile, text=label, text_color="#64748B",
                         font=ctk.CTkFont(size=9, weight="bold")).pack(anchor="w", padx=12, pady=(11, 4))
            value_label = ctk.CTkLabel(tile, text=value, text_color="#14213D",
                                       font=ctk.CTkFont(size=14, weight="bold"))
            value_label.pack(anchor="w", padx=12, pady=(0, 12))
            self.metric_values[key] = value_label

        # Details panel
        self.details_box = ctk.CTkTextbox(
            result_card, height=190, corner_radius=10, border_width=1,
            border_color="#E2E8F0", fg_color="#FBFCFE",
            text_color="#334155", font=ctk.CTkFont(size=12),
            scrollbar_button_color="#CBD5E1"
        )
        self.details_box.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 12))
        self.details_box.insert("1.0", "Your analysis will appear here.\n\n"
                                "Results include emotion, keywords, a short summary, and the reasoning behind the classification.")
        self.details_box.configure(state="disabled")

        bottom = ctk.CTkFrame(result_card, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 18))
        bottom.grid_columnconfigure(0, weight=1)
        self.model_used = ctk.CTkLabel(bottom, text="Powered by Google Gemini API",
                                       text_color="#94A3B8", font=ctk.CTkFont(size=10))
        self.model_used.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            bottom, text="Copy result", height=34, width=110, corner_radius=8,
            fg_color="#EEF2F8", hover_color="#E2E8F0", text_color="#334155",
            command=self.copy_result
        ).grid(row=0, column=1, padx=(0, 8))
        ctk.CTkButton(
            bottom, text="Export CSV", height=34, width=110, corner_radius=8,
            fg_color="#EEF2F8", hover_color="#E2E8F0", text_color="#334155",
            command=self.export_csv
        ).grid(row=0, column=2)

    def _card(self, parent, row: int, padx, pady):
        card = ctk.CTkFrame(parent, fg_color="#FFFFFF", corner_radius=15,
                            border_width=1, border_color="#E7ECF4")
        card.grid(row=row, column=0, sticky="ew", padx=padx, pady=pady)
        return card

    def _section_heading(self, parent, title: str, subtitle: str, row: int, span: int) -> None:
        ctk.CTkLabel(parent, text=title, text_color="#14213D",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=row, column=0, columnspan=span, sticky="w", padx=20, pady=(18, 2)
        )
        ctk.CTkLabel(parent, text=subtitle, text_color="#94A3B8",
                     font=ctk.CTkFont(size=11)).grid(
            row=row + 1, column=0, columnspan=span, sticky="w", padx=20
        )

    def _update_char_count(self, _event=None) -> None:
        count = len(self.input_text.get("1.0", "end-1c"))
        self.char_count.configure(text=f"{count:,} characters")

    def _set_status(self, text: str, color: str = "#64748B") -> None:
        self.status_label.configure(text=f"●  {text}", text_color=color)

    def analyze(self) -> None:
        text = self.input_text.get("1.0", "end-1c").strip()
        model = self.model_var.get().strip()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not text:
            messagebox.showwarning("Text required", "Please enter some text to analyze.")
            return
        if not model:
            messagebox.showwarning("Model required", "Please enter a Gemini model name.")
            return
        if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
            messagebox.showerror("API key missing", "Add your Gemini API key to the .env file and restart the app.")
            return

        self.analyze_btn.configure(state="disabled", text="Analyzing…")
        self._set_status("Analyzing", "#2563EB")
        self._show_details("Gemini is analyzing your text. This may take a few seconds…")
        threading.Thread(
            target=self._run_analysis,
            args=(api_key, model, text, self.mode_var.get(), self.categories_var.get().strip()),
            daemon=True,
        ).start()

    def _run_analysis(self, api_key: str, model: str, text: str, mode: str, custom_categories: str) -> None:
        try:
            client = genai.Client(api_key=api_key)
            category_instruction = (
                f"Classify into one of these categories when suitable: {custom_categories}."
                if custom_categories else f"Use this classification context: {mode}."
            )
            prompt = f"""
You are a careful text classification and sentiment analysis system.
{category_instruction}
Return ONLY valid JSON with this exact structure:
{{
  "category": "main topic/category",
  "sentiment": "Positive, Negative, or Neutral",
  "sentiment_score": 0.0,
  "confidence": 0.0,
  "emotion": "main emotion",
  "keywords": ["keyword 1", "keyword 2"],
  "summary": "one or two sentence summary",
  "reason": "brief explanation"
}}
Rules:
- sentiment_score is from -1.0 (very negative) to 1.0 (very positive).
- confidence is from 0.0 to 1.0.
- Do not invent facts.
- Write summary and reason in the language of the input.
TEXT:
{text}
"""
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    max_output_tokens=1200,
                ),
            )
            data = json.loads((response.text or "").strip())
            self.events.put(("success", {"data": data, "source": text, "model": model}))
        except json.JSONDecodeError:
            self.events.put(("error", "Gemini returned invalid JSON. Please try again."))
        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _show_details(self, text: str) -> None:
        self.details_box.configure(state="normal")
        self.details_box.delete("1.0", "end")
        self.details_box.insert("1.0", text)
        self.details_box.configure(state="disabled")

    def _process_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self.analyze_btn.configure(state="normal", text="✦  Analyze text")
                if kind == "success":
                    data = payload["data"]
                    self.last_result = data
                    self.history.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "source_text": payload["source"],
                        "model": payload["model"],
                        **data,
                    })
                    self._render_result(data, payload["model"])
                    self._set_status("Analysis complete", "#059669")
                else:
                    self._show_details("Analysis could not be completed.\n\n" + str(payload))
                    self._set_status("Analysis failed", "#DC2626")
                    messagebox.showerror("Analysis error", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._process_events)

    def _render_result(self, data: dict, model: str) -> None:
        score = data.get("sentiment_score", "—")
        confidence = data.get("confidence", "—")
        try:
            score_text = f"{float(score):+.2f}"
        except (TypeError, ValueError):
            score_text = str(score)
        try:
            confidence_text = f"{float(confidence) * 100:.0f}%"
        except (TypeError, ValueError):
            confidence_text = str(confidence)

        self.metric_values["category"].configure(text=str(data.get("category", "—"))[:22])
        self.metric_values["sentiment"].configure(text=str(data.get("sentiment", "—")))
        self.metric_values["score"].configure(text=score_text)
        self.metric_values["confidence"].configure(text=confidence_text)

        keywords = data.get("keywords", [])
        if isinstance(keywords, list):
            keywords = ", ".join(map(str, keywords))
        details = (
            f"EMOTION\n{data.get('emotion', '—')}\n\n"
            f"KEYWORDS\n{keywords or '—'}\n\n"
            f"SUMMARY\n{data.get('summary', '—')}\n\n"
            f"WHY THIS RESULT?\n{data.get('reason', '—')}"
        )
        self._show_details(details)
        self.model_used.configure(text=f"Model: {model}  •  Powered by Google Gemini API")

    def clear(self) -> None:
        self.input_text.delete("1.0", "end")
        self._update_char_count()
        self._show_details("Your analysis will appear here.")
        for key in self.metric_values:
            self.metric_values[key].configure(text="—")
        self.last_result = None
        self._set_status("Ready")

    def copy_result(self) -> None:
        if not self.last_result:
            messagebox.showinfo("Nothing to copy", "Run an analysis first.")
            return
        data = self.last_result
        text = json.dumps(data, ensure_ascii=False, indent=2)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Result copied", "#059669")

    def export_csv(self) -> None:
        if not self.history:
            messagebox.showinfo("No history", "Analyze at least one text before exporting.")
            return
        path = filedialog.asksaveasfilename(
            title="Export analysis history", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], initialfile="gemini_analysis_history.csv"
        )
        if not path:
            return
        fields = ["timestamp", "model", "source_text", "category", "sentiment",
                  "sentiment_score", "confidence", "emotion", "keywords", "summary", "reason"]
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in self.history:
                    item = row.copy()
                    if isinstance(item.get("keywords"), list):
                        item["keywords"] = ", ".join(map(str, item["keywords"]))
                    writer.writerow(item)
            messagebox.showinfo("Export complete", f"Exported {len(self.history)} analysis result(s).")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))


def main() -> None:
    root = ctk.CTk()
    GeminiClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
