"""Create a one-page English/Nepali situation report from summary.json only."""

import argparse
import json
import os
from pathlib import Path


ATTRIBUTIONS = (
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors.",
    "Kuro Siwo: Bountos et al., 2024.",
)
NEPALI_FONT_CANDIDATES = (
    Path(os.environ.get("NEPALI_FONT_PATH", "")),
    Path(r"C:\Windows\Fonts\Nirmala.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", nargs="?", default="analysis_output/summary.json")
    parser.add_argument("--output", default="situation_report.pdf")
    parser.add_argument("--font", default=None, help="Optional Unicode Devanagari font file")
    return parser.parse_args()


def find_nepali_font(explicit_font=None):
    candidates = ([Path(explicit_font).expanduser()] if explicit_font else []) + list(NEPALI_FONT_CANDIDATES)
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No Devanagari font found. Set NEPALI_FONT_PATH or pass --font; "
        "Nirmala.ttc is available on many Windows installations."
    )


def load_summary(path):
    summary_path = Path(path).expanduser()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = ("impacts", "access", "source")
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"Summary JSON is missing required fields: {', '.join(missing)}")
    return summary


def build_report(summary, output_path, font_path):
    try:
        from fpdf import FPDF
    except ImportError as error:
        raise RuntimeError("Install fpdf2 and uharfbuzz to generate a shaped Nepali PDF") from error

    pdf = FPDF(format="A4")
    pdf.set_margins(12, 10, 12)
    pdf.set_auto_page_break(auto=False, margin=10)
    pdf.add_font(family="Nepali", fname=str(font_path), collection_font_number=0)
    pdf.set_text_shaping(True)
    pdf.set_title("Flood Situation Report / बाढी अवस्था प्रतिवेदन")
    pdf.set_author("Flood Impact Dashboard")
    pdf.add_page()

    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_fill_color(24, 52, 67)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Nepali", size=15)
    pdf.cell(page_width, 11, "Flood Situation Report  |  बाढी अवस्था प्रतिवेदन", border=0, fill=True, align="C")
    pdf.ln(14)

    source = summary["source"]
    event_date = str(summary.get("event_date", "Not recorded / उल्लेख छैन"))
    snapshot_date = str(source.get("osm_snapshot_date", "Not recorded / उल्लेख छैन"))
    bbox = source.get("bbox_wgs84") or summary.get("requested_area_bbox_wgs84")
    bbox_text = ", ".join(f"{value:g}" for value in bbox) if bbox else "Not recorded / उल्लेख छैन"
    pdf.set_text_color(35, 45, 52)
    pdf.set_font("Nepali", size=8.5)
    pdf.multi_cell(
        page_width, 5,
        f"Event date / घटना मिति: {event_date}    |    OSM snapshot / OSM स्न्यापसट: {snapshot_date}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.multi_cell(
        page_width, 5, f"Area bbox W,S,E,N / क्षेत्र: {bbox_text}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(2)

    impacts = summary["impacts"]
    access = summary["access"]
    rows = (
        ("Buildings intersecting flood", "बाढीले छोएका भवन", str(impacts["buildings_hit"])),
        ("Affected road length (km)", "प्रभावित सडक लम्बाइ (कि.मि.)", f"{float(impacts['road_length_hit_km']):.3f}"),
        ("Bridges intersecting flood", "बाढीले छोएका पुल", str(impacts["bridges_hit"])),
        ("Settlements disconnected", "सम्पर्क विच्छेद भएका बस्ती", str(access["disconnected_count"])),
        ("Settlements retaining a route", "मार्ग कायम रहेका बस्ती", str(access["connected_count"])),
        ("No baseline route in extract", "सुरुआती मार्ग नभएका बस्ती", str(access["no_baseline_path_count"])),
    )
    pdf.set_fill_color(232, 239, 240)
    pdf.set_text_color(24, 52, 67)
    pdf.set_font("Nepali", size=8)
    pdf.cell(74, 7, "Indicator", border=1, fill=True)
    pdf.cell(82, 7, "सूचक", border=1, fill=True)
    pdf.cell(page_width - 156, 7, "Value / मान", border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(35, 45, 52)
    for english, nepali, value in rows:
        pdf.cell(74, 7, english, border=1)
        pdf.cell(82, 7, nepali, border=1)
        pdf.cell(page_width - 156, 7, value, border=1, align="C")
        pdf.ln()

    pdf.ln(3)
    pdf.set_font("Nepali", size=8.5)
    pdf.set_text_color(24, 52, 67)
    pdf.cell(page_width, 6, "Situation / अवस्था", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(35, 45, 52)
    pdf.set_font("Nepali", size=8)
    english_text = (
        f"The flood-class mask intersects {impacts['buildings_hit']} mapped buildings, "
        f"{float(impacts['road_length_hit_km']):.3f} km of mapped roads, and "
        f"{impacts['bridges_hit']} mapped bridges. The OSM snapshot is dated {snapshot_date}."
    )
    nepali_text = (
        f"बाढी वर्गको नक्साले {impacts['buildings_hit']} भवन, "
        f"{float(impacts['road_length_hit_km']):.3f} कि.मि. सडक र "
        f"{impacts['bridges_hit']} पुल छोएको छ। OSM स्न्यापसट मिति {snapshot_date} हो।"
    )
    pdf.multi_cell(page_width, 4.5, english_text, new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(page_width, 4.5, nepali_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    caveat = (
        "Access is evaluated on the extracted OSM road graph. Geometric overlap confidence is a fraction, not a probability. "
        "No debris class is inferred."
    )
    nepali_caveat = (
        "पहुँच OSM सडक सञ्जालमा मूल्याङ्कन गरिएको हो। ज्यामितीय ओभरल्याप विश्वास अंश हो, सम्भाव्यता होइन। "
        "भग्नावशेष वर्ग अनुमान गरिएको छैन।"
    )
    pdf.multi_cell(page_width, 4.5, caveat, new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(page_width, 4.5, nepali_caveat, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Nepali", size=7.5)
    pdf.set_text_color(24, 52, 67)
    pdf.cell(page_width, 5, "Attributions / श्रेय", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(55, 62, 67)
    pdf.set_font("Nepali", size=6.5)
    for attribution in ATTRIBUTIONS:
        pdf.multi_cell(page_width, 3.5, attribution, new_x="LMARGIN", new_y="NEXT")

    if pdf.page_no() != 1 or pdf.get_y() > pdf.h - 8:
        raise RuntimeError("Report content exceeded the one-page layout; shorten the source summary")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def main():
    args = parse_args()
    summary = load_summary(args.summary_json)
    font_path = find_nepali_font(args.font)
    output_path = Path(args.output).expanduser()
    build_report(summary, output_path, font_path)
    print(f"Wrote one-page bilingual report: {output_path.resolve()}")


if __name__ == "__main__":
    main()