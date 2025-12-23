from __future__ import annotations
import os
import pandas as pd
from jinja2 import Template


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>MH Scale Audit Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.35; }
    h1 { margin-bottom: 6px; }
    h2 { margin-top: 18px; margin-bottom: 8px; }
    .muted { color: #555; }
    .box { background: #fafafa; border: 1px solid #eee; padding: 12px; border-radius: 10px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
    th { background: #f5f5f5; text-align: left; }
    .ok { background: #e8f5e9; border: 1px solid #c8e6c9; padding: 10px; border-radius: 8px; }
    .warn { background: #fff3cd; border: 1px solid #ffeeba; padding: 10px; border-radius: 8px; }
    .bad { background: #fdecea; border: 1px solid #f5c2c7; padding: 10px; border-radius: 8px; }
    code { background: #f2f2f2; padding: 2px 5px; border-radius: 4px; }
  </style>
</head>
<body>

<h1>Mental Health Scale Audit Report (v4)</h1>
<p class="muted">Research tool only. Not for diagnosis or clinical decision-making.</p>

<div class="box">
  <p><b>Run tag:</b> {{ tag }}</p>
  <p><b>Group column:</b> {{ group_col }}</p>
</div>

<h2>Verdict</h2>
{{ verdict_box }}
<p class="muted"><b>Reason:</b> {{ verdict_reason }}</p>

<h2>Reliability</h2>
<p><b>Cronbach’s alpha:</b> {{ alpha_value }}</p>

<h2>Worst item–total correlations (lowest first)</h2>
{{ item_total_html }}

<h2>Missingness (highest missing first)</h2>
{{ missingness_html }}

<h2>Floor/Ceiling Effects</h2>
{{ floor_ceiling_html }}

{% if alpha_by_group_html %}
<h2>Alpha by group</h2>
{{ alpha_by_group_html }}
{% endif %}

{% if dif_html %}
<h2>DIF / Bias screening (ranked by q-value)</h2>
<p class="muted">Lower <code>q_group_joint</code> = stronger DIF signal after multiple-testing correction.</p>
{{ dif_html }}
{% endif %}

<h2>Ethics & Limits</h2>
<ul>
  <li>This tool evaluates measurement properties; it does not diagnose mental disorders.</li>
  <li>DIF results are screening-level; high-stakes use requires deeper psychometrics (IRT-DIF/invariance).</li>
</ul>

</body>
</html>
"""


def _verdict_box(verdict: str) -> str:
    if verdict == "COMPARABLE":
        return '<div class="ok"><b>✅ Comparable (preliminary)</b></div>'
    if verdict == "CAUTION":
        return '<div class="warn"><b>⚠️ Use caution</b></div>'
    return '<div class="bad"><b>❌ Not comparable (preliminary)</b></div>'


def write_report_html(
    out_path: str,
    tag: str,
    group_col: str | None,
    alpha_overall_df: pd.DataFrame | None,
    item_total_df: pd.DataFrame | None,
    missing_df: pd.DataFrame | None,
    floor_ceiling_df: pd.DataFrame | None,
    alpha_by_group_df: pd.DataFrame | None,
    dif_df: pd.DataFrame | None,
    verdict: dict
) -> None:
    tmpl = Template(HTML_TEMPLATE)

    alpha_value = "NA"
    if alpha_overall_df is not None and not alpha_overall_df.empty and "alpha" in alpha_overall_df.columns:
        a = alpha_overall_df.loc[0, "alpha"]
        alpha_value = f"{float(a):.4f}" if pd.notnull(a) else "NA"

    # keep top rows to keep report readable
    def to_html(df: pd.DataFrame | None, max_rows: int = 30) -> str:
        if df is None or df.empty:
            return "<p class='muted'>No data.</p>"
        d = df.head(max_rows).copy()
        return d.to_html(index=False, float_format=lambda x: f"{x:.6g}" if pd.notnull(x) else "")

    html = tmpl.render(
        tag=tag,
        group_col=group_col or "None",
        verdict_box=_verdict_box(verdict["verdict"]),
        verdict_reason=verdict["reason"],
        alpha_value=alpha_value,
        item_total_html=to_html(item_total_df, 25),
        missingness_html=to_html(missing_df, 25),
        floor_ceiling_html=to_html(floor_ceiling_df, 25),
        alpha_by_group_html=(to_html(alpha_by_group_df, 50) if alpha_by_group_df is not None and not alpha_by_group_df.empty else None),
        dif_html=(to_html(dif_df, 30) if dif_df is not None and not dif_df.empty else None),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
