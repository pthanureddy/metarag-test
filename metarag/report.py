from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MetaRAG-Test Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #172033; background: #f7f8fa; }
    body { margin: 0; }
    main { width: min(1160px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 56px; }
    h1 { font-size: clamp(2.4rem, 5vw, 4.6rem); line-height: 1; margin: 0 0 12px; letter-spacing: 0; }
    p, li { color: #4b5870; line-height: 1.65; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 24px 0; }
    .metric, .card { background: #fff; border: 1px solid #d9dee8; border-radius: 8px; padding: 16px; }
    .metric strong { display: block; font-size: 2rem; color: #172033; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dee8; border-radius: 8px; overflow: hidden; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #edf0f4; text-align: left; font-size: 0.92rem; vertical-align: top; }
    th { background: #eef2f7; }
    .pass { color: #0f766e; font-weight: 800; }
    .fail { color: #b42318; font-weight: 800; }
    code { background: #eef2f7; padding: 2px 5px; border-radius: 5px; }
  </style>
</head>
<body>
<main>
  <header>
    <p><strong>Research prototype</strong> · Metamorphic and groundedness testing for RAG</p>
    <h1>MetaRAG-Test</h1>
    <p>This static dashboard evaluates a RAG pipeline over a public NIST AI RMF study corpus using groundedness checks, corrupted-context validation, paraphrase metamorphic tests, and consistency scoring.</p>
  </header>

  <section class="metrics">
    <div class="metric"><span>Groundedness accuracy</span><strong>{{ "%.0f"|format(summary.groundedness_accuracy * 100) }}%</strong></div>
    <div class="metric"><span>Hallucination catch</span><strong>{{ "%.0f"|format(summary.hallucination_catch_rate * 100) }}%</strong></div>
    <div class="metric"><span>Metamorphic violations</span><strong>{{ "%.0f"|format(summary.metamorphic_violation_rate * 100) }}%</strong></div>
    <div class="metric"><span>RAGAS agreement</span><strong>{{ "%.0f"|format(summary.ragas_agreement * 100) }}%</strong></div>
  </section>

  <div id="chart" class="card" style="height: 360px;"></div>

  <h2>Method</h2>
  <div class="card">
    <p>The corpus is a public NIST AI RMF study corpus. The test runner retrieves context, answers seed and paraphrased questions, judges groundedness with a local NLI oracle or fallback, validates against deliberately corrupted context, and compares the decision with a RAGAS-style faithfulness proxy.</p>
    <p>Chroma persisted: <code>{{ summary.chroma_persisted }}</code>. OpenAI judge included: <code>{{ summary.openai_judge_included }}</code>.</p>
  </div>

  <h2>Grounded Cases</h2>
  <table>
    <thead><tr><th>ID</th><th>Question</th><th>Grounded</th><th>Score</th><th>Oracle</th></tr></thead>
    <tbody>
    {% for row in cases[:12] %}
      <tr>
        <td><code>{{ row.id }}</code></td>
        <td>{{ row.question }}</td>
        <td class="{{ 'pass' if row.actual_grounded else 'fail' }}">{{ row.actual_grounded }}</td>
        <td>{{ "%.2f"|format(row.score) }}</td>
        <td>{{ row.oracle }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Limitations</h2>
  <ul>
    <li>CI uses deterministic local fallbacks so no API keys are required.</li>
    <li>The NLI and RAGAS integrations become stronger when optional heavyweight dependencies are installed locally.</li>
    <li>The dataset is intentionally public and unrelated to employer data.</li>
  </ul>
</main>
<script>
  const summary = {{ summary_json }};
  Plotly.newPlot('chart', [{
    type: 'bar',
    x: ['Groundedness', 'Hallucination catch', 'Metamorphic violations', 'RAGAS agreement'],
    y: [
      summary.groundedness_accuracy,
      summary.hallucination_catch_rate,
      summary.metamorphic_violation_rate,
      summary.ragas_agreement
    ],
    marker: { color: ['#0f766e', '#1d4ed8', '#b45309', '#6d28d9'] }
  }], { yaxis: { range: [0, 1], tickformat: '.0%' }, margin: { t: 16 } }, { displayModeBar: false });
</script>
</body>
</html>
"""


def generate_report(evaluation_path: Path, docs_dir: Path) -> None:
    data = json.loads(evaluation_path.read_text(encoding="utf-8"))
    docs_dir.mkdir(parents=True, exist_ok=True)
    html = Template(HTML).render(
        summary=data["summary"],
        summary_json=json.dumps(data["summary"]),
        cases=data["cases"],
    )
    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    (docs_dir / "report.md").write_text(markdown_report(data), encoding="utf-8")


def markdown_report(data: dict) -> str:
    summary = data["summary"]
    rows = ["| Metric | Value |", "|---|---:|"]
    for key in [
        "groundedness_accuracy",
        "hallucination_catch_rate",
        "metamorphic_violation_rate",
        "mean_flakiness_variance",
        "ragas_agreement",
    ]:
        rows.append(f"| {key.replace('_', ' ')} | {summary[key]:.3f} |")
    return "\n".join(
        [
            "# MetaRAG-Test Report",
            "",
            "## Summary",
            "",
            *rows,
            "",
            "## Dataset",
            "",
            "The corpus is a compact public NIST AI RMF study corpus, chosen to avoid employer IP and to focus the project on testing RAG behavior.",
            "",
            "## Validation",
            "",
            "The runner validates groundedness on clean retrieved context, then deliberately corrupts retrieved context and checks whether the oracle catches unsupported answers.",
        ]
    )
