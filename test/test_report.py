"""MD and self-contained HTML report generation for the HII boundary test framework.

Generates two output files:
- test_plan_result.md  (markdown with embedded image references)
- test_plan_result.html (self-contained HTML, images base64-embedded, KaTeX for math)
"""

import base64
import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from astropy.table import Table


# ---------------------------------------------------------------------------
# 1. Base64 image encoding
# ---------------------------------------------------------------------------

def base64_encode_image(image_path: Path) -> str:
    """Read a PNG/JPG image and return a ``data:image/...;base64,...`` URI string.

    Parameters
    ----------
    image_path : Path
        Path to the image file.

    Returns
    -------
    str
        Base64 data URI.
    """
    ext = image_path.suffix.lower().lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'
    elif ext == 'svg':
        ext = 'svg+xml'
    elif ext not in ('png', 'jpeg', 'gif', 'webp', 'svg+xml'):
        ext = 'png'  # fallback

    with open(image_path, 'rb') as fh:
        data = base64.b64encode(fh.read()).decode('ascii')

    return f'data:image/{ext};base64,{data}'


# ---------------------------------------------------------------------------
# 2. Markdown report builder
# ---------------------------------------------------------------------------

def _fmt(val, digits=2) -> str:
    """Format a float for display, handling None gracefully."""
    if val is None:
        return 'N/A'
    if isinstance(val, float):
        return f'{val:.{digits}f}'
    return str(val)


def _badge(passed) -> str:
    """Return a pass/fail/info badge.

    Parameters
    ----------
    passed : bool or None
        True = PASS, False = FAIL, None = INFO (informational test).
    """
    if passed is None:
        return 'ℹ️ INFO'
    if passed:
        return '✅ PASS'
    return '❌ FAIL'


def _expected_calculation_section(expected, sigma_r=None, model_summary_path=None) -> str:
    """Build a markdown section showing the expected boundary calculation steps.

    Parameters
    ----------
    expected : ExpectedBoundary or None
    sigma_r : float or None
    """
    if expected is None:
        return ''
    r = expected.r_grid
    f_raw = expected.f_raw
    f_smooth = expected.f_smooth
    G = expected.gradient
    score = expected.score
    exp_r = expected.radius

    # Find the peak index
    peak_idx = np.argmax(score)

    lines = []
    lines.append('### Expected Boundary Calculation')
    lines.append('')

    lines.append(f'**Algorithm:** sample on polar grid → Gaussian smooth '
                 f'($\sigma_r$ = {sigma_r:.4f} px, FWHM = {sigma_r * 2.355:.1f} px) '
                 f'→ Sobel gradient → score = $G^2 / 2$ → argmax in ROI')
    lines.append('')
    lines.append(f'**Polar grid:** $dr = {r[1] - r[0]:.4f}$ px ({len(r)} samples in analysis window)')
    lines.append('')

    # Brief summary of key values
    lines.append(f'|  | $r$ (px) | $f_{{raw}}$ | $f_{{smooth}}$ | $G(r)$ | Score |')
    lines.append(f'|---------|---------|----------|-------------|--------|-------|')
    for offset, label in [(-1, 'Below peak'), (0, '**Peak**'), (1, 'Above peak')]:
        i = peak_idx + offset
        if 0 <= i < len(r):
            lines.append(
                f'| {label} | {r[i]:.3f} | {f_raw[i]:.3f} | {f_smooth[i]:.4f} | '
                f'{G[i]:+.4f} | {score[i]:.4f} |'
            )
    lines.append('')
    lines.append(f'**Expected radius:** $r_{{\\text{{exp}}}} = {exp_r:.3f}$ px (score = {score[peak_idx]:.4f})')
    lines.append('')

    # Embed the model summary plot
    if model_summary_path:
        lines.append(f'![Model summary: image + radial profile]({model_summary_path})')
        lines.append('')

    return '\n'.join(lines)


def _error_metrics_table(metrics: dict) -> str:
    """Build a markdown error-metrics sub-table."""
    rows = [
        f'| Mean Detected r | {_fmt(metrics.get("r_detected_mean"))} px |',
        f'| Min / Max Detected r | {_fmt(metrics.get("r_detected_min"))} / {_fmt(metrics.get("r_detected_max"))} px |',
        f'| MRE | {_fmt(metrics.get("mre"))} px |',
        f'| RMS Error | {_fmt(metrics.get("rms"))} px |',
        f'| Max Error | {_fmt(metrics.get("max_error"))} px |',
        f'| Angular Std | {_fmt(metrics.get("angular_std"))} px |',
    ]
    if metrics.get("mean_uncertainty") is not None:
        rows.append(f'| Mean Uncertainty (1σ) | {_fmt(metrics.get("mean_uncertainty"))} px |')
        rows.append(f'| Max Uncertainty (1σ) | {_fmt(metrics.get("max_uncertainty"))} px |')
        rows.append(f'| Bootstrap Scenario | {metrics.get("scenario", "N/A")} |')
    return '\n'.join(rows)


def _params_table(params: dict) -> str:
    """Build a markdown parameter table."""
    rows = ['| Parameter | Value |', '|-----------|-------|']
    for k, v in params.items():
        rows.append(f'| {k} | {_fmt(v, digits=4)} |')
    return '\n'.join(rows)


def _plots_section(plots: dict) -> str:
    """Build a markdown image-reference section for diagnostic plots."""
    if not plots:
        return ''
    lines = ['### Diagnostic Plots', '']
    for name, path in plots.items():
        lines.append(f'![{name}]({path})')
        lines.append('')
    return '\n'.join(lines)


def _sub_results_section(sub_results: dict, label: str) -> str:
    """Render a sub-dict of parameter-sweep results (e.g. k_results, sigma_results, noise_results)."""
    if not sub_results:
        return ''

    lines = [f'### {label}', '']
    for key, metrics in sub_results.items():
        passed = metrics.get('passed', abs(metrics.get('mre', 999)) < 1.0)
        lines.append(f'**{label} = {_fmt(key, 4)}**  {_badge(passed)}')
        lines.append('')
        lines.append(_error_metrics_table(metrics))
        lines.append('')
    return '\n'.join(lines)


def build_md_report(results: dict, config: dict) -> str:
    """Build a complete markdown test report.

    Parameters
    ----------
    results : dict
        Test results keyed by test identifier.  Each value is a dict with::

            name : str
            model_description : str  (e.g. 'crater_sharp', 'crater_sigmoid')
            params : dict          (model parameters, e.g. {'R0': 60.0})
            expected_radius : float
            error_metrics : dict   (from ``compare_boundaries``)
            passed : bool
            plots : dict           (name → path to PNG)

        Optional keys *at the top level* of the results dict:

        - ``noise_results`` : dict of noise-level → per-level error_metrics
        - ``k_results`` / ``sigma_results`` : parameter-sweep sub-dicts for
          sigmoid / ring models.

    config : dict
        Global configuration with keys::

            timestamp, git_commit, python_version, shape,
            smoothing_fwhm, cost_map_smoothing_sigma, gradient_smoothing_sigma

    Returns
    -------
    str
        Complete markdown report.
    """
    lines = []

    # --- Header ---
    lines.append('# HII Region Boundary Detection -- Test Report')
    lines.append('')
    lines.append(f'**Date:** {config.get("timestamp", "N/A")}  ')
    lines.append(f'**Git commit:** `{config.get("git_commit", "N/A")}`  ')
    lines.append(f'**Python:** {config.get("python_version", sys.version.split()[0])}  ')
    lines.append(f'**Image shape:** {config.get("shape", "N/A")}  ')
    lines.append('')
    lines.append('### Algorithm Parameters')
    lines.append('')
    lines.append(f'| Parameter | Value |')
    lines.append(f'|-----------|-------|')

    algo_params = config.get("algo_params", {})
    # Display names for known parameters (English label → display name)
    param_labels = {
        'method': 'Method',
        'smoothing_fwhm': 'Radial smoothing FWHM',
        'cost_map_smoothing_sigma': 'Cost-map smoothing σ',
        'gradient_smoothing_sigma': 'Gradient smoothing σ',
        'boundary_smoothing_sigma': 'Boundary smoothing σ',
        'rmin_start_ratio': 'rmin start ratio',
        'rmin_min_pixels': 'rmin min pixels',
        'rmax_limit_ratio': 'rmax limit ratio',
        'angular_snr_weighting': 'Angular SNR weighting',
        'angular_snr_sigma': 'Angular SNR σ',
        'coherence_penalty_weight': 'Coherence penalty weight',
        'coherence_sigma': 'Coherence σ',
        'stable_window': 'Stable window',
        'stable_threshold': 'Stable threshold',
        'n_steps': 'Scan steps (n_steps)',
        'n_bootstrap': 'Bootstrap iterations',
        'detect_rising_edge': 'Detect rising edge',
    }
    # Units for parameters that have them
    param_units = {
        'smoothing_fwhm': 'px',
        'cost_map_smoothing_sigma': 'px',
        'gradient_smoothing_sigma': 'px',
        'boundary_smoothing_sigma': 'px',
        'rmin_min_pixels': 'px',
        'angular_snr_sigma': 'px',
        'coherence_sigma': 'px',
        'stable_threshold': 'px',
    }

    for key, val in algo_params.items():
        if key.startswith('#') or val is None:
            continue
        label = param_labels.get(key, key)
        unit = param_units.get(key, '')
        if isinstance(val, bool):
            display_val = 'true' if val else 'false'
        elif isinstance(val, str):
            display_val = f'"{val}"'
        else:
            display_val = _fmt(val)
        if unit:
            display_val += f' {unit}'
        lines.append(f'| {label} | {display_val} |')

    lines.append('')

    # --- Summary table ---
    lines.append('## Summary')
    lines.append('')
    lines.append('| Test | Model | Exp r | DP MRE | DP RMS | argmax_f MRE | argmax_r MRE | Mean Unc | Result |')
    lines.append('|------|-------|-------|--------|--------|-------------|-------------|----------|--------|')

    for tid, tdata in results.items():
        if not isinstance(tdata, dict):
            continue
        name = tdata.get('name', tid)
        model = tdata.get('model_description', '')
        e_r = _fmt(tdata.get('expected_radius'))
        em = tdata.get('error_metrics', {})
        mre = _fmt(em.get('mre'))
        rms = _fmt(em.get('rms'))
        unc = _fmt(em.get('mean_uncertainty'))

        # Argmax MRE columns
        argmax_m = tdata.get('argmax_metrics', {})
        am_final = _fmt(argmax_m.get('final', {}).get('mre')) if argmax_m else 'N/A'
        am_raw = _fmt(argmax_m.get('raw', {}).get('mre')) if argmax_m else 'N/A'

        badge = _badge(tdata.get('passed'))
        lines.append(
            f'| {name} | {model} | {e_r} | {mre} | {rms} | {am_final} | {am_raw} | {unc} | {badge} |'
        )

    lines.append('')

    # --- Per-test sections ---
    for tid, tdata in results.items():
        if not isinstance(tdata, dict):
            continue
        name = tdata.get('name', tid)
        model = tdata.get('model_description', '')
        params = tdata.get('params', {})
        exp_r = tdata.get('expected_radius', None)
        em = tdata.get('error_metrics', {})
        passed = tdata.get('passed')
        plots = tdata.get('plots', {})

        lines.append(f'## {name}')
        lines.append('')
        lines.append(f'**Model:** `{model}`  ')
        if exp_r is not None:
            lines.append(f'**Expected radius:** $r_0 = {_fmt(exp_r)}$ px  ')
        lines.append(f'**Result:** {_badge(passed)}')
        lines.append('')

        # Model parameters
        if params:
            lines.append('### Model Parameters')
            lines.append('')
            lines.append(_params_table(params))
            lines.append('')

        # Expected boundary calculation (may be in top-level or _internal)
        exp_obj = tdata.get('expected', None)
        if exp_obj is None:
            exp_obj = tdata.get('_internal', {}).get('expected', None)
        sigma_r = tdata.get('sigma_r', None)
        if sigma_r is None:
            sigma_r = tdata.get('_internal', {}).get('sigma_r', None)
        model_plot = plots.pop('model_summary', None)
        plots.pop('radial_profile', None)
        plots.pop('boundary_overlay', None)
        plots.pop('cost_map_path', None)
        plots.pop('shift_curves', None)
        lines.append(_expected_calculation_section(exp_obj, sigma_r, model_plot))

        # Error metrics
        if em:
            lines.append('### Error Metrics')
            lines.append('')
            lines.append(_error_metrics_table(em))
            lines.append('')

        # Argmax baseline comparison (if available)
        argmax_m = tdata.get('argmax_metrics', {})
        if argmax_m:
            lines.append('')
            lines.append('**Argmax baseline comparison:**')
            lines.append('')
            lines.append('| Method | MRE | RMS | Max Err | Angular Std |')
            lines.append('|--------|-----|-----|---------|-------------|')
            # DP row
            lines.append(
                f'| DP (Viterbi) | {_fmt(em.get("mre"))} | {_fmt(em.get("rms"))} '
                f'| {_fmt(em.get("max_error"))} | {_fmt(em.get("angular_std"))} |'
            )
            # argmax rows
            for key, label in [("final", "argmax (final grad)"), ("raw", "argmax (raw grad)")]:
                am = argmax_m.get(key, {})
                if am:
                    lines.append(
                        f'| {label} | {_fmt(am.get("mre"))} | {_fmt(am.get("rms"))} '
                        f'| {_fmt(am.get("max_error"))} | {_fmt(am.get("angular_std"))} |'
                    )
            lines.append('')

        # Diagnostic plots
        lines.append(_plots_section(plots))

        # --- Sub-results (noise, k, sigma sweeps) ---
        noise = tdata.get('noise_results', {})
        if noise:
            lines.append(_sub_results_section(noise, 'Noise Sweep'))

        k_res = tdata.get('k_results', {})
        if k_res:
            lines.append(_sub_results_section(k_res, 'Sigmoid $k$ Sweep'))

        sigma_res = tdata.get('sigma_results', {})
        if sigma_res:
            lines.append(_sub_results_section(sigma_res, 'Ring $\\sigma$ Sweep'))

        lines.append('---')
        lines.append('')

    # ---- Argmax Baseline Synthesis ----
    synthesis_rows = []
    for tid, tdata in results.items():
        if not isinstance(tdata, dict):
            continue
        argmax_m = tdata.get('argmax_metrics', {})
        if not argmax_m:
            continue
        em = tdata.get('error_metrics', {})
        dp_mre = abs(em.get('mre', 0))

        for key in ["final", "raw"]:
            am = argmax_m.get(key, {})
            if not am:
                continue
            am_mre = abs(am.get('mre', 0))
            delta = am_mre - dp_mre
            win_factor = am_mre / max(abs(dp_mre), 1e-6)
            param_str = _fmt(tdata.get('params', {}).get('noise_level',
                         tdata.get('params', {}).get('offset_frac',
                         tdata.get('params', {}).get('rmax_val', ''))))
            synthesis_rows.append({
                'name': tdata.get('name', tid),
                'model': tdata.get('model_description', ''),
                'param': param_str,
                'key': key,
                'delta': delta,
                'win_factor': win_factor,
            })

    if synthesis_rows:
        lines.append('## Argmax Baseline — DP Smoothing Contribution')
        lines.append('')
        lines.append('ΔMRE = |MRE_argmax| − |MRE_DP|. Positive = DP better.')
        lines.append('')
        lines.append('| Model | Test | Param | Grad | ΔMRE | DP Win Factor |')
        lines.append('|-------|------|-------|------|------|---------------|')
        for r in synthesis_rows:
            lines.append(
                f'| {r["model"]} | {r["name"]} | {r["param"]} '
                f'| {r["key"]} | {_fmt(r["delta"])} px | {r["win_factor"]:.1f}× |'
            )
        lines.append('')

        final_deltas = [r['delta'] for r in synthesis_rows if r['key'] == 'final']
        raw_deltas = [r['delta'] for r in synthesis_rows if r['key'] == 'raw']
        final_factors = [r['win_factor'] for r in synthesis_rows if r['key'] == 'final']

        if final_deltas:
            lines.append(f'**Overall (final grad):** avg ΔMRE = {np.mean(final_deltas):.2f} px, '
                         f'median DP win factor = {np.median(final_factors):.1f}×')
        if raw_deltas:
            lines.append(f'**Overall (raw grad):** avg ΔMRE = {np.mean(raw_deltas):.2f} px')
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. HTML report builder (self-contained, KaTeX, base64 images)
# ---------------------------------------------------------------------------

def _md_to_html_fallback(md_text: str) -> str:
    """Convert a subset of markdown to HTML using regex (no external deps).

    Handles: ``#/##/###`` headers, `` ``` `` code blocks, ``|`` tables,
    ``![]()`` images, ``**bold**``, `` ``code`` ``, ``---`` horizontal rules,
    blank-line paragraph breaks, and inline ``$...$`` / ``$$...$$`` LaTeX math
    (preserved for KaTeX).
    """
    out = []
    in_code_block = False
    in_paragraph = False
    code_lang = ''
    code_lines = []

    def _flush_paragraph():
        nonlocal in_paragraph
        if in_paragraph:
            out.append('</p>')
            in_paragraph = False

    def _process_inline(text: str) -> str:
        """Minimal inline markdown processing (bold, code, images)."""
        # Images: ![alt](url)  -- but only if not inside code
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', text)
        # Bold: **text**
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Inline code: `text`
        text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)
        return text

    def _emit_block(tag, text, hid=None):
        _flush_paragraph()
        if hid:
            out.append(f'<{tag} id="{hid}">{_process_inline(text)}</{tag}>')
        else:
            out.append(f'<{tag}>{_process_inline(text)}</{tag}>')

    # Pre-scan: collect table blocks so we can emit them as <table> at the right place
    # Build a set of line indices that belong to tables
    all_lines = md_text.split('\n')
    table_line_set = set()

    # Identify table blocks: a sequence of |...| lines (including separator |---|)
    i = 0
    while i < len(all_lines):
        line = all_lines[i].strip()
        if line.startswith('|') and line.endswith('|') and '|' in line[1:-1]:
            # Start of a table block
            block_start = i
            while i < len(all_lines) and all_lines[i].strip().startswith('|'):
                table_line_set.add(i)
                i += 1
            # This is a table block; we'll emit it when we encounter a special
            # placeholder in the stream.  Store the block.
            # To handle this cleanly, we emit a placeholder token and collect
            # tables later.
        else:
            i += 1

    # Build table blocks: group consecutive table line indices
    sorted_table_indices = sorted(table_line_set)
    table_blocks = []  # list of (first_line_idx, list_of_md_lines)
    if sorted_table_indices:
        block_start = sorted_table_indices[0]
        block_lines = [all_lines[block_start]]
        for idx in sorted_table_indices[1:]:
            if idx == block_start + len(block_lines):
                block_lines.append(all_lines[idx])
            else:
                table_blocks.append((block_start, block_lines))
                block_start = idx
                block_lines = [all_lines[idx]]
        table_blocks.append((block_start, block_lines))

    # Map from line index to table block index
    line_to_table_block = {}
    for tb_idx, (start, _lines) in enumerate(table_blocks):
        for offset in range(len(_lines)):
            line_to_table_block[start + offset] = tb_idx

    emitted_tables = set()
    line_idx = 0

    for line in all_lines:
        # If this line is part of a table block
        if line_idx in table_line_set:
            tb_idx = line_to_table_block.get(line_idx)
            if tb_idx is not None and tb_idx not in emitted_tables:
                emitted_tables.add(tb_idx)
                _flush_paragraph()
                _start, md_lines = table_blocks[tb_idx]
                out.append(_render_table(md_lines, _process_inline))
            line_idx += 1
            continue

        line_idx += 1

        # Code block fence
        stripped = line.strip()
        fence_match = re.match(r'^```(\w*)$', stripped)
        if fence_match and not in_code_block:
            _flush_paragraph()
            in_code_block = True
            code_lang = fence_match.group(1)
            code_lines = []
            continue
        if re.match(r'^```$', stripped) and in_code_block:
            lang_attr = f' class="language-{code_lang}"' if code_lang else ''
            out.append(f'<pre><code{lang_attr}>')
            out.append('\n'.join(code_lines))
            out.append('</code></pre>')
            in_code_block = False
            code_lines = []
            continue
        if in_code_block:
            code_lines.append(line)
            continue

        # Horizontal rule
        if stripped == '---':
            _flush_paragraph()
            out.append('<hr>')
            continue

        # Headers
        h3 = re.match(r'^### (.+)$', line)
        if h3:
            hid = _make_header_id(h3.group(1))
            _emit_block('h3', h3.group(1), hid)
            continue
        h2 = re.match(r'^## (.+)$', line)
        if h2:
            hid = _make_header_id(h2.group(1))
            _emit_block('h2', h2.group(1), hid)
            continue
        h1 = re.match(r'^# (.+)$', line)
        if h1:
            _emit_block('h1', h1.group(1))
            continue

        # Blank line -> paragraph break
        if stripped == '':
            _flush_paragraph()
            continue

        # Regular paragraph content
        if not in_paragraph:
            out.append('<p>')
            in_paragraph = True
            out.append(_process_inline(line))
        else:
            out.append('\n' + _process_inline(line))

    _flush_paragraph()

    raw = '\n'.join(out)
    # Clean up any leftover empty paragraphs
    raw = raw.replace('<p></p>', '')
    raw = re.sub(r'<p>\s*</p>', '', raw)

    return raw


def _render_table(md_lines, process_fn):
    """Render a sequence of markdown table lines as an HTML <table>."""
    # Parse rows
    header = [c.strip() for c in md_lines[0].strip('|').split('|')]
    rows = []
    for line in md_lines[1:]:
        stripped = line.strip()
        # Skip separator rows like |---|---|
        cells = [c.strip() for c in stripped.strip('|').split('|')]
        if all(re.match(r'^-{1,}:?-{1,}$', c) for c in cells if c):
            continue
        rows.append(cells)

    html_parts = ['<table>']
    # Header row
    html_parts.append('<tr>' + ''.join(f'<th>{process_fn(c)}</th>' for c in header) + '</tr>')
    # Data rows
    for row in rows:
        # Pad row if it has fewer cells than header
        while len(row) < len(header):
            row.append('')
        html_parts.append('<tr>' + ''.join(f'<td>{process_fn(c)}</td>' for c in row) + '</tr>')
    html_parts.append('</table>')
    return '\n'.join(html_parts)


def _make_header_id(text: str) -> str:
    """Generate an HTML-safe id from header text.

    Strips HTML tags, LaTeX math delimiters/commands, and special chars.
    """
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'[$\\()\[\]{}%=#]', '', clean)
    return clean.strip().lower().replace(' ', '-').replace('.', '')


def _add_header_ids(html: str) -> str:
    """Post-process HTML to add ``id`` attributes to h2/h3 elements."""

    def _add_id(match):
        tag = match.group(1)
        text = match.group(2)
        return f'<{tag} id="{_make_header_id(text)}">{text}'

    return re.sub(r'<(h[23])>(.+?)</\1>', _add_id, html)


def build_html_report(md_content: str, plot_dir: Path) -> str:
    """Build a self-contained HTML report from markdown content.

    Uses ``markdown`` library with ``tables`` and ``fenced_code`` extensions
    if available; otherwise falls back to a regex-based converter.

    Parameters
    ----------
    md_content : str
        Complete markdown report text.
    plot_dir : Path
        Directory used to resolve relative image paths (images are looked up
        relative to ``plot_dir.parent``).

    Returns
    -------
    str
        Self-contained HTML document with base64-embedded images.
    """
    # --- MD → HTML ---
    try:
        import markdown
        body_html = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code']
        )
    except ImportError:
        body_html = _md_to_html_fallback(md_content)

    # --- Add id attributes to headers (markdown library doesn't add them) ---
    body_html = _add_header_ids(body_html)

    # --- Base64-encode images ---
    # Match both local <img src="path"> references (exclude data: URIs)
    base_dir = plot_dir.parent.resolve()

    def _embed_image(match):
        src = match.group(1)
        if src.startswith('data:') or src.startswith('http://') or src.startswith('https://'):
            return match.group(0)
        img_path = base_dir / src
        if img_path.exists():
            try:
                data_uri = base64_encode_image(img_path)
                return match.group(0).replace(src, data_uri)
            except Exception:
                return match.group(0)
        return match.group(0)

    body_html = re.sub(
        r'<img\s+[^>]*src="([^"]+)"[^>]*>',
        _embed_image,
        body_html
    )

    # --- Build TOC from h2 headers ---
    toc_items = re.findall(r'<h2[^>]*>(.+?)</h2>', body_html)
    toc_html = '<ul>\n'
    for title in toc_items:
        hid = _make_header_id(title)
        toc_html += f'  <li><a href="#{hid}">{title}</a></li>\n'
    toc_html += '</ul>'

    # --- Full HTML document ---
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HII Boundary Detection -- Test Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0;
}}
#sidebar {{
    position: fixed;
    left: 0;
    top: 0;
    width: 260px;
    height: 100vh;
    background: #f7f7f7;
    border-right: 1px solid #ddd;
    padding: 20px;
    overflow-y: auto;
    box-sizing: border-box;
}}
#sidebar ul {{
    list-style: none;
    padding-left: 0;
}}
#sidebar li {{
    margin-bottom: 6px;
}}
#sidebar a {{
    text-decoration: none;
    color: #0366d6;
}}
#sidebar a:hover {{
    text-decoration: underline;
}}
#main {{
    margin-left: 280px;
    padding: 30px 40px;
    max-width: 1100px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 8px 12px;
}}
th {{
    background: #f5f5f5;
}}
tr:nth-child(even) {{
    background: #fafafa;
}}
pre {{
    background: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 15px;
    overflow-x: auto;
}}
code {{
    background: #f0f0f0;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: monospace;
}}
img {{
    max-width: 100%;
    border: 1px solid #eee;
    border-radius: 4px;
}}
@media (max-width: 768px) {{
    #sidebar {{
        position: static;
        width: 100%;
        height: auto;
    }}
    #main {{
        margin-left: 0;
    }}
}}
</style>
</head>
<body>
<div id="sidebar">
<h3>Contents</h3>
{toc_html}
</div>
<div id="main">
{body_html}
</div>
<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.getElementById("main"), {{
        delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$", right: "$", display: false}},
        ],
        throwOnError: false
    }});
}});
</script>
</body>
</html>'''

    return html


# ---------------------------------------------------------------------------
# 4. File writer
# ---------------------------------------------------------------------------

def write_report_files(
    md_content: str,
    html_content: str,
    output_dir: Path,
    suffix: str = "",
) -> Tuple[Path, Path]:
    """Write ``test_plan_result<suffix>.md`` and ``.html`` to *output_dir*.

    Parameters
    ----------
    suffix : str
        Appended to filename stem, e.g. ``"_1A"`` → ``test_plan_result_1A.html``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"test_plan_result{suffix}"
    md_path = output_dir / f"{stem}.md"
    html_path = output_dir / f"{stem}.html"

    md_path.write_text(md_content, encoding='utf-8')
    html_path.write_text(html_content, encoding='utf-8')

    return md_path, html_path


# ---------------------------------------------------------------------------
# 5. Real-data pipeline report
# ---------------------------------------------------------------------------

def _fmt_val(val, digits=3) -> str:
    """Format a value for table display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def build_real_report(
    phases: dict,
    catalog: Table,
    manifest: list[dict],
    config: dict,
    output_dir: Path,
) -> str:
    """Build a self-contained HTML report for the real-data pipeline.

    Parameters
    ----------
    phases : dict
        Per-phase status dicts (catalog, download, preprocess, detection).
    catalog : Table
        Astropy Table of catalog results.
    manifest : list[dict]
        Download manifest entries.
    config : dict
        Report configuration: timestamp, git_commit, python_version, algo_params.
    output_dir : Path
        Root output directory (plots_dir for image resolution).

    Returns
    -------
    str
        Self-contained HTML document.
    """
    lines = []

    # ── Header ──
    lines.append('# HII Region Boundary Detection — Real Data Pipeline Report')
    lines.append('')
    lines.append(f'**Date:** {config.get("timestamp", "N/A")}  ')
    lines.append(f'**Git commit:** `{config.get("git_commit", "N/A")}`  ')
    lines.append(f'**Python:** {config.get("python_version", "N/A")}  ')
    lines.append(f'**Config:** `{config.get("config_path", "N/A")}`  ')
    lines.append('')

    # ── Pipeline Summary ──
    lines.append('## Pipeline Summary')
    lines.append('')
    lines.append('| Phase | Status | Details |')
    lines.append('|-------|--------|---------|')
    for phase_name in ["catalog", "download", "preprocess", "detection"]:
        p = phases.get(phase_name, {})
        status = p.get("status", "pending")
        badge = {"done": "✅", "running": "🔄", "pending": "⏳", "failed": "❌"}.get(status, "❓")
        detail = ""
        if phase_name == "catalog":
            detail = f'{p.get("n_bubbles", "?")} bubbles'
        elif phase_name == "download":
            detail = f'{p.get("n_downloaded", "?")} images'
        elif phase_name == "preprocess":
            rr = p.get("result", {})
            detail = f'{rr.get("name", "?")} sources={rr.get("n_sources", "?")}'
        elif phase_name == "detection":
            rr = p.get("result")
            if rr and rr.success:
                detail = f'R={rr.boundary_mean_r_pixel:.1f} px  σ={rr.uncertainty_mean or 0:.2f} px'
            else:
                detail = f'FAIL — {getattr(rr, "error_message", "unknown") if rr else "no result"}'
        lines.append(f'| {phase_name} | {badge} {status} | {detail} |')
    lines.append('')

    # ── 1. Catalog ──
    lines.append('## 1. Catalog')
    lines.append('')
    lines.append(f'**Source:** Vizier J/ApJ/649/759/bubbles (Churchwell+ 2006)  ')
    lines.append(f'**Filter:** MFlags == "C" (closed ring morphology)  ')
    lines.append(f'**Sort:** <R> descending, top {len(catalog)}  ')
    lines.append('')

    # Build catalog table
    lines.append('| Name | GLON (°) | GLAT (°) | <R> (\') | MFlags |')
    try:
        radius_col = None
        for c in ["<R>", "R", "Radius", "rad", "AvgR"]:
            if c in catalog.colnames:
                radius_col = c
                break
        name_col = None
        for c in ["Name", "[CPA2006]", "CPA2006", "___"]:
            if c in catalog.colnames:
                name_col = c
                break
        for row in catalog:
            name = str(row[name_col]).strip() if name_col else "?"
            glon = row["GLON"] if "GLON" in catalog.colnames else "?"
            glat = row["GLAT"] if "GLAT" in catalog.colnames else "?"
            r_val = row[radius_col] if radius_col else "?"
            mf = row["MFlags"] if "MFlags" in catalog.colnames else "?"
            lines.append(f'| {name} | {_fmt_val(glon)} | {_fmt_val(glat)} | {_fmt_val(r_val)} | {mf} |')
    except Exception:
        lines.append('| (catalog parsing error) | | | | |')
    lines.append('')

    # ── 2. Image Download ──
    lines.append('## 2. Image Download')
    lines.append('')
    lines.append('| Name | ICRS (ra, dec) | Pixel Scale (\") | FITS Shape | Status |')
    lines.append('|------|-----------------|-----------------|------------|--------|')
    for entry in manifest:
        name = entry["name"]
        fits_path = Path(entry["fits_science"])
        if not fits_path.exists():
            # Fallback: check images/ directory
            fits_path = output_dir / "images" / fits_path.name
        status = "✅" if fits_path.exists() else "❌"
        ps = entry.get("pixel_scale_arcmin", "?")
        shape = entry.get("image_shape", ["?", "?"])
        shape_str = f"{shape[0]}×{shape[1]}" if isinstance(shape, list) else str(shape)
        lines.append(f'| {name} | — | {_fmt_val(ps)} | {shape_str} | {status} |')
    lines.append('')

    # ── 3. Preprocessing ──
    lines.append('## 3. Preprocessing')
    lines.append('')
    pp_phase = phases.get("preprocess", {})
    pp_result = pp_phase.get("result", {})
    if pp_result:
        name = pp_result.get("name", "?")
        lines.append(f'### {name}')
        lines.append('')
        lines.append(f'| Metric | Value |')
        lines.append(f'|--------|-------|')
        lines.append(f'| Point Sources | {pp_result.get("n_sources", "N/A")} |')
        lines.append(f'| σ_map mean | {_fmt_val(pp_result.get("sigma_mean"))} |')
        lines.append(f'| σ_map median | {_fmt_val(pp_result.get("sigma_median"))} |')
        lines.append('')

        diag_path = pp_result.get("diagnostic_path")
        if diag_path and Path(diag_path).exists():
            lines.append(f'![Preprocess Diagnostic]({diag_path})')
            lines.append('')
        # Scales diagnostic if debug_scales was enabled
        if config.get("debug_scales"):
            scales_diag = Path(diag_path).parent / f"{name}_scales_diagnostic.png" if diag_path else None
            if scales_diag and scales_diag.exists():
                lines.append(f'![Scales Diagnostic]({scales_diag})')
                lines.append('')
    else:
        lines.append('_No preprocessing data_')
        lines.append('')
    lines.append('')

    # ── 4. Boundary Detection ──
    lines.append('## 4. Boundary Detection')
    lines.append('')
    det_phase = phases.get("detection", {})
    det_result = det_phase.get("result")
    if det_result and det_result.success:
        name = det_result.name
        lines.append(f'### {name}')
        lines.append('')
        lines.append(f'| Metric | Value |')
        lines.append(f'|--------|-------|')
        lines.append(f'| Catalog Rout | {det_result.R_arcmin:.2f} arcmin |')
        lines.append(f'| Detected R | {det_result.boundary_mean_r_arcmin:.3f} arcmin ({det_result.boundary_mean_r_pixel:.1f} px) |')
        lines.append(f'| Mean Uncertainty | {_fmt_val(det_result.uncertainty_mean)} px |')
        lines.append(f'| Scenario | {det_result.scenario or "N/A"} |')
        lines.append('')

        for plot_key, plot_label in [("overlay", "Overlay"),
                                       ("pipeline", "Polar Pipeline"), ("algo_diag", "Algorithm Diagnostic")]:
            plot_path = det_result.plots.get(plot_key, "")
            if plot_path and Path(plot_path).exists():
                lines.append(f'![{plot_label}]({plot_path})')
                lines.append('')
    else:
        err = getattr(det_result, "error_message", "unknown") if det_result else "no result"
        lines.append(f'**Status:** ❌ FAIL — {err}')
        lines.append('')
    lines.append('')

    # ── Algorithm Parameters ──
    lines.append('## Algorithm Parameters')
    lines.append('')
    lines.append('| Parameter | Value |')
    lines.append('|-----------|-------|')
    algo_params = config.get("algo_params", {})
    if algo_params:
        for k, v in algo_params.items():
            lines.append(f'| {k} | {_fmt_val(v)} |')
    else:
        lines.append('| _(using hii_detection_config.yaml defaults)_ | |')
    lines.append('')

    # ── Build HTML ──
    md_content = '\n'.join(lines)
    return build_html_report(md_content, output_dir)
