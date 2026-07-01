"""Build and install the boundary_detection package.

Copies core modules + test suite from py/ root, rewrites internal imports for
package-relative resolution, generates setup.py, and optionally runs pip install.

Usage:
    python build_package.py                 # build + pip install -e
    python build_package.py --no-install    # build only
    python build_package.py --version 1.2.3 # explicit version
    python build_package.py --clean         # delete package dir + rebuild
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


# --- Constants ---

PACKAGE_DIR = Path(__file__).parent  # package/
PKG_NAME = 'boundary_detection'
PKG_PATH = PACKAGE_DIR / PKG_NAME      # package/boundary_detection/
TEST_PATH = PACKAGE_DIR / 'test'       # package/test/

# Source files are in the parent directory (py/ root)
SOURCE_DIR = PACKAGE_DIR.parent

# Saved during clean, restored by _copy_doc_files (zh + en test READMEs)
_saved_test_readme_zh: str | None = None
_saved_test_readme_en: str | None = None

# Core .py files to copy → package/boundary_detection/
CORE_MODULES = [
    'find_boundary_dp',
    'find_global_optimal_boundary',
    'extract_circle_boundary',
    'find_stable_boundary_by_scan',
    'find_stable_boundary',
    'bootstrap_boundary',
    'detect_hii_boundary',
    'detect_hii_boundary_elliptical',
]

# Test .py files to copy → package/test/
TEST_MODULES = [
    'test_analysis',
    'test_argmax_baseline',
    'test_generators',
    'test_models',
    'test_diagnostics',
    'test_runner',
    'test_runner_elliptical',
    'test_report',
    'run_test_plan',
    'run_test_plan_elliptical',
    'test_contrast_deep_diagnostic',
    'test_contrast_gradient_diagnostic',
    'logging_setup',
    # Real-data pipeline
    'fetch_bubble_catalog',
    'download_glimpse_images',
    'preprocess_glimpse',
    'test_real_bubbles',
    'run_test_plan_real',
    'quick_test',
]

# Additional files to copy as-is (configs etc.)
COPY_FILES = ['hii_detection_config.yaml']
TEST_COPY_FILES = ['test_config.yaml', 'test_config_elliptical.yaml', 'test_config_real.yaml']

# Bundled data directory (inside package)
DATA_PATH = PKG_PATH / 'data'

# Data files to bundle: (source_subdir, glob_pattern) relative to output_dir in test_config_real.yaml
# These are copied from the real-data output directory into the package.
# The default output_dir is "test_plots_real".
BUNDLED_DATA_SOURCE = 'test_plots_real'


# --- Version Resolution ---

def _resolve_version(explicit_version=None):
    """Resolve version: explicit > git describe --tags > '0.1.0'."""
    if explicit_version:
        return explicit_version.lstrip('v')

    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            capture_output=True, text=True, cwd=SOURCE_DIR,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().lstrip('v')
    except FileNotFoundError:
        pass

    return '0.1.0'


# --- File Operations ---

def _copy_source_files():
    """Copy core .py files and config from py/ root to package/boundary_detection/."""
    PKG_PATH.mkdir(parents=True, exist_ok=True)

    for name in CORE_MODULES:
        src = SOURCE_DIR / f'{name}.py'
        dst = PKG_PATH / f'{name}.py'
        shutil.copy2(src, dst)
        print(f'  copied: {name}.py')

    for name in COPY_FILES:
        src = SOURCE_DIR / name
        if src.exists():
            dst = PKG_PATH / name
            shutil.copy2(src, dst)
            print(f'  copied: {name}')


def _rewrite_imports():
    """Rewrite internal absolute imports to package-relative imports.

    e.g. 'from find_stable_boundary import ...' -> 'from .find_stable_boundary import ...'
    """
    for name in CORE_MODULES:
        file_path = PKG_PATH / f'{name}.py'
        content = file_path.read_text()

        for mod in CORE_MODULES:
            # Replace 'from <mod> import' -> 'from .<mod> import'
            old = f'from {mod} import'
            new = f'from .{mod} import'
            content = content.replace(old, new)

        file_path.write_text(content)
    print('  imports rewritten')


def _write_setup_py(version):
    """Generate setup.py in the package directory (package/setup.py)."""
    setup_content = f'''from setuptools import setup, find_packages

setup(
    name='{PKG_NAME}',
    version='{version}',
    description='HII region stable boundary detection with bootstrap uncertainty',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24,<2',
        'scipy',
        'matplotlib',
        'astropy',
        'pyyaml',
        'scikit-image',
        'astroquery',
        'watroo',
    ],
    python_requires='>=3.10',
    include_package_data=True,
)
'''
    (PACKAGE_DIR / 'setup.py').write_text(setup_content)
    print(f'  generated: setup.py (version {version})')


def _write_init_py():
    """Write __init__.py with public API export."""
    init_content = '''# boundary_detection - HII region boundary detection with bootstrap uncertainty
from .detect_hii_boundary import detect_hii_boundary
from .detect_hii_boundary_elliptical import detect_hii_boundary_elliptical
'''
    (PKG_PATH / '__init__.py').write_text(init_content)
    print('  generated: __init__.py')


def _copy_test_files():
    """Copy test .py files and configs from py/ root to package/test/."""
    TEST_PATH.mkdir(parents=True, exist_ok=True)

    for name in TEST_MODULES:
        src = SOURCE_DIR / f'{name}.py'
        if src.exists():
            dst = TEST_PATH / f'{name}.py'
            shutil.copy2(src, dst)
            print(f'  copied: test/{name}.py')

    for name in TEST_COPY_FILES:
        src = SOURCE_DIR / name
        if src.exists():
            dst = TEST_PATH / name
            shutil.copy2(src, dst)
            print(f'  copied: test/{name}')


def _rewrite_test_imports():
    """Rewrite test file imports for the package structure.

    Only rewrites core module imports \u2192 boundary_detection.xxx.
    Test-internal imports are left as absolute (from test_xxx import)
    so files can be run directly with ``python run_test_plan.py``.
    (The ``test`` package name conflicts with stdlib, so -m import
    of test.* is not viable regardless.)
    """
    for name in TEST_MODULES:
        file_path = TEST_PATH / f'{name}.py'
        content = file_path.read_text()

        # Core module imports \u2192 boundary_detection.xxx
        for mod in CORE_MODULES:
            old = f'from {mod} import'
            new = f'from boundary_detection.{mod} import'
            content = content.replace(old, new)

        file_path.write_text(content)
    print('  test imports rewritten')


def _write_test_init_py():
    """Write minimal __init__.py for test package."""
    (TEST_PATH / '__init__.py').write_text('# boundary_detection test suite\n')
    print('  generated: test/__init__.py')


def _copy_data_files():
    """Copy bundled data (catalog, images, manifest) into the package.

    Copies from ``SOURCE_DIR / BUNDLED_DATA_SOURCE`` to ``DATA_PATH``.
    Rewrites the manifest so ``fits_science`` paths are relative filenames.
    """
    src_data = SOURCE_DIR / BUNDLED_DATA_SOURCE
    if not src_data.is_dir():
        print(f'  skipped data (source not found: {src_data})')
        return

    _clean_dir(DATA_PATH)

    # Catalog
    src_cat = src_data / 'catalog' / 'bubble_catalog.csv'
    if src_cat.exists():
        dst_cat = DATA_PATH / 'catalog'
        dst_cat.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_cat, dst_cat / 'bubble_catalog.csv')
        print('  copied data: catalog/bubble_catalog.csv')

    # Images (manifest + FITS files)
    src_img = src_data / 'images'
    if src_img.is_dir():
        dst_img = DATA_PATH / 'images'
        dst_img.mkdir(parents=True, exist_ok=True)

        # Copy manifest — rewrite fits paths to filenames only
        manifest_src = src_img / 'download_manifest.json'
        if manifest_src.exists():
            with open(manifest_src) as f:
                manifest = json.load(f)
            for entry in manifest:
                if entry.get('fits_science'):
                    entry['fits_science'] = Path(entry['fits_science']).name
            manifest_dst = dst_img / 'download_manifest.json'
            with open(manifest_dst, 'w') as f:
                json.dump(manifest, f, indent=2)
            print('  copied data: images/download_manifest.json (paths rewritten)')

        # Copy FITS science files
        for fits_file in sorted(src_img.glob('*_8um_science.fits')):
            shutil.copy2(fits_file, dst_img / fits_file.name)
            size_mb = fits_file.stat().st_size / (1024 * 1024)
            print(f'  copied data: images/{fits_file.name} ({size_mb:.0f} MB)')


def _clean_dir(path: Path):
    """Remove directory if it exists, then recreate empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _ensure_doc_files():
    """Ensure bilingual README files (zh + en) are present in the package.

    - ``package/README.zh.md`` + ``package/README.en.md``: main package READMEs
      (live at package root, not affected by clean; verify existence).
    - ``test/README.zh.md`` + ``test/README.en.md``: test READMEs, restored from
      saved content after clean, or copied from source if available.
    """
    global _saved_test_readme_zh, _saved_test_readme_en

    # Main READMEs — live at package root, not affected by clean
    for lang in ('zh', 'en'):
        path = PACKAGE_DIR / f'README.{lang}.md'
        status = '✓' if path.exists() else '— MISSING (create manually)'
        print(f'  doc: package/README.{lang}.md {status}')

    # Test READMEs — restore after clean, or copy from source
    for lang, saved in [('zh', _saved_test_readme_zh), ('en', _saved_test_readme_en)]:
        filename = f'README.{lang}.md'
        test_readme = TEST_PATH / filename

        if saved is not None:
            test_readme.write_text(saved)
            print(f'  doc: test/{filename} (restored)')
        elif test_readme.exists():
            print(f'  doc: test/{filename}')
        else:
            src = SOURCE_DIR / 'package' / 'test' / filename
            if src.exists():
                shutil.copy2(src, test_readme)
                print(f'  doc: test/{filename} (copied from source)')
            else:
                print(f'  doc: test/{filename} — MISSING (create manually)')


# --- Install ---

def _pip_install():
    """Run pip install -e on the package."""
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-e', str(PACKAGE_DIR)],
        check=True,
    )
    print(f'  installed: {PKG_NAME} (editable)')


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description='Build and install boundary_detection package'
    )
    parser.add_argument('--no-install', action='store_true',
                        help='Build only, skip pip install')
    parser.add_argument('--version', type=str, default=None,
                        help='Explicit version string (e.g. 1.2.3)')
    parser.add_argument('--clean', action='store_true',
                        help='Delete package directory and rebuild')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip post-install verification (quick_test.py)')
    args = parser.parse_args()

    # Clean
    global _saved_test_readme_zh, _saved_test_readme_en
    _saved_test_readme_zh = None
    _saved_test_readme_en = None
    if args.clean:
        if PKG_PATH.exists():
            shutil.rmtree(PKG_PATH)
            print(f'cleaned: {PKG_PATH}')
        if TEST_PATH.exists():
            zh = TEST_PATH / 'README.zh.md'
            if zh.exists():
                _saved_test_readme_zh = zh.read_text()
            en = TEST_PATH / 'README.en.md'
            if en.exists():
                _saved_test_readme_en = en.read_text()
            shutil.rmtree(TEST_PATH)
            print(f'cleaned: {TEST_PATH}')
        if DATA_PATH.exists():
            shutil.rmtree(DATA_PATH)
            print(f'cleaned: {DATA_PATH}')
        setup_py = PACKAGE_DIR / 'setup.py'
        if setup_py.exists():
            setup_py.unlink()
            print(f'cleaned: {setup_py}')

    # Version
    version = _resolve_version(args.version)
    print(f'version: {version}')

    # Build core
    print('building core package...')
    _copy_source_files()
    _rewrite_imports()
    _write_setup_py(version)
    _write_init_py()

    # Build test suite
    print('building test suite...')
    _copy_test_files()
    _rewrite_test_imports()
    _write_test_init_py()

    # Ensure documentation files
    print('checking documentation...')
    _ensure_doc_files()

    # Bundle data files
    print('bundling data files...')
    _copy_data_files()

    # Install
    if not args.no_install:
        print('installing...')
        _pip_install()

    # Post-install verification
    if not args.no_test:
        print('\n' + '=' * 60)
        print('Running post-install verification (quick_test.py)')
        print('=' * 60)
        result = subprocess.run(
            [sys.executable, str(TEST_PATH / 'quick_test.py')],
            cwd=str(TEST_PATH),
        )
        if result.returncode != 0:
            print(f'\n\u274c Post-install tests FAILED (exit {result.returncode})')
            sys.exit(result.returncode)
        print('Post-install tests passed \u2705')

    print(f'\n\u2705 {PKG_NAME} v{version} ready')
    print(f'   core: {PKG_PATH}')
    print(f'   test: {TEST_PATH}')


if __name__ == '__main__':
    main()
