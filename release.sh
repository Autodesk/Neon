#!/bin/bash
set -euo pipefail

# =============================================================================
# GitHub Release Script — Neon wheels
# =============================================================================
# Creates/updates a GitHub release and uploads ALL matching wheel assets.
# Configured for the multi-Python build output in dist-multi/.
#
# Usage:
#   ./release.sh            # Create release for current version
#   ./release.sh --force    # Move existing tag to current commit
#   ./release.sh --dry-run  # Show what would be done without doing it
#   ./release.sh --help     # Show this help message
#
# Prerequisites:
#   - GitHub CLI (gh) installed and authenticated (gh auth login)
#   - Wheels already built in dist-multi/
#   - Changes committed and pushed
# =============================================================================

# --- Configuration ----------------------------------------------------------
VERSION="0.5.2a2"
TAG="v${VERSION}"
# All wheels for this version (multi-Python build output).
ASSET_PATTERN="dist-multi/neon_gpu-${VERSION}-*.whl"
GITHUB_REPO="Autodesk/Neon"

RELEASE_NOTES="## Release ${VERSION}

Pre-release of the \`neon_gpu\` wheels built against the update4 Warp stack.

### Highlights
- Built against Warp (external-source-support-update4) with the BVH shared-stack
  fix for Neon's 3D thread blocks (correct mesh boundary masks / force coeffs).
- Adds Blackwell GPU support (sm_100 data-center, sm_120 consumer) in addition to
  Volta–Hopper (70, 75, 80, 86, 89, 90).

### Supported GPU architectures
- sm_70 (Volta), sm_75 (Turing), sm_80/sm_86 (Ampere), sm_89 (Ada),
  sm_90 (Hopper), sm_100 (Blackwell DC), sm_120 (Blackwell consumer)

### Python versions
- CPython 3.11, 3.12, 3.13, 3.14 (linux_x86_64)

### Installation
\`\`\`bash
pip install https://github.com/${GITHUB_REPO}/releases/download/${TAG}/<wheel-for-your-python>.whl
\`\`\`
"
# --- End configuration -------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
FORCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --force|-f) FORCE=true; shift ;;
        --help|-h) head -20 "$0" | tail -16; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: GitHub CLI (gh) is not installed. Install it and run 'gh auth login'."
    exit 1
fi

# Collect all matching wheels.
mapfile -t ASSET_FILES < <(ls -1 ${ASSET_PATTERN} 2>/dev/null || true)
if [[ ${#ASSET_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No wheels found matching: ${ASSET_PATTERN}"
    echo "Build them first (e.g. ./docker/build-wheels-multi.sh) and ensure the"
    echo "version in pyproject.toml is ${VERSION}."
    exit 1
fi

# Detect pre-release (contains a/b/rc/dev).
PRERELEASE_FLAG=""
if [[ "$VERSION" =~ (a|b|rc|dev) ]]; then
    PRERELEASE_FLAG="--prerelease"
fi

echo "==> Release Configuration"
echo "    Version:    ${VERSION}"
echo "    Tag:        ${TAG}"
echo "    Repository: ${GITHUB_REPO}"
echo "    Pre-release: ${PRERELEASE_FLAG:-no}"
echo "    Assets (${#ASSET_FILES[@]}):"
for f in "${ASSET_FILES[@]}"; do echo "      - $f"; done
echo ""

if $DRY_RUN; then
    echo "==> DRY RUN — would execute:"
    if $FORCE; then
        echo "git push origin :refs/tags/${TAG}   # (if exists)"
        echo "git tag -d ${TAG}                    # (if exists)"
    fi
    echo "git tag -a ${TAG} -m \"Release ${VERSION}\""
    echo "git push origin ${TAG}"
    echo "gh release create ${TAG} -R ${GITHUB_REPO} --title ${TAG} ${PRERELEASE_FLAG} <notes> ${ASSET_FILES[*]}"
    exit 0
fi

# Tag handling.
if git rev-parse "$TAG" >/dev/null 2>&1; then
    if $FORCE; then
        echo "==> Force: moving tag ${TAG} to current commit..."
        git push origin ":refs/tags/${TAG}" 2>/dev/null || true
        git tag -d "${TAG}" 2>/dev/null || true
        git tag -a "${TAG}" -m "Release ${VERSION}"
        git push origin "${TAG}"
    else
        echo "==> Tag ${TAG} already exists (use --force to move it)."
    fi
else
    echo "==> Creating tag ${TAG}..."
    git tag -a "${TAG}" -m "Release ${VERSION}"
    git push origin "${TAG}"
fi

# Create or update the release, then upload all wheels.
if gh release view "${TAG}" -R "${GITHUB_REPO}" >/dev/null 2>&1; then
    echo "==> Release ${TAG} exists; uploading/replacing assets..."
    gh release upload "${TAG}" -R "${GITHUB_REPO}" "${ASSET_FILES[@]}" --clobber
else
    echo "==> Creating GitHub release ${TAG}..."
    gh release create "${TAG}" -R "${GITHUB_REPO}" \
        --title "${TAG}" \
        --notes "${RELEASE_NOTES}" \
        ${PRERELEASE_FLAG} \
        "${ASSET_FILES[@]}"
fi

echo ""
echo "==> Release complete: https://github.com/${GITHUB_REPO}/releases/tag/${TAG}"
