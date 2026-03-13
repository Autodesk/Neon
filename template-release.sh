#!/bin/bash
set -e

# =============================================================================
# GitHub Release Script Template
# =============================================================================
# Creates a GitHub release with an asset file (e.g., wheel).
#
# Usage:
#   ./release.sh                     # Create release for current version
#   ./release.sh --force             # Move existing tag to current commit
#   ./release.sh --dry-run           # Show what would be done without doing it
#   ./release.sh --help              # Show this help message
#
# Prerequisites:
#   - GitHub CLI (gh) installed and authenticated
#   - Asset file already built
#   - Changes committed and pushed
# =============================================================================

# =============================================================================
# CONFIGURATION - Edit these parameters for your release
# =============================================================================

# Version string (e.g., "0.5.2a1", "1.0.0")
VERSION="0.5.2a1"

# Tag name (typically "v" + VERSION)
TAG="v${VERSION}"

# Path to asset file(s) to upload (e.g., wheel, tarball)
# Use wildcards if needed, or specify exact path
ASSET_PATTERN="dist/neon_gpu-${VERSION}-*.whl"

# GitHub repository (owner/repo format)
GITHUB_REPO="Autodesk/Neon"

# Release notes - ADD YOUR NOTES HERE
RELEASE_NOTES="## Release ${VERSION}

### Highlights
- Add your release highlights here
- Another highlight

### Installation

\`\`\`bash
pip install https://github.com/${GITHUB_REPO}/releases/download/${TAG}/ASSET_FILENAME_HERE
\`\`\`

### New Features
- Feature 1: Description
- Feature 2: Description

### Bug Fixes
- Fix 1: Description

### Breaking Changes
- None (or list them)

### Supported GPU Architectures
- sm_70 (Volta)
- sm_75 (Turing)
- sm_80 (Ampere)
- sm_86 (Ampere)
- sm_89 (Ada Lovelace)
- sm_90 (Hopper)
"

# =============================================================================
# END OF CONFIGURATION - No need to edit below this line
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            head -16 "$0" | tail -13
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find the asset file
ASSET_FILE=$(ls -1 ${ASSET_PATTERN} 2>/dev/null | head -1)

if [[ -z "$ASSET_FILE" ]]; then
    echo "ERROR: No asset file found matching pattern: ${ASSET_PATTERN}"
    echo "Make sure the file exists before running this script."
    exit 1
fi

# Update release notes with actual asset filename
RELEASE_NOTES="${RELEASE_NOTES//ASSET_FILENAME_HERE/$(basename ${ASSET_FILE})}"

echo "==> Release Configuration"
echo "    Version: ${VERSION}"
echo "    Tag: ${TAG}"
echo "    Asset: ${ASSET_FILE}"
echo "    Repository: ${GITHUB_REPO}"
echo ""

# Check if it's a pre-release (contains a, b, rc, or dev)
PRERELEASE_FLAG=""
if [[ "$VERSION" =~ (a|b|rc|dev) ]]; then
    PRERELEASE_FLAG="--prerelease"
    echo "    Type: Pre-release (alpha/beta/rc)"
else
    echo "    Type: Stable release"
fi
echo ""

if $DRY_RUN; then
    echo "==> DRY RUN - Would execute the following commands:"
    echo ""
    if $FORCE; then
        echo "# Force mode: delete and recreate tag"
        echo "git push origin :refs/tags/${TAG}"
        echo "git tag -d ${TAG}"
    fi
    echo "git tag -a ${TAG} -m \"Release ${VERSION}\""
    echo "git push origin ${TAG}"
    echo "gh release create ${TAG} --title \"${TAG}\" ${PRERELEASE_FLAG} \"${ASSET_FILE}\""
    echo ""
    echo "Release notes would be:"
    echo "----------------------------------------"
    echo "$RELEASE_NOTES"
    echo "----------------------------------------"
    exit 0
fi

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    if $FORCE; then
        echo "==> Force mode: Moving tag ${TAG} to current commit..."
        # Delete remote tag
        echo "    Deleting remote tag..."
        git push origin ":refs/tags/${TAG}" 2>/dev/null || true
        # Delete local tag
        echo "    Deleting local tag..."
        git tag -d "${TAG}" 2>/dev/null || true
        # Create new tag
        echo "    Creating new tag..."
        git tag -a "${TAG}" -m "Release ${VERSION}"
        echo "    Pushing tag to remote..."
        git push origin "${TAG}"
    else
        echo "==> Tag ${TAG} already exists (use --force to move it to current commit)"
    fi
else
    echo "==> Creating tag ${TAG}..."
    git tag -a "${TAG}" -m "Release ${VERSION}"
    echo "==> Pushing tag to remote..."
    git push origin "${TAG}"
fi

# Check if release already exists
if gh release view "${TAG}" >/dev/null 2>&1; then
    echo "==> Release ${TAG} already exists. Updating..."
    gh release upload "${TAG}" "${ASSET_FILE}" --clobber
else
    echo "==> Creating GitHub release..."
    gh release create "${TAG}" \
        --title "${TAG}" \
        --notes "${RELEASE_NOTES}" \
        ${PRERELEASE_FLAG} \
        "${ASSET_FILE}"
fi

echo ""
echo "==> Release complete!"
echo "    URL: https://github.com/${GITHUB_REPO}/releases/tag/${TAG}"
echo ""
echo "    Install with:"
echo "    pip install https://github.com/${GITHUB_REPO}/releases/download/${TAG}/$(basename ${ASSET_FILE})"
