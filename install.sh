#!/usr/bin/env bash
#
# palmar-velle zero-compile installer
#
#   curl -fsSL https://raw.githubusercontent.com/your-org/palmar-velle/main/install.sh | sudo bash
#   sudo ./install.sh --uninstall
#
# Environment overrides:
#   PALMAR_VELLE_REPO      GitHub "owner/repo"   (default: your-org/palmar-velle)
#   PALMAR_VELLE_VERSION   release tag, e.g. v0.2.0   (default: latest)
#
# Release assets expected:
#   palmar-velle-x86_64-linux, palmar-velle-aarch64-linux, hand_landmark.onnx,
#   palmar-velle.desktop, palmar-velle.svg, SHA256SUMS
set -euo pipefail

REPO="${PALMAR_VELLE_REPO:-your-org/palmar-velle}"
VERSION="${PALMAR_VELLE_VERSION:-latest}"

BIN_DIR=/usr/local/bin
SHARE_DIR=/usr/local/share/palmar-velle
APP_DIR=/usr/share/applications
ICON_DIR=/usr/share/icons/hicolor/scalable/apps
UDEV_RULE=/etc/udev/rules.d/99-palmar-velle.rules
MODLOAD=/etc/modules-load.d/palmar-velle.conf

log()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root, e.g.:  curl -fsSL <url>/install.sh | sudo bash"
[[ "$(uname -s)" == Linux ]] || die "Linux only"

refresh_caches() {
    command -v update-desktop-database >/dev/null && update-desktop-database -q "$APP_DIR" || true
    command -v gtk-update-icon-cache  >/dev/null && gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
}

reload_udev() {
    if command -v udevadm >/dev/null; then
        udevadm control --reload-rules || true
        udevadm trigger --action=add --subsystem-match=misc --sysname-match=uinput || true
    fi
}

uninstall() {
    log "Removing palmar-velle"
    rm -f "$BIN_DIR/palmar-velle" "$APP_DIR/palmar-velle.desktop" \
          "$ICON_DIR/palmar-velle.svg" "$UDEV_RULE" "$MODLOAD"
    rm -rf "$SHARE_DIR"
    reload_udev
    refresh_caches
    log "Done. Group membership ('input') and ~/.config/palmar-velle were left untouched."
    exit 0
}

case "${1:-}" in
    --uninstall) uninstall ;;
    -h|--help)   sed -n '2,15p' "$0" 2>/dev/null || true; exit 0 ;;
    "")          ;;
    *)           die "unknown option: $1" ;;
esac

command -v curl >/dev/null      || die "curl is required"
command -v sha256sum >/dev/null || die "sha256sum is required"

# --- architecture -----------------------------------------------------------
case "$(uname -m)" in
    x86_64|amd64)  ARCH=x86_64 ;;
    aarch64|arm64) ARCH=aarch64 ;;
    *) die "unsupported architecture: $(uname -m) (need x86_64 or aarch64)" ;;
esac

if [[ "$VERSION" == latest ]]; then
    BASE="https://github.com/$REPO/releases/latest/download"
else
    BASE="https://github.com/$REPO/releases/download/$VERSION"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetch() {
    curl -fL --retry 3 --proto '=https' --tlsv1.2 -sS -o "$TMP/$1" "$BASE/$1" \
        || die "download failed: $BASE/$1"
}

# --- download + verify ------------------------------------------------------
log "Downloading $REPO ($VERSION) for $ARCH"
for f in "palmar-velle-$ARCH-linux" hand_landmark.onnx palmar-velle.desktop palmar-velle.svg SHA256SUMS; do
    fetch "$f"
done

log "Verifying checksums"
( cd "$TMP" && sha256sum --ignore-missing -c SHA256SUMS ) || die "checksum verification failed"

# --- install files ----------------------------------------------------------
log "Installing files"
install -Dm755 "$TMP/palmar-velle-$ARCH-linux"  "$BIN_DIR/palmar-velle"
install -Dm644 "$TMP/hand_landmark.onnx"        "$SHARE_DIR/hand_landmark.onnx"
install -Dm644 "$TMP/palmar-velle.desktop"      "$APP_DIR/palmar-velle.desktop"
install -Dm644 "$TMP/palmar-velle.svg"          "$ICON_DIR/palmar-velle.svg"

# --- uinput access ----------------------------------------------------------
log "Writing $UDEV_RULE"
cat > "$UDEV_RULE" <<'EOF'
# palmar-velle: let the 'input' group (and the active-seat user, via uaccess)
# create virtual input devices through /dev/uinput.
KERNEL=="uinput", SUBSYSTEM=="misc", OPTIONS+="static_node=uinput", TAG+="uaccess", MODE="0660", GROUP="input"
EOF
chmod 644 "$UDEV_RULE"

# uinput is not loaded by default on every distro
echo uinput > "$MODLOAD"
modprobe uinput 2>/dev/null || warn "could not load the uinput module now; it will load at next boot"

getent group input >/dev/null || groupadd -r input

INSTALL_USER="${SUDO_USER:-}"
if [[ -z "$INSTALL_USER" || "$INSTALL_USER" == root ]]; then
    INSTALL_USER="$(logname 2>/dev/null || true)"
fi
GROUP_ADDED=0
if [[ -n "$INSTALL_USER" && "$INSTALL_USER" != root ]] && id "$INSTALL_USER" >/dev/null 2>&1; then
    if id -nG "$INSTALL_USER" | tr ' ' '\n' | grep -qx input; then
        log "$INSTALL_USER is already in the 'input' group"
    else
        usermod -aG input "$INSTALL_USER"
        GROUP_ADDED=1
        log "Added $INSTALL_USER to the 'input' group"
    fi
else
    warn "could not determine the invoking user; run: sudo usermod -aG input <user>"
fi

reload_udev
refresh_caches

log "Installed palmar-velle to $BIN_DIR/palmar-velle"
if [[ $GROUP_ADDED -eq 1 ]]; then
    warn "Log out and back in so the new 'input' group membership takes effect."
fi
echo "Launch it from your app menu, or run:  palmar-velle --gui"
