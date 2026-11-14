# Maintainer: Your Name <you@example.com>
pkgname=palmar-velle
pkgver=0.2.0
pkgrel=1
pkgdesc="Webcam hand-gesture mouse, shortcut and command control (uinput, Wayland-friendly)"
arch=('x86_64' 'aarch64')
url="https://github.com/your-org/palmar-velle"
license=('AGPL-3.0-or-later')
depends=('opencv' 'v4l-utils' 'libxkbcommon' 'wayland' 'libglvnd' 'systemd-libs')
makedepends=('cargo' 'clang')
optdepends=('libnotify: notify-send used by freshly recorded gestures')
source=("$pkgname-$pkgver.tar.gz::$url/archive/v$pkgver.tar.gz"
        "hand_landmark.onnx::$url/releases/download/v$pkgver/hand_landmark.onnx")
# Run `updpkgsums` to fill these in.
sha256sums=('SKIP'
            'SKIP')

prepare() {
    cd "$pkgname-$pkgver"
    export RUSTUP_TOOLCHAIN=stable
    cargo fetch --locked --target "$(rustc -vV | sed -n 's/host: //p')"
}

build() {
    cd "$pkgname-$pkgver"
    export RUSTUP_TOOLCHAIN=stable
    export CARGO_TARGET_DIR=target
    cargo build --frozen --release
}

package() {
    cd "$pkgname-$pkgver"
    install -Dm755 target/release/palmar-velle "$pkgdir/usr/bin/palmar-velle"
    install -Dm644 "$srcdir/hand_landmark.onnx" "$pkgdir/usr/share/palmar-velle/hand_landmark.onnx"
    install -Dm644 assets/palmar-velle.desktop "$pkgdir/usr/share/applications/palmar-velle.desktop"
    install -Dm644 assets/palmar-velle.svg "$pkgdir/usr/share/icons/hicolor/scalable/apps/palmar-velle.svg"
    install -Dm644 assets/99-palmar-velle.rules "$pkgdir/usr/lib/udev/rules.d/99-palmar-velle.rules"
    install -Dm644 packaging/palmar-velle-modules.conf "$pkgdir/usr/lib/modules-load.d/palmar-velle.conf"
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
