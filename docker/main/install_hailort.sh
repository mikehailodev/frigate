#!/bin/bash

set -euxo pipefail

hailo_version="4.23.0"

if [[ "${TARGETARCH}" == "amd64" ]]; then
    arch="x86_64"
    # amd64: use frigate-nvr repackage (no official .deb available)
    wget -qO- "https://github.com/frigate-nvr/hailort/releases/download/v${hailo_version}/hailort-debian12-${TARGETARCH}.tar.gz" | tar -C / -xzf -
elif [[ "${TARGETARCH}" == "arm64" ]]; then
    arch="aarch64"
    # arm64: use official Hailo .deb (built with HAILO_BUILD_SERVICE for multi-process support)
    wget -q "https://github.com/mikehailodev/frigate/releases/download/hailort-4.23.0-service/hailort_${hailo_version}_${TARGETARCH}.deb" -O /tmp/hailort.deb
    tmpdir=$(mktemp -d)
    dpkg-deb -x /tmp/hailort.deb "$tmpdir"
    # Create /rootfs/ structure expected by Dockerfile COPY --from=wheels /rootfs/ /
    mkdir -p /rootfs/usr/local/lib /rootfs/usr/local/bin
    cp "$tmpdir"/usr/lib/libhailort.so* /rootfs/usr/local/lib/
    cp "$tmpdir"/usr/bin/hailortcli /rootfs/usr/local/bin/
    # Also install to system paths for wheel build
    cp "$tmpdir"/usr/lib/libhailort.so* /usr/local/lib/
    cp "$tmpdir"/usr/bin/hailortcli /usr/local/bin/
    ldconfig
    rm -rf "$tmpdir" /tmp/hailort.deb
fi

wget -P /wheels/ "https://github.com/frigate-nvr/hailort/releases/download/v${hailo_version}/hailort-${hailo_version}-cp311-cp311-linux_${arch}.whl"
