#!/bin/bash

set -euxo pipefail

# HailoRT versions: 4.x for Hailo-8/8L, 5.x for Hailo-10H
HAILO4_VERSION="4.23.0"
HAILO5_VERSION="5.3.0"

# Map Docker TARGETARCH to wheel arch suffix
if [[ "${TARGETARCH}" == "amd64" ]]; then
    whl_arch="x86_64"
    deb_arch="amd64"
elif [[ "${TARGETARCH}" == "arm64" ]]; then
    whl_arch="aarch64"
    deb_arch="arm64"
else
    echo "Unsupported architecture: ${TARGETARCH}"
    exit 1
fi

# Artifact directory (bind-mounted during build, or empty)
ARTIFACTS_DIR="/deps/hailort_artifacts"

# Output directories
mkdir -p /rootfs/usr/local/lib /rootfs/opt/hailort4 /rootfs/opt/hailort5

# --- Helper: get artifact (local file or download) ---
get_artifact() {
    local filename="$1"
    local url="$2"
    local dest="$3"

    if [[ -f "${ARTIFACTS_DIR}/${filename}" ]]; then
        echo "Using local artifact: ${filename}"
        cp "${ARTIFACTS_DIR}/${filename}" "${dest}"
    elif [[ -n "${url}" ]]; then
        echo "Downloading: ${url}"
        wget -q "${url}" -O "${dest}"
    else
        echo "ERROR: No local file and no URL for ${filename}"
        exit 1
    fi
}

# --- Install HailoRT 4.x (Hailo-8/8L) ---
HAILO4_DEB="hailort_${HAILO4_VERSION}_${deb_arch}.deb"
HAILO4_WHL="hailort-${HAILO4_VERSION}-cp311-cp311-linux_${whl_arch}.whl"
# URLs default to GitHub release; can be overridden via build args
HAILO4_DEB_URL="${HAILO4_DEB_URL:-https://github.com/mikehailodev/frigate/releases/download/hailort-dual-4.23-5.3/${HAILO4_DEB}}"
HAILO4_WHL_URL="${HAILO4_WHL_URL:-https://github.com/mikehailodev/frigate/releases/download/hailort-dual-4.23-5.3/${HAILO4_WHL}}"

get_artifact "${HAILO4_DEB}" "${HAILO4_DEB_URL}" "/tmp/hailort4.deb"
tmpdir4=$(mktemp -d)
dpkg-deb -x /tmp/hailort4.deb "${tmpdir4}"
# Install only the shared library (no hailortcli, no hailort_service)
cp "${tmpdir4}"/usr/lib/libhailort.so.${HAILO4_VERSION} /rootfs/usr/local/lib/
ln -sf libhailort.so.${HAILO4_VERSION} /rootfs/usr/local/lib/libhailort.so.4
rm -rf "${tmpdir4}" /tmp/hailort4.deb

# Install Python wheel to isolated path
get_artifact "${HAILO4_WHL}" "${HAILO4_WHL_URL}" "/tmp/${HAILO4_WHL}"
pip3 install --no-deps --target=/rootfs/opt/hailort4 "/tmp/${HAILO4_WHL}"
rm -f "/tmp/${HAILO4_WHL}"

# --- Install HailoRT 5.x (Hailo-10H) ---
HAILO5_DEB="hailort_${HAILO5_VERSION}_${deb_arch}.deb"
HAILO5_WHL="hailort-${HAILO5_VERSION}-cp311-cp311-linux_${whl_arch}.whl"
HAILO5_DEB_URL="${HAILO5_DEB_URL:-https://github.com/mikehailodev/frigate/releases/download/hailort-dual-4.23-5.3/${HAILO5_DEB}}"
HAILO5_WHL_URL="${HAILO5_WHL_URL:-https://github.com/mikehailodev/frigate/releases/download/hailort-dual-4.23-5.3/${HAILO5_WHL}}"

get_artifact "${HAILO5_DEB}" "${HAILO5_DEB_URL}" "/tmp/hailort5.deb"
tmpdir5=$(mktemp -d)
dpkg-deb -x /tmp/hailort5.deb "${tmpdir5}"
cp "${tmpdir5}"/usr/lib/libhailort.so.${HAILO5_VERSION} /rootfs/usr/local/lib/
ln -sf libhailort.so.${HAILO5_VERSION} /rootfs/usr/local/lib/libhailort.so.5
rm -rf "${tmpdir5}" /tmp/hailort5.deb

get_artifact "${HAILO5_WHL}" "${HAILO5_WHL_URL}" "/tmp/${HAILO5_WHL}"
pip3 install --no-deps --target=/rootfs/opt/hailort5 "/tmp/${HAILO5_WHL}"
rm -f "/tmp/${HAILO5_WHL}"

# --- ldconfig will run in the deps stage after COPY --from=deps-rootfs ---
echo "HailoRT installation complete: 4.x at /opt/hailort4, 5.x at /opt/hailort5"
