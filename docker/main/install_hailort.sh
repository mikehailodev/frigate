#!/bin/bash

set -euxo pipefail

# HailoRT 4.x for Hailo-8/8L
HAILO_VERSION="4.23.0"

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

ARTIFACTS_DIR="/deps/hailort_artifacts"

HAILO_DEB="hailort_${HAILO_VERSION}_${deb_arch}.deb"
HAILO_WHL="hailort-${HAILO_VERSION}-cp311-cp311-linux_${whl_arch}.whl"

# Extract shared library from .deb into /rootfs staging area
mkdir -p /rootfs/usr/local/lib
tmpdir=$(mktemp -d)
dpkg-deb -x "${ARTIFACTS_DIR}/${HAILO_DEB}" "${tmpdir}"
cp "${tmpdir}"/usr/lib/libhailort.so.${HAILO_VERSION} /rootfs/usr/local/lib/
ln -sf libhailort.so.${HAILO_VERSION} /rootfs/usr/local/lib/libhailort.so.4
rm -rf "${tmpdir}"

# Copy Python wheel to /wheels for later pip install
cp "${ARTIFACTS_DIR}/${HAILO_WHL}" /wheels/

echo "HailoRT ${HAILO_VERSION} installation complete"
