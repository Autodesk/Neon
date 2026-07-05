#!/bin/bash
# Create /etc/passwd and /etc/group entries for the host user, then drop privileges.
# Used by build-run-docker-multi.sh so bind-mounted files are writable and the shell
# shows the correct username (avoids "I have no name!" and unknown group warnings).
set -euo pipefail

HOST_UID="${HOST_UID:?HOST_UID is required}"
HOST_GID="${HOST_GID:?HOST_GID is required}"
HOST_USER="${HOST_USER:-host}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"

if ! getent group "${HOST_GID}" >/dev/null; then
    groupadd -o -g "${HOST_GID}" "${HOST_USER}" 2>/dev/null || \
        groupadd -g "${HOST_GID}" "gid${HOST_GID}"
fi
GROUP_NAME="$(getent group "${HOST_GID}" | cut -d: -f1)"

if ! getent passwd "${HOST_UID}" >/dev/null; then
    useradd -o -u "${HOST_UID}" -g "${GROUP_NAME}" -d "${CONTAINER_WORKDIR}" -s /bin/bash -M "${HOST_USER}" 2>/dev/null || \
        useradd -u "${HOST_UID}" -g "${GROUP_NAME}" -d "${CONTAINER_WORKDIR}" -s /bin/bash -M "${HOST_USER}"
fi

# Drop to the host user with a proper interactive shell (avoid su -c nesting,
# which breaks job control and prints "no job control in this shell").
cd "${CONTAINER_WORKDIR}"
if command -v runuser >/dev/null 2>&1; then
    exec runuser -u "${HOST_USER}" -w "${CONTAINER_WORKDIR}" -- /bin/bash -l
fi
exec su - "${HOST_USER}" -w "${CONTAINER_WORKDIR}"
