# Docker Troubleshooting Guide

## Permission Denied Error

If you encounter the error:
```
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

This means your user doesn't have permission to access the Docker daemon. Here are solutions:

### Solution 1: Add User to Docker Group (Recommended)

1. **Add your user to the docker group:**
   ```bash
   sudo usermod -aG docker $USER
   ```

2. **Log out and log back in** (or restart your terminal session) for the changes to take effect.

3. **Verify the fix:**
   ```bash
   groups  # Should show 'docker' in the list
   docker ps  # Should work without sudo
   ```

### Solution 2: Use sudo (Not Recommended)

If you can't add yourself to the docker group, you can use sudo, but this is not recommended for security reasons:

```bash
sudo ./build-run-docker.sh
```

**Note:** Using sudo may cause file permission issues with mounted volumes.

### Solution 3: Fix Docker Socket Permissions (Alternative)

If the docker group doesn't exist or you can't use it:

```bash
# Check if docker group exists
getent group docker

# If it doesn't exist, create it
sudo groupadd docker

# Add your user to the group
sudo usermod -aG docker $USER

# Change socket permissions (temporary fix)
sudo chmod 666 /var/run/docker.sock
```

**Warning:** Changing socket permissions to 666 is a security risk and should only be used temporarily.

## BuildKit / buildx Error

If you see:
```
ERROR: BuildKit is enabled but the buildx component is missing or broken.
```

**Option A (automatic):** The `build-run-docker.sh` script now detects when buildx is missing and falls back to the legacy builder, so you can simply run the script again—it will build without BuildKit.

**Option B (install buildx for faster builds):** Install the Docker buildx plugin: https://docs.docker.com/go/buildx/  
On many systems it’s included with Docker Desktop or the `docker-buildx-plugin` package (e.g. `sudo apt install docker-buildx-plugin` on Debian/Ubuntu).

## BuildKit Deprecation Warning

If you see:
```
DEPRECATED: The legacy builder is deprecated and will be removed in a future release.
```

This is just a warning. To use BuildKit (recommended):

1. **Enable BuildKit:**
   ```bash
   export DOCKER_BUILDKIT=1
   ```

2. **Or add to your `~/.bashrc` or `~/.zshrc`:**
   ```bash
   echo 'export DOCKER_BUILDKIT=1' >> ~/.zshrc
   source ~/.zshrc
   ```

## Docker Not Running

If Docker isn't running, start it:

```bash
# Check Docker status
sudo systemctl status docker

# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker
```

## Verify Docker Installation

To verify Docker is properly installed and configured:

```bash
# Check Docker version
docker --version

# Check Docker daemon is running
docker info

# Test with a simple command
docker run hello-world
```

If all these commands work, Docker is properly configured.




