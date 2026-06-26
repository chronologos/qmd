#!/usr/bin/env python3
"""
QMD server deployment script.
Deploys both vLLM (query expansion) and embed/rerank services.

Usage:
    sudo python3 deploy.py              # Deploy all services
    sudo python3 deploy.py --status     # Show status of all services
    sudo python3 deploy.py --logs       # Show embed/rerank logs
    sudo python3 deploy.py --restart    # Restart all services
    sudo python3 deploy.py --stop       # Stop all services

    # Individual service control
    sudo python3 deploy.py vllm --status
    sudo python3 deploy.py vllm --logs
    sudo python3 deploy.py vllm --restart
    sudo python3 deploy.py vllm --model "Qwen/Qwen3-4B"

    sudo python3 deploy.py embed --status
    sudo python3 deploy.py embed --logs
    sudo python3 deploy.py embed --restart

Prerequisites:
    1. Create Tailscale Services in admin console:
       https://login.tailscale.com/admin/services
       - "qmd-embed" with endpoint tcp:443
       - "qmd-vllm" with endpoint tcp:443
    2. Docker installed and running
    3. Virtual environment set up: uv venv && uv pip install -r requirements.txt
"""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

DEPLOY_DIR = Path(__file__).parent.resolve()
VENV_DIR = DEPLOY_DIR / ".venv"

# Embed/Rerank service config
EMBED_SERVICE_NAME = "qmd-embed"
EMBED_SERVICE_FILE = Path(f"/etc/systemd/system/{EMBED_SERVICE_NAME}.service")
EMBED_PORT = 8001

# vLLM container config
VLLM_CONTAINER_NAME = "qmd-vllm"
VLLM_IMAGE = "nvcr.io/nvidia/vllm:25.12.post1-py3"
VLLM_PORT = 8000
VLLM_DEFAULT_MODEL = "Qwen/Qwen3-4B"
VLLM_GPU_MEMORY_UTILIZATION = 0.3  # Leave room for embed/rerank on same GPU


# =============================================================================
# Utilities
# =============================================================================

def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def require_root() -> None:
    """Exit if not running as root."""
    if os.geteuid() != 0:
        print("Error: This command must be run with sudo")
        sys.exit(1)


def get_tailnet_suffix() -> str:
    """Get the tailnet suffix from tailscale status."""
    result = run(["tailscale", "status", "--json"], capture=True)
    status = json.loads(result.stdout)
    dns_name = status.get("Self", {}).get("DNSName", "").rstrip(".")
    parts = dns_name.split(".")
    if len(parts) >= 3:
        return "." + ".".join(parts[1:])
    return ""


def setup_tailscale_serve(service_name: str, port: int) -> str:
    """Configure Tailscale Serve for a service. Returns the full service URL."""
    print(f"\n[{service_name}] Configuring Tailscale Serve...")
    run([
        "tailscale", "serve",
        "--service", f"svc:{service_name}",
        "--bg", "--https=443",
        f"127.0.0.1:{port}"
    ])
    print(f"  - Tailscale Serve configured ({service_name} service)")
    return f"{service_name}{get_tailnet_suffix()}"


# =============================================================================
# vLLM Docker Container Management
# =============================================================================

def check_docker() -> None:
    """Verify docker is available."""
    result = run(["docker", "info"], check=False, capture=True)
    if result.returncode != 0:
        print("Error: Docker is not running or not accessible")
        sys.exit(1)
    print("  - Docker: OK")


def get_vllm_container_info() -> dict | None:
    """Get info about the vLLM container if it exists."""
    result = run(
        ["docker", "inspect", VLLM_CONTAINER_NAME],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)[0]


def get_current_vllm_model() -> str | None:
    """Get the model currently being served by the vLLM container."""
    info = get_vllm_container_info()
    if not info:
        return None
    # Parse the command to find the model argument
    cmd = info.get("Config", {}).get("Cmd", [])
    for i, arg in enumerate(cmd):
        if arg == "serve" and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def deploy_vllm(model: str | None = None, force: bool = False) -> None:
    """Deploy the vLLM container."""
    model = model or VLLM_DEFAULT_MODEL
    print(f"\n[vLLM] Deploying container with model: {model}")

    current_model = get_current_vllm_model()
    container_exists = current_model is not None

    if container_exists and current_model == model and not force:
        print(f"  - Container already running with {model}")
        # Just ensure it's started
        run(["docker", "start", VLLM_CONTAINER_NAME], check=False)
        return

    # Stop and remove existing container if model changed or force
    if container_exists:
        if current_model != model:
            print(f"  - Model changed: {current_model} -> {model}")
        print("  - Removing existing container...")
        run(["docker", "stop", VLLM_CONTAINER_NAME], check=False)
        run(["docker", "rm", VLLM_CONTAINER_NAME], check=False)

    # Create new container
    print("  - Creating new container...")
    run([
        "docker", "run", "-d",
        "--name", VLLM_CONTAINER_NAME,
        "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-e", "NVIDIA_DISABLE_REQUIRE=1",  # Enable CUDA forward compatibility
        "-p", f"127.0.0.1:{VLLM_PORT}:{VLLM_PORT}",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        VLLM_IMAGE,
        "vllm", "serve", model,
        "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
        "--port", str(VLLM_PORT),
    ])
    print(f"  - Container '{VLLM_CONTAINER_NAME}' created and started")


def stop_vllm() -> None:
    """Stop the vLLM container."""
    print("\n[vLLM] Stopping container...")
    run(["docker", "stop", VLLM_CONTAINER_NAME], check=False)
    print("  - Container stopped")


def show_vllm_status() -> None:
    """Show vLLM container status."""
    print("\n[vLLM Container Status]")
    info = get_vllm_container_info()
    if not info:
        print("  Container does not exist")
        return

    state = info.get("State", {})
    config = info.get("Config", {})

    print(f"  Name: {VLLM_CONTAINER_NAME}")
    print(f"  Status: {state.get('Status', 'unknown')}")
    print(f"  Running: {state.get('Running', False)}")
    print(f"  Model: {get_current_vllm_model()}")
    print(f"  Image: {config.get('Image', 'unknown')}")

    if state.get("Running"):
        # Check if the API is responding
        result = run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             f"http://127.0.0.1:{VLLM_PORT}/health"],
            check=False, capture=True
        )
        api_status = "healthy" if result.stdout.strip() == "200" else "starting/unhealthy"
        print(f"  API: {api_status}")


def show_vllm_logs(follow: bool = True) -> None:
    """Show vLLM container logs."""
    cmd = ["docker", "logs", VLLM_CONTAINER_NAME]
    if follow:
        cmd.append("-f")
    run(cmd, check=False)


def setup_vllm_tailscale() -> str:
    """Configure Tailscale Serve for vLLM."""
    return setup_tailscale_serve(VLLM_CONTAINER_NAME, VLLM_PORT)


# =============================================================================
# Embed/Rerank Service Management
# =============================================================================

def check_embed_prerequisites() -> None:
    """Verify embed/rerank prerequisites."""
    # Check virtual environment exists
    if not (VENV_DIR / "bin" / "uvicorn").exists():
        print(f"Error: Virtual environment not found at {VENV_DIR}")
        print("Run: uv venv && uv pip install -r requirements.txt")
        sys.exit(1)
    print(f"  - Virtual environment: OK ({VENV_DIR})")

    # Check embed_rerank.py exists
    if not (DEPLOY_DIR / "embed_rerank.py").exists():
        print(f"Error: embed_rerank.py not found in {DEPLOY_DIR}")
        sys.exit(1)
    print("  - embed_rerank.py: OK")


def create_embed_systemd_service() -> None:
    """Create the systemd service file for embed/rerank."""
    print("\n[Embed] Creating systemd service...")

    sudo_user = os.environ.get("SUDO_USER", "root")

    service_content = f"""[Unit]
Description=QMD Embed/Rerank Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=5
User={sudo_user}
Group={sudo_user}
WorkingDirectory={DEPLOY_DIR}
Environment="PATH={VENV_DIR}/bin:/usr/local/bin:/usr/bin:/bin"
Environment="MAX_BATCH_SIZE=64"

# Start uvicorn
ExecStart={VENV_DIR}/bin/uvicorn embed_rerank:app --host 127.0.0.1 --port {EMBED_PORT}

[Install]
WantedBy=multi-user.target
"""

    with open(EMBED_SERVICE_FILE, "w") as f:
        f.write(service_content)
    print(f"  - Service file written to {EMBED_SERVICE_FILE}")

    run(["systemctl", "daemon-reload"])
    print("  - systemd daemon reloaded")


def deploy_embed() -> None:
    """Deploy the embed/rerank service."""
    print("\n[Embed] Deploying service...")
    run(["systemctl", "enable", EMBED_SERVICE_NAME])
    print(f"  - Service '{EMBED_SERVICE_NAME}' enabled")

    run(["systemctl", "restart", EMBED_SERVICE_NAME])
    print(f"  - Service '{EMBED_SERVICE_NAME}' started")


def stop_embed() -> None:
    """Stop the embed/rerank service."""
    print("\n[Embed] Stopping service...")
    run(["systemctl", "stop", EMBED_SERVICE_NAME])
    print("  - Service stopped")


def show_embed_status() -> None:
    """Show embed/rerank service status."""
    print("\n[Embed/Rerank Service Status]")
    run(["systemctl", "status", EMBED_SERVICE_NAME, "--no-pager"], check=False)


def show_embed_logs(follow: bool = True) -> None:
    """Show embed/rerank service logs."""
    cmd = ["journalctl", "-u", EMBED_SERVICE_NAME, "--no-pager"]
    if follow:
        cmd.append("-f")
    run(cmd, check=False)


def setup_embed_tailscale() -> str:
    """Configure Tailscale Serve for embed/rerank."""
    return setup_tailscale_serve(EMBED_SERVICE_NAME, EMBED_PORT)


# =============================================================================
# Combined Operations
# =============================================================================

def check_all_prerequisites() -> None:
    """Check all prerequisites."""
    print("\n[Prerequisites]")

    require_root()
    print("  - Running as root: OK")

    # Check Tailscale
    result = run(["tailscale", "status"], check=False, capture=True)
    if result.returncode != 0:
        print("Error: Tailscale is not connected")
        sys.exit(1)
    print("  - Tailscale: OK")

    # Check Docker
    check_docker()

    # Check embed prerequisites
    check_embed_prerequisites()


def deploy_all(vllm_model: str | None = None) -> None:
    """Deploy all services."""
    check_all_prerequisites()

    # Deploy vLLM
    deploy_vllm(model=vllm_model)
    vllm_url = setup_vllm_tailscale()

    # Deploy embed/rerank
    create_embed_systemd_service()
    deploy_embed()
    embed_url = setup_embed_tailscale()

    # Print summary
    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)
    print(f"\nvLLM (query expansion): https://{vllm_url}")
    print(f"Embed/Rerank:           https://{embed_url}")
    print("\nTest endpoints:")
    print(f"  curl https://{vllm_url}/health")
    print(f"  curl https://{embed_url}/health")
    print("\nUseful commands:")
    print(f"  sudo python3 {__file__} --status   # Check all services")
    print(f"  sudo python3 {__file__} vllm --logs    # vLLM logs")
    print(f"  sudo python3 {__file__} embed --logs   # Embed logs")


def show_all_status() -> None:
    """Show status of all services."""
    show_vllm_status()
    show_embed_status()


def stop_all() -> None:
    """Stop all services."""
    stop_vllm()
    stop_embed()


def restart_all(vllm_model: str | None = None) -> None:
    """Restart all services."""
    print("Restarting all services...")
    deploy_vllm(model=vllm_model, force=True)
    run(["systemctl", "restart", EMBED_SERVICE_NAME])
    print("All services restarted")


# =============================================================================
# CLI
# =============================================================================

def add_service_args(parser: argparse.ArgumentParser) -> None:
    """Add common service management arguments to a parser."""
    parser.add_argument("--status", action="store_true", help="Show service status")
    parser.add_argument("--logs", action="store_true", help="Show service logs")
    parser.add_argument("--restart", action="store_true", help="Restart service")
    parser.add_argument("--stop", action="store_true", help="Stop service")


def restart_embed() -> None:
    """Restart the embed/rerank service."""
    require_root()
    run(["systemctl", "restart", EMBED_SERVICE_NAME])
    print("Embed service restarted")


def handle_service_command(
    args: argparse.Namespace,
    show_status: Callable[[], None],
    show_logs: Callable[[], None],
    stop: Callable[[], None],
    restart: Callable[[], None],
) -> None:
    """Handle common service subcommand dispatch."""
    if args.status:
        show_status()
    elif args.logs:
        show_logs()
    elif args.stop:
        stop()
    elif args.restart:
        restart()
    else:
        show_status()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy QMD server (vLLM + embed/rerank)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_service_args(parser)
    parser.add_argument("--model", type=str, help=f"vLLM model (default: {VLLM_DEFAULT_MODEL})")

    subparsers = parser.add_subparsers(dest="service", help="Individual service commands")

    vllm_parser = subparsers.add_parser("vllm", help="vLLM container management")
    add_service_args(vllm_parser)
    vllm_parser.add_argument("--model", type=str, help="Model to serve")

    embed_parser = subparsers.add_parser("embed", help="Embed/rerank service management")
    add_service_args(embed_parser)

    args = parser.parse_args()

    if args.service == "vllm":
        # vLLM has special handling for --model flag
        if args.restart or args.model:
            require_root()
            deploy_vllm(model=args.model, force=args.restart)
        else:
            handle_service_command(args, show_vllm_status, show_vllm_logs, stop_vllm, lambda: None)
        return

    if args.service == "embed":
        handle_service_command(args, show_embed_status, show_embed_logs, stop_embed, restart_embed)
        return

    # Global commands (no subcommand specified)
    if args.status:
        show_all_status()
    elif args.logs:
        show_embed_logs()
    elif args.stop:
        stop_all()
    elif args.restart:
        require_root()
        restart_all(vllm_model=args.model)
    else:
        deploy_all(vllm_model=args.model)


if __name__ == "__main__":
    main()
