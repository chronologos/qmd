#!/usr/bin/env python3
"""
QMD Embed/Rerank server deployment script.
Deploys the embed/rerank service managed by systemd, accessible via Tailscale.

Usage:
    sudo python3 deploy.py              # Initial deploy
    sudo python3 deploy.py --restart    # Restart without rebuilding
    sudo python3 deploy.py --stop       # Stop the service
    sudo python3 deploy.py --status     # Show service status
    sudo python3 deploy.py --logs       # Show service logs

Prerequisites:
    1. Create Tailscale Service "qmd-embed" in admin console:
       https://login.tailscale.com/admin/services
    2. Set endpoint to tcp:443
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Configuration
DEPLOY_DIR = Path(__file__).parent.resolve()
VENV_DIR = DEPLOY_DIR / ".venv"
SERVICE_NAME = "qmd-embed"
SERVICE_FILE = Path(f"/etc/systemd/system/{SERVICE_NAME}.service")
LOCAL_PORT = 8001


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def check_prerequisites() -> None:
    """Verify all prerequisites are met."""
    print("\n[1/5] Checking prerequisites...")

    # Check running as root (needed for systemd)
    if os.geteuid() != 0:
        print("Error: This script must be run with sudo")
        sys.exit(1)
    print("  - Running as root: OK")

    # Check Tailscale
    result = run(["tailscale", "status"], check=False, capture=True)
    if result.returncode != 0:
        print("Error: Tailscale is not connected")
        sys.exit(1)
    print("  - Tailscale: OK")

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


def get_tailscale_ip() -> str:
    """Get the Tailscale IPv4 address."""
    print("\n[2/5] Getting Tailscale IP...")
    result = run(["tailscale", "ip", "-4"], capture=True)
    ip = result.stdout.strip()
    print(f"  - Tailscale IP: {ip}")
    return ip


def create_systemd_service() -> None:
    """Create the systemd service file."""
    print("\n[3/5] Creating systemd service...")

    # Get the real user who ran sudo (for HOME directory)
    sudo_user = os.environ.get("SUDO_USER", "root")
    sudo_uid = os.environ.get("SUDO_UID", "0")
    sudo_gid = os.environ.get("SUDO_GID", "0")

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

# Start uvicorn
ExecStart={VENV_DIR}/bin/uvicorn embed_rerank:app --host 127.0.0.1 --port {LOCAL_PORT}

[Install]
WantedBy=multi-user.target
"""

    with open(SERVICE_FILE, "w") as f:
        f.write(service_content)
    print(f"  - Service file written to {SERVICE_FILE}")

    # Reload systemd
    run(["systemctl", "daemon-reload"])
    print("  - systemd daemon reloaded")


def enable_and_start() -> None:
    """Enable and start the systemd service."""
    print("\n[4/5] Enabling and starting service...")

    run(["systemctl", "enable", SERVICE_NAME])
    print(f"  - Service '{SERVICE_NAME}' enabled")

    run(["systemctl", "restart", SERVICE_NAME])
    print(f"  - Service '{SERVICE_NAME}' started")


def setup_tailscale_serve() -> str:
    """Configure Tailscale Serve for HTTPS termination.

    Prerequisites:
    - Create a Tailscale Service named "qmd-embed" in admin console
    - Set endpoint to tcp:443
    """
    print("\n[5/5] Configuring Tailscale Serve...")

    # Set up tailscale serve with dedicated service hostname
    run(["tailscale", "serve", "--service", f"svc:{SERVICE_NAME}", "--bg", "--https=443", f"127.0.0.1:{LOCAL_PORT}"])
    print(f"  - Tailscale Serve configured ({SERVICE_NAME} service, HTTPS:443 -> localhost:{LOCAL_PORT})")

    # Get the tailnet name from tailscale status
    result = run(["tailscale", "status", "--json"], capture=True)
    status = json.loads(result.stdout)
    dns_name = status.get("Self", {}).get("DNSName", "").rstrip(".")
    # Extract tailnet from machine DNS name and construct service URL
    parts = dns_name.split(".")
    if len(parts) >= 3:
        tailnet = ".".join(parts[1:])
        return f"{SERVICE_NAME}.{tailnet}"
    return dns_name


def show_status() -> None:
    """Show the service status."""
    run(["systemctl", "status", SERVICE_NAME, "--no-pager"], check=False)


def show_logs() -> None:
    """Show service logs."""
    run(["journalctl", "-u", SERVICE_NAME, "-f", "--no-pager"], check=False)


def stop_service() -> None:
    """Stop the service."""
    print("Stopping service...")
    run(["systemctl", "stop", SERVICE_NAME])
    print("Service stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy QMD Embed/Rerank server")
    parser.add_argument("--restart", action="store_true", help="Restart the service")
    parser.add_argument("--stop", action="store_true", help="Stop the service")
    parser.add_argument("--status", action="store_true", help="Show service status")
    parser.add_argument("--logs", action="store_true", help="Show service logs")
    args = parser.parse_args()

    # Handle simple commands
    if args.status:
        show_status()
        return
    if args.logs:
        show_logs()
        return
    if args.stop:
        stop_service()
        return

    # Full deploy or restart
    check_prerequisites()
    get_tailscale_ip()
    create_systemd_service()
    enable_and_start()
    dns_name = setup_tailscale_serve()

    print("\n" + "=" * 50)
    print("Deployment complete!")
    print("=" * 50)
    print(f"\nQMD Embed/Rerank server is now accessible at: https://{dns_name}")
    print("\nTest endpoints:")
    print(f"  curl https://{dns_name}/health")
    print(f"  curl -X POST https://{dns_name}/v1/embeddings -H 'Content-Type: application/json' -d '{{\"input\": [\"test\"]}}'")
    print("\nUseful commands:")
    print(f"  sudo python3 {__file__} --status   # Check status")
    print(f"  sudo python3 {__file__} --logs     # View logs")
    print(f"  sudo python3 {__file__} --restart  # Restart service")
    print(f"  tailscale serve status             # Check Tailscale Serve config")


if __name__ == "__main__":
    main()
