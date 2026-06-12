"""Auxiliary CLI commands (``profiles``, ``configure``, ``doctor``, ``web``).

Kept separate from the large ``record`` command in ``main.py``. Registered onto
the shared Typer ``app`` via :func:`register`.
"""

from pathlib import Path
from typing import Optional

import typer

from .config_profiles import ProfileManager, ProfileType, SystemCapabilityDetector
from .doctor import run_doctor
from .user_config import (
    DEFAULT_CONFIG,
    get_config_path,
    load_user_config,
    save_user_config,
)


def doctor(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file to validate (defaults to ~/.config/spotify_splitter/config.json)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print machine-readable JSON.",
    ),
) -> None:
    """Check whether the system is ready to record Spotify playback."""
    import json
    from rich.console import Console
    from rich.table import Table

    report = run_doctor(config_path)

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        console = Console()
        color = "green" if report.ok else "red"
        console.print(f"[bold {color}]{report.summary}[/bold {color}]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Status", no_wrap=True)
        table.add_column("Check")
        table.add_column("Result")
        table.add_column("Action")

        icons = {"ok": "✓", "warning": "!", "error": "✗"}
        styles = {"ok": "green", "warning": "yellow", "error": "red"}
        for check in report.checks:
            table.add_row(
                f"[{styles[check.status]}]{icons[check.status]}[/{styles[check.status]}]",
                check.label,
                check.message,
                check.action or "",
            )
        console.print(table)

    if not report.ok:
        raise typer.Exit(code=1)


def web(
    ctx: typer.Context,
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file for the web UI to manage.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host/IP for the web UI. Use 0.0.0.0 only on a trusted network.",
    ),
    port: int = typer.Option(
        8730,
        "--port",
        help="Port for the web UI.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the web UI in the default browser after startup.",
    ),
) -> None:
    """Start the local web UI."""
    from spoti2_service.service_app import run_service

    resolved_config = config_path or (ctx.obj.get("config_path") if ctx.obj else None)
    verbose = bool(ctx.obj.get("verbose")) if ctx.obj else False
    run_service(
        host=host,
        port=port,
        config=resolved_config,
        verbose=verbose,
        open_browser=open_browser,
    )


def profiles():
    """List available configuration profiles and system capabilities."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Display system capabilities
    console.print("\n[bold cyan]System Capabilities Detection[/bold cyan]")
    try:
        capabilities = SystemCapabilityDetector.detect_capabilities()

        cap_table = Table(show_header=False, box=None, padding=(0, 1))
        cap_table.add_column("Property", style="cyan")
        cap_table.add_column("Value", style="white")

        cap_table.add_row("CPU Cores:", str(capabilities.cpu_cores))
        cap_table.add_row("Memory:", f"{capabilities.memory_gb:.1f} GB")
        cap_table.add_row("System Load:", f"{capabilities.system_load:.1%}")
        cap_table.add_row("Headless Mode:", "Yes" if capabilities.is_headless else "No")
        cap_table.add_row("Audio Backend:", capabilities.audio_backend)
        cap_table.add_row("GUI Available:", "Yes" if capabilities.has_gui else "No")

        console.print(cap_table)

        # Show recommended profile
        recommended_profile = ProfileManager.select_optimal_profile(capabilities)
        console.print(f"\n[bold green]Recommended Profile:[/bold green] {recommended_profile.name}")
        console.print(f"[dim]{recommended_profile.description}[/dim]")

    except Exception as e:
        console.print(f"[red]Error detecting system capabilities: {e}[/red]")

    # Display available profiles
    console.print("\n[bold cyan]Available Configuration Profiles[/bold cyan]")

    profiles_table = Table()
    profiles_table.add_column("Profile", style="cyan", no_wrap=True)
    profiles_table.add_column("Description", style="white")
    profiles_table.add_column("Queue Size", style="green")
    profiles_table.add_column("Latency", style="magenta")

    for profile_type in ProfileType:
        if profile_type == ProfileType.AUTO:
            continue

        try:
            profile = ProfileManager.get_profile(profile_type)
            profiles_table.add_row(
                profile.name,
                profile.description,
                str(profile.queue_size),
                f"{profile.latency*1000:.0f}ms"
            )
        except Exception as e:
            profiles_table.add_row(
                profile_type.value,
                f"Error loading profile: {e}",
                "-", "-"
            )

    # Add auto profile
    profiles_table.add_row(
        "auto",
        "Automatically select optimal profile based on system capabilities",
        "varies",
        "varies"
    )

    console.print(profiles_table)

    console.print("\n[bold cyan]Usage Examples[/bold cyan]")
    console.print("  spotify-splitter record --profile headless")
    console.print("  spotify-splitter record --profile desktop")
    console.print("  spotify-splitter record --profile high_performance")


def configure(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Where to write the configuration file (defaults to ~/.config/spotify_splitter/config.json)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Default directory for recordings",
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Preferred output format",
    ),
    player: Optional[str] = typer.Option(
        None,
        "--player",
        "-p",
        help="Default MPRIS player name",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Default configuration profile to apply",
    ),
    playlist: Optional[str] = typer.Option(
        None,
        "--playlist",
        help="Default M3U playlist to update",
    ),
    bundle_playlist: Optional[bool] = typer.Option(
        None,
        "--bundle-playlist/--no-bundle-playlist",
        help="Bundle playlist tracks into a compilation album by default",
    ),
    queue_size: Optional[int] = typer.Option(
        None,
        "--queue-size",
        help="Default audio buffer queue size",
    ),
    blocksize: Optional[int] = typer.Option(
        None,
        "--blocksize",
        help="Default PortAudio blocksize",
    ),
    latency: Optional[float] = typer.Option(
        None,
        "--latency",
        help="Default latency hint for the audio stream",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Do not prompt; rely solely on provided flags",
    ),
):
    """Create or update a saved configuration for simplified usage."""
    config_file = get_config_path(config_path)
    existing_file = config_file.exists()
    existing = load_user_config(config_path)
    prompt_user = not non_interactive

    def prompt_text(
        key: str,
        provided: Optional[str],
        prompt_message: str,
        allow_empty: bool = False,
    ) -> Optional[str]:
        if provided is not None:
            return provided
        if not prompt_user:
            return existing.get(key)
        default_value = existing.get(key) or DEFAULT_CONFIG.get(key) or ""
        value = typer.prompt(prompt_message, default=default_value)
        if not value and not allow_empty:
            return default_value
        return value if value else None

    def prompt_bool(
        key: str,
        provided: Optional[bool],
        prompt_message: str,
    ) -> bool:
        if provided is not None:
            return provided
        if not prompt_user:
            return bool(existing.get(key, DEFAULT_CONFIG.get(key, False)))
        default_value = bool(existing.get(key, DEFAULT_CONFIG.get(key, False)))
        return typer.confirm(prompt_message, default=default_value)

    updates = {}

    output_value = prompt_text(
        "output",
        output,
        "Where should recordings be saved?",
    )
    if not output_value:
        output_value = existing.get("output") or DEFAULT_CONFIG["output"]
    updates["output"] = str(Path(output_value).expanduser())

    format_value = prompt_text(
        "format",
        format,
        "Preferred audio format",
    )
    updates["format"] = format_value or existing.get("format") or DEFAULT_CONFIG["format"]

    player_value = prompt_text(
        "player",
        player,
        "Default MPRIS player name",
    )
    updates["player"] = player_value or existing.get("player") or DEFAULT_CONFIG["player"]

    profile_value = prompt_text(
        "profile",
        profile,
        "Default profile (auto/headless/desktop/high_performance)",
    )
    updates["profile"] = profile_value or existing.get("profile") or DEFAULT_CONFIG["profile"]

    playlist_value = prompt_text(
        "playlist",
        playlist,
        "Playlist file to update (enter to skip)",
        allow_empty=True,
    )
    updates["playlist"] = (
        str(Path(playlist_value).expanduser()) if playlist_value else None
    )
    updates["bundle_playlist"] = prompt_bool(
        "bundle_playlist",
        bundle_playlist,
        "Bundle playlist tracks into a compilation album by default?",
    )

    if queue_size is not None:
        updates["queue_size"] = queue_size
    if blocksize is not None:
        updates["blocksize"] = blocksize
    if latency is not None:
        updates["latency"] = latency

    merged = existing.copy()
    merged.update(updates)

    saved_path = save_user_config(merged, config_path)

    typer.echo(f"Configuration saved to {saved_path}")
    typer.echo("Defaults applied to 'spotify-splitter record':")
    summary_keys = [
        "output",
        "format",
        "player",
        "profile",
        "playlist",
        "bundle_playlist",
        "lastfm_api_key",
    ]
    for key in summary_keys:
        typer.echo(f"  {key}: {merged.get(key)}")

    if not existing_file:
        typer.echo(
            "\nTip: re-run 'spotify-splitter configure' any time you want to update these defaults."
        )


def register(app: typer.Typer) -> None:
    """Register the auxiliary commands onto the shared Typer app."""
    app.command()(doctor)
    app.command()(web)
    app.command()(profiles)
    app.command()(configure)
