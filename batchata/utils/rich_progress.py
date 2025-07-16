"""Rich progress display for batch runs."""

import threading
from datetime import datetime
from typing import Dict, Optional

from rich.console import Console
from rich.live import Live
from rich.tree import Tree
from rich.text import Text


class RichBatchProgressDisplay:
    """Rich-based progress display for batch runs."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the rich progress display.
        
        Args:
            console: Rich console instance, creates new if None
        """
        self.console = console or Console()
        self.live: Optional[Live] = None
        self.batches: Dict[str, Dict] = {}
        self.overall_stats: Dict = {}
        self.config: Dict = {}
        self.start_time: Optional[datetime] = None
        self.last_update: Optional[datetime] = None
        self._lock = threading.Lock()
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index = 0
    
    def start(self, stats: Dict, config: Dict):
        """Start the live progress display.
        
        Args:
            stats: Initial batch statistics
            config: Batch configuration
        """
        with self._lock:
            self.overall_stats = stats
            self.config = config
            self.start_time = datetime.now()
            self.last_update = self.start_time
            
            # Create live display
            self.live = Live(
                self._create_display(),
                console=self.console,
                refresh_per_second=10,
                auto_refresh=True
            )
            self.live.start()
    
    def update(self, stats: Dict, batch_data: Dict, elapsed_time: float):
        """Update the progress display.
        
        Args:
            stats: Current batch statistics
            batch_data: Dictionary of batch information
            elapsed_time: Elapsed time in seconds
        """
        with self._lock:
            self.overall_stats = stats
            self.batches = batch_data
            self.last_update = datetime.now()
            
            # Advance spinner
            self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
            
            # Update live display
            if self.live:
                self.live.update(self._create_display())
    
    def stop(self):
        """Stop the live progress display."""
        with self._lock:
            if self.live:
                self.live.stop()
                self.live = None
    
    def _create_display(self) -> Tree:
        """Create the rich display tree."""
        # Overall statistics
        stats = self.overall_stats
        running_count = len([b for b in self.batches.values() if b.get('status') == 'running'])
        failed_count = stats.get('failed', 0)
        cancelled_count = len([b for b in self.batches.values() if b.get('status') == 'cancelled'])
        
        # Main tree
        header_parts = [
            f"Batches: {stats.get('batches_completed', 0)}/{stats.get('batches_total', 0)}",
            f"Requests: {stats.get('completed', 0)}/{stats.get('total', 0)}"
        ]
        
        if running_count > 0:
            header_parts.append(f"Running: {running_count}")
        if failed_count > 0:
            header_parts.append(f"Failed: {failed_count}")
        if cancelled_count > 0:
            header_parts.append(f"[yellow]Cancelled: {cancelled_count}[/yellow]")
            
        tree = Tree(f"[bold]{' '.join(header_parts)}[/bold]")
        
        # Add batch information
        if self.batches:
            batch_ids = sorted(self.batches.keys())
            num_batches = len(batch_ids)
            
            # Show initializing message if no batches have started yet
            if num_batches > 0 and all(b.get('status') == 'pending' and not b.get('start_time') for b in self.batches.values()):
                tree.add("[dim italic]Initializing batch requests...[/dim italic]")
            
            for idx, batch_id in enumerate(batch_ids):
                batch_info = self.batches[batch_id]
                status = batch_info.get('status', 'pending')
                completed = batch_info.get('completed', 0)
                total = batch_info.get('total', 1)
                cost = batch_info.get('cost', 0.0)
                
                # Determine tree symbol
                is_last = idx == num_batches - 1
                tree_symbol = "└─" if is_last else "├─"
                
                # Format progress bar with better styling
                progress_pct = (completed / total) if total > 0 else 0
                filled_width = int(progress_pct * 25)
                
                if status == 'complete':
                    bar = "[bold green]" + "━" * 25 + "[/bold green]"
                elif status == 'failed':
                    bar = "[bold red]" + "━" * 25 + "[/bold red]"
                elif status == 'cancelled':
                    bar = "[bold yellow]" + "━" * filled_width + "[/bold yellow]"
                    if filled_width < 25:
                        bar += "[dim yellow]" + "━" * (25 - filled_width) + "[/dim yellow]"
                elif status == 'running':
                    bar = "[bold blue]" + "━" * filled_width + "[/bold blue]"
                    if filled_width < 25:
                        bar += "[blue]╸[/blue]" + "[dim white]" + "━" * (24 - filled_width) + "[/dim white]"
                else:
                    bar = "[dim white]" + "━" * 25 + "[/dim white]"
                
                # Format status with better colors and fixed width
                if status == 'complete':
                    status_text = "[bold green]Ended[/bold green]  "
                elif status == 'failed':
                    status_text = "[bold red]Failed[/bold red] "
                elif status == 'cancelled':
                    status_text = "[bold yellow]Cancelled[/bold yellow]"
                elif status == 'running':
                    spinner = self._spinner_frames[self._spinner_index]
                    status_text = f"[bold blue]{spinner} Running[/bold blue]"
                else:
                    status_text = "[dim]Pending[/dim]"
                
                # Calculate elapsed time
                start_time = batch_info.get('start_time')
                if start_time and status in ['running', 'complete', 'failed', 'cancelled']:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    elapsed_hours = int(elapsed // 3600)
                    elapsed_minutes = int((elapsed % 3600) // 60)
                    elapsed_seconds = int(elapsed % 60)
                    
                    if elapsed_hours > 0:
                        time_str = f"{elapsed_hours}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"
                    else:
                        time_str = f"{elapsed_minutes:02d}:{elapsed_seconds:02d}"
                else:
                    time_str = "-:--:--"
                
                # Format percentage
                percentage = int(progress_pct * 100)
                
                # Get output filenames if completed
                output_file = ""
                if status == 'complete' and self.config.get('results_dir'):
                    # Get all job IDs from batch info
                    jobs = batch_info.get('jobs', [])
                    results_dir = self.config.get('results_dir', '')
                    if jobs:
                        job_ids = []
                        for job in jobs:
                            if hasattr(job, 'id'):
                                job_ids.append(job.id)
                        
                        if len(job_ids) == 1:
                            # Show full path for single file
                            path_sep = "" if results_dir.endswith("/") else "/"
                            output_file = f"→ {results_dir}{path_sep}{job_ids[0]}.json"
                        elif len(job_ids) > 1:
                            # Show first file path and count of others
                            path_sep = "" if results_dir.endswith("/") else "/"
                            output_file = f"→ {results_dir}{path_sep}{job_ids[0]}.json (+{len(job_ids)-1} more)"
                
                # Create the batch line with proper spacing
                batch_line = (
                    f"{batch_id:<18} {bar} "
                    f"{completed:>2}/{total:<2} {percentage:>3}% "
                    f"{status_text} "
                    f"${cost:>5.3f} "
                    f"{time_str:>8}"
                )
                
                # Add output file if available
                if output_file:
                    batch_line += f" {output_file}"
                
                tree.add(batch_line)
        
        # Footer information
        footer_parts = []
        
        # Add total cost first
        total_cost = stats.get('cost_usd', 0.0)
        footer_parts.append(f"[bold]Total Cost: ${total_cost:.3f}[/bold]")
        
        if self.config.get('results_dir'):
            footer_parts.append(f"Results: {self.config['results_dir']}")
        if self.config.get('state_file'):
            footer_parts.append(f"State: {self.config['state_file']}")
        if self.config.get('items_per_batch'):
            footer_parts.append(f"Items/Batch: {self.config['items_per_batch']}")
        if self.config.get('max_concurrent'):
            footer_parts.append(f"Max Concurrent: {self.config['max_concurrent']}")
        
        # Add elapsed time
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            elapsed_hours = int(elapsed // 3600)
            elapsed_minutes = int((elapsed % 3600) // 60)
            elapsed_seconds = int(elapsed % 60)
            
            if elapsed_hours > 0:
                elapsed_str = f"{elapsed_hours}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"
            else:
                elapsed_str = f"{elapsed_minutes:02d}:{elapsed_seconds:02d}"
                
            footer_parts.append(f"Elapsed: {elapsed_str}")
        
        if footer_parts:
            footer = " │ ".join(footer_parts)
            tree.add(f"\n[dim]{footer}[/dim]")
        
        return tree