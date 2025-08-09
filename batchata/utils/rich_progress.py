"""Rich progress display for batch runs."""

import threading
from datetime import datetime
from typing import Dict, Optional

from rich.console import Console
from rich.live import Live
from rich.tree import Tree
from rich.text import Text

# Constants
PROGRESS_BAR_WIDTH = 25


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
                refresh_per_second=4,  # Reduced refresh rate to avoid flicker
                auto_refresh=False  # Disable auto-refresh to prevent race conditions with manual updates
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
            
            # Update live display (synchronized to prevent race conditions)
            if self.live:
                self.live.update(self._create_display())
                # Force refresh since auto_refresh is disabled
                self.live.refresh()
    
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
        
        # Main tree with colored header
        batches_done = stats.get('batches_completed', 0)
        batches_total = stats.get('batches_total', 0)
        requests_done = stats.get('completed', 0) 
        requests_total = stats.get('total', 0)
        
        header_parts = []
        
        # Color batches based on completion
        if batches_done == batches_total and batches_total > 0:
            header_parts.append(f"[green]Batches: {batches_done}/{batches_total}[/green]")
        else:
            header_parts.append(f"[cyan]Batches: {batches_done}/{batches_total}[/cyan]")
            
        # Color requests based on completion
        if requests_done == requests_total and requests_total > 0:
            header_parts.append(f"[green]Requests: {requests_done}/{requests_total}[/green]")
        else:
            header_parts.append(f"[cyan]Requests: {requests_done}/{requests_total}[/cyan]")
        
        if running_count > 0:
            header_parts.append(f"[blue]Running: {running_count}[/blue]")
        if failed_count > 0:
            header_parts.append(f"[red]Failed: {failed_count}[/red]")
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
                estimated_cost = batch_info.get('estimated_cost', 0.0)
                provider = batch_info.get('provider', 'Unknown')
                
                # Determine tree symbol
                is_last = idx == num_batches - 1
                tree_symbol = "└─" if is_last else "├─"
                
                # Extract job counts
                failed_count = batch_info.get('failed', 0)
                success_count = completed
                total_processed = success_count + failed_count
                progress_pct = (total_processed / total) if total > 0 else 0
                
                # Create progress bar based on status
                bar = self._create_progress_bar(status, success_count, failed_count, total, progress_pct)
                
                # Format status text
                status_text = self._format_status_text(status, failed_count)
                
                # Calculate elapsed time
                start_time = batch_info.get('start_time')
                if start_time and status in ['running', 'complete', 'failed', 'cancelled']:
                    # For completed batches, use completion time to freeze the timer
                    if status in ['complete', 'failed', 'cancelled']:
                        completion_time = batch_info.get('completion_time')
                        if completion_time:
                            elapsed = (completion_time - start_time).total_seconds()
                        else:
                            # Fallback if completion_time not available
                            elapsed = (datetime.now() - start_time).total_seconds()
                    else:
                        # For running batches, use current time
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
                
                # Format percentage based on total processed (successful + failed)
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
                
                # Format cost display based on status
                if status in ['running', 'pending']:
                    cost_text = f"${estimated_cost:>5.3f} (estimated)"
                else:
                    cost_text = f"${cost:>5.3f}"
                
                # Create the batch line
                display_stats = self._get_display_stats(status, success_count, failed_count, total)
                batch_line = (
                    f"{provider} {batch_id:<18} {bar} "
                    f"{display_stats['completed']:>2}/{total:<2} ({display_stats['percentage']}% done) {status_text:<15} "
                    f"{cost_text} "
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
        if self.config.get('max_parallel_batches'):
            footer_parts.append(f"Max Parallel Batches: {self.config['max_parallel_batches']}")
        
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
    
    def _create_progress_bar(self, status: str, success_count: int, failed_count: int, total: int, progress_pct: float) -> str:
        """Create a progress bar showing success/failure proportions."""
        
        if status == 'complete':
            return f"[bold green]{'━' * PROGRESS_BAR_WIDTH}[/bold green]"
        
        if status == 'failed':
            return self._create_mixed_bar(success_count, failed_count, total, PROGRESS_BAR_WIDTH)
        
        if status == 'cancelled':
            filled = int(progress_pct * PROGRESS_BAR_WIDTH)
            return f"[bold yellow]{'━' * filled}[/bold yellow][dim yellow]{'━' * (PROGRESS_BAR_WIDTH - filled)}[/dim yellow]"
        
        if status == 'running':
            filled = int(progress_pct * PROGRESS_BAR_WIDTH)
            if filled < PROGRESS_BAR_WIDTH:
                return f"[bold blue]{'━' * filled}[/bold blue][blue]╸[/blue][dim white]{'━' * (PROGRESS_BAR_WIDTH - filled - 1)}[/dim white]"
            return f"[bold blue]{'━' * PROGRESS_BAR_WIDTH}[/bold blue]"
        
        # Pending
        return f"[dim white]{'━' * PROGRESS_BAR_WIDTH}[/dim white]"
    
    def _create_mixed_bar(self, success_count: int, failed_count: int, total: int, bar_width: int) -> str:
        """Create a bar showing green (success) and red (failed) proportions."""
        if total == 0:
            return f"[dim white]{'━' * bar_width}[/dim white]"
        
        # Use integer division to calculate base widths
        success_width = (success_count * bar_width) // total
        failed_width = (failed_count * bar_width) // total
        
        # Distribute remainder to maintain exact bar_width
        remainder = bar_width - success_width - failed_width
        if remainder > 0:
            # Distribute remainder based on which segment has larger fractional part
            success_fraction = (success_count * bar_width) % total
            failed_fraction = (failed_count * bar_width) % total
            
            if success_fraction >= failed_fraction:
                success_width += remainder
            else:
                failed_width += remainder
        
        # Build the bar
        bar_parts = []
        if success_width > 0:
            bar_parts.append(f"[bold green]{'━' * success_width}[/bold green]")
        if failed_width > 0:
            bar_parts.append(f"[bold red]{'━' * failed_width}[/bold red]")
        
        return "".join(bar_parts)
    
    def _format_status_text(self, status: str, failed_count: int) -> str:
        """Format the status text with appropriate colors and details."""
        if status == 'complete':
            return "[bold green]Complete[/bold green]"
        elif status == 'failed':
            if failed_count > 0:
                return f"[bold red]Failed ({failed_count})[/bold red]"
            return "[bold red]Failed[/bold red]"
        elif status == 'cancelled':
            return "[bold yellow]Cancelled[/bold yellow]"
        elif status == 'running':
            spinner = self._spinner_frames[self._spinner_index]
            return f"[bold blue]{spinner} Running[/bold blue]"
        else:
            return "[dim]Pending[/dim]"
    
    def _get_display_stats(self, status: str, success_count: int, failed_count: int, total: int) -> dict:
        """Get the display statistics (completed count and percentage)."""
        if status == 'failed' and failed_count > 0:
            # For failed batches, show success count to make it clear
            completed = success_count
            percentage = int((success_count / total) * 100) if total > 0 else 0
        else:
            # For other statuses, show total processed
            completed = success_count + failed_count
            percentage = int((completed / total) * 100) if total > 0 else 0
        
        return {'completed': completed, 'percentage': percentage}