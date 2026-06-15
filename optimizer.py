import numpy as np
from rich.console import Console

from config import MAX_TARGET_LEVERAGE, OPTIMIZER_TOLERANCE, MAX_MARGIN_CALL_PROBABILITY

console = Console()

class CRNGridOptimizer:
    def __init__(self, simulator):
        self.simulator = simulator

    def optimize(self) -> dict:
        console.print("\n[bold]Executing Vectorized CRN Leverage Search...[/bold]")
        
        leverage_grid = np.round(np.arange(1.0, MAX_TARGET_LEVERAGE + OPTIMIZER_TOLERANCE, OPTIMIZER_TOLERANCE), 2)
        ruin_probs = self.simulator.simulate_grid(leverage_grid)
        
        for lev, prob in zip(leverage_grid[::25], ruin_probs[::25]):
             console.print(f"[dim]  Tier Check: {lev:.2f}x -> SMA/EL Breach Prob: {prob:.2%}[/dim]")
        
        safe_indices = np.where(ruin_probs <= MAX_MARGIN_CALL_PROBABILITY)[0]
        
        if len(safe_indices) == 0:
            console.print("[bold red][!] WARNING: Portfolio breaches risk budget even with NO margin leverage applied (1.0x).[/bold red]")
            best_l = 1.0
        else:
            best_idx = safe_indices[-1]
            best_l = float(leverage_grid[best_idx])
            best_prob = ruin_probs[best_idx]
            console.print(f"✓ Optimal limit constraint found: [bold green]{best_l:.2f}x[/bold green] (Ruin Prob: {best_prob:.2%})\n")

        console.print("[dim]Extracting full portfolio trajectories for plotting...[/dim]")
        final_sim = self.simulator.simulate(target_leverage=best_l, store_paths=True, store_history=True)
        final_sim["optimal_target_leverage"] = best_l
        final_sim["optimizer_method"] = "Vectorized_CRN_Grid"
        final_sim["grid_leverages"] = leverage_grid
        final_sim["grid_ruin_probs"] = ruin_probs
        
        return final_sim