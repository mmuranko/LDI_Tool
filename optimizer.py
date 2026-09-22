import numpy as np
from rich.console import Console

from config import MAX_TARGET_LEVERAGE, OPTIMIZER_TOLERANCE, MAX_MARGIN_CALL_PROBABILITY

console = Console(highlight=False)

class CRNGridOptimizer:
    def __init__(self, simulator):
        self.simulator = simulator

    def optimize(self) -> dict:
        console.print("\n[bold]OPTIMIZATION[/bold]")
        console.print("────────────────────────────────────────────────────────────")
        
        leverage_grid = np.round(np.arange(1.0, MAX_TARGET_LEVERAGE + OPTIMIZER_TOLERANCE, OPTIMIZER_TOLERANCE), 2)
        ruin_probs = self.simulator.simulate_grid(leverage_grid)
        
        step = max(1, len(leverage_grid) // 5)
        for lev, prob in zip(leverage_grid[::step], ruin_probs[::step]):
             console.print(f"Tier Evaluated: {lev:.2f}x -> SMA/EL Ruin Prob: {prob:.2%}")
        
        safe_indices = np.where(ruin_probs <= MAX_MARGIN_CALL_PROBABILITY)[0]
        if len(safe_indices) == 0:
            best_l, best_prob = 1.00, ruin_probs[0]
            console.print("\nWarning: Constraint breached at baseline (1.00x). Deleveraging recommended.\n")
        else:
            best_idx = safe_indices[-1]
            best_l, best_prob = float(leverage_grid[best_idx]), ruin_probs[best_idx]
            console.print(f"\nOptimal Configuration Found: [bold]{best_l:.2f}x[/bold] (Risk: {best_prob:.2%})\n")

        final_sim = self.simulator.simulate(target_leverage=best_l, store_paths=True, store_history=True)
        final_sim.update({"optimal_target_leverage": best_l, "optimizer_method": "Vectorized_CRN_Grid", "grid_leverages": leverage_grid, "grid_ruin_probs": ruin_probs})
        return final_sim