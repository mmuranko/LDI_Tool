import numpy as np
from rich.console import Console

from config import (
    MAX_TARGET_LEVERAGE, OPTIMIZER_TOLERANCE,
    COARSE_GRID_STEP, MAX_MARGIN_CALL_PROBABILITY
)

console = Console(highlight=False)


class CRNGridOptimizer:
    def __init__(self, simulator):
        self.simulator = simulator

    def optimize(self, store_history: bool = False) -> dict:
        console.print("\n[bold]OPTIMIZATION (2-Stage Coarse-to-Fine CRN Search)[/bold]")
        console.print("────────────────────────────────────────────────────────────")

        # Stage 1: Coarse Grid Evaluation
        coarse_grid = np.round(
            np.arange(1.0, MAX_TARGET_LEVERAGE + COARSE_GRID_STEP * 0.5, COARSE_GRID_STEP), 3
        )
        coarse_probs = self.simulator.simulate_grid(
            coarse_grid, desc=f"Stage 1: Coarse Search ({len(coarse_grid)} tiers)..."
        )

        for lev, prob in zip(coarse_grid[::3], coarse_probs[::3]):
            console.print(f"Coarse Tier: {lev:.3f}x -> SMA/EL Ruin Prob: {prob:.2%}")

        safe_coarse = np.where(coarse_probs <= MAX_MARGIN_CALL_PROBABILITY)[0]
        if len(safe_coarse) == 0:
            best_l = 1.00
            best_prob = float(coarse_probs[0])
            full_leverages = coarse_grid
            full_probs = coarse_probs
            console.print("\nWarning: Constraint breached at baseline (1.00x). Deleveraging recommended.\n")
        else:
            last_safe_idx = int(safe_coarse[-1])
            low_lev = float(coarse_grid[last_safe_idx])
            high_lev = float(coarse_grid[min(last_safe_idx + 1, len(coarse_grid) - 1)])

            if high_lev > low_lev + OPTIMIZER_TOLERANCE * 0.5:
                fine_grid = np.round(
                    np.arange(low_lev, high_lev + OPTIMIZER_TOLERANCE * 0.5, OPTIMIZER_TOLERANCE), 3
                )
                fine_probs = self.simulator.simulate_grid(
                    fine_grid, desc=f"Stage 2: Fine Refinement [{low_lev:.2f}x - {high_lev:.2f}x]..."
                )
                safe_fine = np.where(fine_probs <= MAX_MARGIN_CALL_PROBABILITY)[0]
                if len(safe_fine) > 0:
                    best_fine_idx = int(safe_fine[-1])
                    best_l = float(fine_grid[best_fine_idx])
                    best_prob = float(fine_probs[best_fine_idx])
                else:
                    best_l = low_lev
                    best_prob = float(coarse_probs[last_safe_idx])

                # Merge coarse and fine points for the risk-frontier visualization
                combined_dict = {round(float(k), 4): float(v) for k, v in zip(coarse_grid, coarse_probs)}
                for k, v in zip(fine_grid, fine_probs):
                    combined_dict[round(float(k), 4)] = float(v)
                sorted_keys = sorted(combined_dict.keys())
                full_leverages = np.array(sorted_keys, dtype=np.float32)
                full_probs = np.array([combined_dict[k] for k in sorted_keys], dtype=np.float32)
            else:
                best_l = low_lev
                best_prob = float(coarse_probs[last_safe_idx])
                full_leverages = coarse_grid
                full_probs = coarse_probs

            console.print(f"\nOptimal Configuration Found: [bold]{best_l:.3f}x[/bold] (Risk: {best_prob:.2%})\n")

        final_sim = self.simulator.simulate(
            target_leverage=best_l, store_paths=True, store_history=store_history
        )
        final_sim.update({
            "optimal_target_leverage": best_l,
            "optimizer_method": "TwoStage_CRN_Grid",
            "grid_leverages": full_leverages,
            "grid_ruin_probs": full_probs
        })
        return final_sim