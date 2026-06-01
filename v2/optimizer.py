import math
import numpy as np
from tqdm import tqdm
from config import (
    MAX_TARGET_LEVERAGE,
    MAX_MARGIN_CALL_PROBABILITY,
    OPTIMIZER_GRID_POINTS,
    OPTIMIZER_REFINEMENT_POINTS,
    MONOTONICITY_TOLERANCE
)

class MarginOptimizer:
    """Optimizes the unified policy leverage."""
    
    def __init__(self, simulator):
        self.simulator = simulator

    def optimize(self) -> dict:
        """Finds the optimal unified leverage parameter with safety checks."""
        print("\n[*] Running leverage optimizer with baseline and monotonicity checks...")

        tolerance = 0.001
        risk_cache = {}

        def risk_at(leverage: float) -> float:
            key = round(float(leverage), 8)

            if key not in risk_cache:
                res = self.simulator.simulate(float(leverage), store_paths=False)
                risk_cache[key] = res["prob_ruin"]

            return risk_cache[key]

        # --- 1. Baseline feasibility check ---
        baseline_risk = risk_at(1.0)

        if baseline_risk > MAX_MARGIN_CALL_PROBABILITY:
            raise RuntimeError(
                "\n[!] No safe additional-purchase policy exists under the current assumptions.\n"
                f"    No-new-purchase policy risk at L=1.0: {baseline_risk:.2%}\n"
                f"    Maximum allowed risk: {MAX_MARGIN_CALL_PROBABILITY:.2%}\n"
                "    Note: L=1.0 does not forcibly delever the existing balance sheet."
            )

        # --- 2. Upper-bound check ---
        upper_risk = risk_at(MAX_TARGET_LEVERAGE)

        if upper_risk <= MAX_MARGIN_CALL_PROBABILITY:
            final_sim = self.simulator.simulate(MAX_TARGET_LEVERAGE, store_paths=True)
            final_sim["optimal_target_leverage"] = MAX_TARGET_LEVERAGE
            final_sim["constraint_binding"] = False
            final_sim["optimizer_method"] = "upper_bound_safe"
            final_sim["risk_curve_non_monotonic"] = False
            return final_sim

        # --- 3. Coarse monotonicity check ---
        grid = np.linspace(1.0, MAX_TARGET_LEVERAGE, OPTIMIZER_GRID_POINTS)
        grid_risks = []

        with tqdm(
            total=len(grid),
            desc="Coarse Risk Grid",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour="yellow"
        ) as pbar:
            for leverage in grid:
                pbar.set_postfix({"Testing L*": f"{leverage:.3f}x"})
                grid_risks.append(risk_at(float(leverage)))
                pbar.update(1)

        grid_risks = np.array(grid_risks)

        risk_diffs = np.diff(grid_risks)
        non_monotonic = bool(np.any(risk_diffs < -MONOTONICITY_TOLERANCE))

        safe_grid = grid[grid_risks <= MAX_MARGIN_CALL_PROBABILITY]

        if len(safe_grid) == 0:
            raise RuntimeError(
                "\n[!] Baseline passed earlier, but coarse grid found no safe leverage.\n"
                "    This indicates numerical instability in the optimizer."
            )

        if non_monotonic:
            print(
                "[!] Risk curve appears non-monotonic at coarse resolution. "
                "Using robust local grid refinement instead of bisection."
            )

            grid_step = (MAX_TARGET_LEVERAGE - 1.0) / (OPTIMIZER_GRID_POINTS - 1)
            best_coarse = float(np.max(safe_grid))

            left = max(1.0, best_coarse - grid_step)
            right = min(MAX_TARGET_LEVERAGE, best_coarse + grid_step)

            refine_grid = np.linspace(left, right, OPTIMIZER_REFINEMENT_POINTS)
            refine_risks = []

            with tqdm(
                total=len(refine_grid),
                desc="Refined Risk Grid",
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="green"
            ) as pbar:
                for leverage in refine_grid:
                    pbar.set_postfix({"Testing L*": f"{leverage:.3f}x"})
                    refine_risks.append(risk_at(float(leverage)))
                    pbar.update(1)

            refine_risks = np.array(refine_risks)
            safe_refined = refine_grid[refine_risks <= MAX_MARGIN_CALL_PROBABILITY]

            if len(safe_refined) > 0:
                optimal_leverage = float(np.max(safe_refined))
            else:
                optimal_leverage = best_coarse

            optimizer_method = "grid_refinement"

        else:
            # --- 4. Standard bisection, now justified by the monotonicity check ---
            print("[*] Coarse risk curve passed monotonicity check. Running bisection search...")

            low, high = 1.0, MAX_TARGET_LEVERAGE
            optimal_leverage = 1.0

            expected_steps = math.ceil(math.log2((high - low) / tolerance))

            with tqdm(
                total=expected_steps,
                desc="Bisection Optimizer",
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="green"
            ) as pbar:
                while high - low > tolerance:
                    mid = (low + high) / 2.0
                    pbar.set_postfix({"Testing L*": f"{mid:.3f}x"})

                    mid_risk = risk_at(mid)

                    if mid_risk <= MAX_MARGIN_CALL_PROBABILITY:
                        optimal_leverage = mid
                        low = mid
                    else:
                        high = mid

                    pbar.update(1)

            optimizer_method = "bisection"

        final_sim = self.simulator.simulate(optimal_leverage, store_paths=True)
        final_sim["optimal_target_leverage"] = optimal_leverage
        final_sim["constraint_binding"] = True
        final_sim["optimizer_method"] = optimizer_method
        final_sim["risk_curve_non_monotonic"] = non_monotonic

        return final_sim