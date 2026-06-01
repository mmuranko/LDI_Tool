import math
from tqdm import tqdm
from config import MAX_TARGET_LEVERAGE, MAX_MARGIN_CALL_PROBABILITY

class MarginOptimizer:
    """Optimizes the unified policy leverage against the ruin epsilon."""
    
    def __init__(self, simulator):
        self.simulator = simulator

    def optimize(self) -> dict:
        """Finds the optimal unified leverage parameter L* using Bisection Search."""
        print("\n[*] Running Bisection Search for Optimal Unified Policy Leverage...")
        
        low, high = 1.0, MAX_TARGET_LEVERAGE
        optimal_leverage = 1.0
        tolerance = 0.001 
        
        # Calculate exact number of steps for the progress bar
        expected_steps = math.ceil(math.log2((high - low) / tolerance))
        
        with tqdm(total=expected_steps, desc="Bisection Optimizer", bar_format="{l_bar}{bar:30}{r_bar}", colour="green") as pbar:
            while high - low > tolerance:
                mid = (low + high) / 2.0
                
                # Update the progress bar to show the current L* being tested
                pbar.set_postfix({"Testing L*": f"{mid:.3f}x"})
                
                res = self.simulator.simulate(mid)
                
                if res["prob_ruin"] <= MAX_MARGIN_CALL_PROBABILITY:
                    optimal_leverage = mid
                    low = mid # We can afford more leverage
                else:
                    high = mid # Too risky, reduce leverage
                    
                pbar.update(1)
                
        # Run one final simulation at the optimal leverage
        final_sim = self.simulator.simulate(optimal_leverage)
        
        # [FIX]: Re-tag the dictionary so main.py can read it
        final_sim["optimal_target_leverage"] = optimal_leverage
        
        return final_sim