import pytest
import torch
from lnn import Proposition, ProductAnd, Model

def test_product_and_basic():
    A = Proposition("A")
    B = Proposition("B")
    my_and = ProductAnd(A, B)

    # Create a fresh model for each test run
    model = Model()
    model.add_knowledge(my_and)

    for a in [0.0, 0.2, 0.5, 0.8, 1.0]:
        for b in [0.0, 0.2, 0.5, 0.8, 1.0]:
            # ============================================================
            # CRITICAL: Reset the model state to prevent stale bounds
            # ============================================================
            if hasattr(model, 'clear_data'):
                model.clear_data()
            elif hasattr(model, 'reset_bounds'):
                model.reset_bounds()
            else:
                # Fallback: manually reset propositions to Unknown
                model.add_data({A: (0.0, 1.0), B: (0.0, 1.0)})

            # Now set the new exact bounds
            model.add_data({A: (a, a), B: (b, b)})

            model.infer(iterations=10)

            # Retrieve bounds
            bounds = my_and.get_data()
            if bounds is None:
                bounds = my_and.neuron.bounds_table

            # Extract lower bound
            if isinstance(bounds, torch.Tensor):
                # bounds shape is [batch, 2] or [2]
                if bounds.dim() == 2:
                    val = bounds[0, 0].item()
                else:
                    val = bounds[0].item()
            elif isinstance(bounds, tuple):
                val = bounds[0]
            else:
                val = bounds

            val = float(val)
            gt = a * b

            assert abs(val - gt) < 1e-5, f"Failed at {a}*{b}: Expected {gt}, got {val}"