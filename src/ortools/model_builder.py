"""
CP-SAT Model Builder

Assembles the complete CP-SAT model with variables and constraints.

Coordinates:
    - Variable creation (via VariableFactory)
    - Constraint addition (via ConstraintFactory)
    - Model assembly and validation
"""

from typing import Dict
from ortools.sat.python import cp_model

from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.ortools.variable_factory import VariableFactory
from src.ortools.constraint_factory import ConstraintFactory


class ModelBuilder:
    """
    Builds complete CP-SAT model for course scheduling.

    Orchestrates variable creation and constraint addition.
    """

    def __init__(self, context: SchedulingContext, qts: QuantumTimeSystem):
        """
        Initialize model builder.

        Args:
            context: SchedulingContext with all entities
            qts: QuantumTimeSystem for time calculations
        """
        self.context = context
        self.qts = qts
        self.var_factory = VariableFactory(context)

    def build_model(self) -> tuple[cp_model.CpModel, Dict, VariableFactory]:
        """
        Build complete CP-SAT model.

        Steps:
            1. Create CP-SAT model
            2. Create decision variables
            3. Add hard constraints

        Returns:
            Tuple of (model, session_vars, var_factory)
        """
        print("\n[Building CP-SAT Model]")

        # Create model
        model = cp_model.CpModel()

        # Create variables
        print("Creating decision variables...")
        session_vars = self.var_factory.create_session_variables(model)
        print(f"  Created {len(session_vars)} session variables")

        # Create constraint factory
        constraint_factory = ConstraintFactory(self.context, self.var_factory, self.qts)

        # Add constraints
        print("Adding constraints...")
        constraint_factory.add_all_constraints(model, session_vars)

        print("[Model built successfully]\n")

        return model, session_vars, self.var_factory
