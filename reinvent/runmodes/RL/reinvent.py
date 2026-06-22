"""The Reinvent optimization algorithm"""

from __future__ import annotations

__all__ = ["ReinventLearning"]
import logging
from typing import TYPE_CHECKING

import numpy as np

from reinvent.models.model_factory.sample_batch import SmilesState
from .learning import Learning

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults

logger = logging.getLogger(__name__)


class ReinventLearning(Learning):
    """Reinvent optimization"""

    def update(self, results: ScoreResults, orig_smilies):
        """Run the learning strategy"""
        distances=[1]*len(results.total_scores)
        all_distances=[]
        eps = 1e-6
        for component in results.completed_components:
            if component.component_result.uncertainty_type and component.component_result.uncertainty_type.startswith("reward"):
                component_distances=component.transformed_scores[0]/(np.mean(component.transformed_scores[0])+eps)
                all_distances.append(component_distances)

        if all_distances:
            distances = np.mean(all_distances, axis=0)
        
        agent_nlls = self._state.agent.likelihood_smiles(self.sampled.items2)
        prior_nlls = self.prior.likelihood_smiles(self.sampled.items2)
        return self.reward_nlls(
            orig_smilies,
            results.total_scores,
            agent_nlls,
            prior_nlls,
            np.argwhere(self.sampled.states == SmilesState.VALID).flatten(),
            self.inception,
            self._state.agent,
            distances=np.array(distances)
        )
