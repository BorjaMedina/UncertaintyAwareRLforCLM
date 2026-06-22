"""Compute the scores and the transformed scores"""

__all__ = ["compute_transform"]

from typing import List, Tuple, Callable, Optional
import logging

import numpy as np

from reinvent_plugins.components.component_results import (
    ComponentResults,
    SmilesAssociatedComponentResults,
)
from .results import TransformResults

from pumas.desirability.catalogue import desirability_catalogue

logger = logging.getLogger(__name__)
SCORE_FUNC = Callable[[List[str]], ComponentResults]


def compute_component_scores(
    smilies: List[str],
    scoring_function: SCORE_FUNC,
    cache,
    filter_mask: Optional[np.ndarray[bool]],
    index_smiles: Optional[List[str]] = None,
) -> SmilesAssociatedComponentResults:
    """Compute a single component's scores and cache the results

    The mask filters out all SMILES unwanted for score computation: SMILES not
    passing a previous filter, invalid SMILES.  Scores are NaN
    when their values are unknown.
    Note: duplicates need to be handled above this level since ComponentResults is SMILES based

    The logic is as follows:
     1 get the smiles that need to be scored in this call
       this is all inputs except for masked smiles or those in cache
     2 compute the score for the new smiles and add to cache, or create a blank ComponentResults object
     3 update the scores of cached smiles to the SmilesResult object (scores + metadata) for cache hits
     4 add zeros-scores for any unique masked SMILES, e.g. invalid but non-duplicated

     In the case of fragments, we get SMILES from the fragment not the full molecule, but for consistency
     in output writing, caching scores etc

    :param smilies: list of SMILES
    :param scoring_function: the component callable
    :param cache: the cache for the component (will be modified)
    :param filter_mask: array mask to filter out invalid SMILES
    :param index_smiles: list of SMILES to index scores, used when the scored SMILES are fragments
    :returns: the scores
    """
    smilies_to_score, smiles_with_masked_scores, cache_hits = [], [], []
    # to avoid issues with duplicates in smilies we need to explicitly track indices,
    # this can occur for example in LinkInvent when multiple warheads generate the same linker
    smilies_to_score_indices, masked_scores_indices, cache_hits_indices = [], [], []

    use_cache = cache is not None  # some components need the cache to be turned off

    for idx, (filter_flag, smiles) in enumerate(zip(filter_mask, smilies)):
        if filter_flag:
            if use_cache and smiles in cache.keys():
                cache_hits.append(smiles)
                cache_hits_indices.append(idx)
            else:
                smilies_to_score.append(smiles)
                smilies_to_score_indices.append(idx)
        else:
            smiles_with_masked_scores.append(smiles)
            masked_scores_indices.append(idx)

    # we need different behaviour for duplicates and invalids - duplicates should not overwrite the scores
    # in ComponentResults. Therefore, we should always keep the scored version of smilies if it occurs with & without the filter variable
    smiles_with_masked_scores_filtered = []
    masked_scores_indices_filtered = []
    for smiles, idx in zip(smiles_with_masked_scores, masked_scores_indices):
        if smiles not in smilies_to_score + cache_hits:
            smiles_with_masked_scores_filtered.append(smiles)
            masked_scores_indices_filtered.append(idx)
    smiles_with_masked_scores = smiles_with_masked_scores_filtered
    masked_scores_indices = masked_scores_indices_filtered

    # handle the case of fragment SMILES, here we will use the full SMILES to keep the score associated with the record
    # while the fragment only for score computation
    if index_smiles is not None:
        index_smiles_to_score = [index_smiles[i] for i in smilies_to_score_indices]
        index_smiles_with_masked_scores = [index_smiles[i] for i in masked_scores_indices]
        cache_hits = [index_smiles[i] for i in cache_hits_indices]
        logger.debug(
            f"Using index smilies for fragment component {type(scoring_function).__name__}"
        )
    else:
        index_smiles_to_score = smilies_to_score
        index_smiles_with_masked_scores = smiles_with_masked_scores

    if len(smilies_to_score) > 0:
        component_results = SmilesAssociatedComponentResults(
            component_results=scoring_function(smilies_to_score), smiles=index_smiles_to_score
        )

        if use_cache:
            cache.update((smiles, component_results[smiles]) for smiles in index_smiles_to_score)
    else:
        # in this case, there are no compounds to score. Create blank ComponentResults
        component_results = SmilesAssociatedComponentResults.create_from_scores(
            smiles=[], scores=[[]]
        )

    if use_cache and len(cache_hits) > 0:  # update the results
        for smiles in cache_hits:
            component_results.data[smiles] = cache[smiles]

    if len(smiles_with_masked_scores) > 0:
        # one score per endpoint
        masked_scores = [(0.0,) * len(smiles_with_masked_scores)] * getattr(
            scoring_function, "number_of_endpoints", 1
        )

        component_results.update_scores(
            smiles=index_smiles_with_masked_scores, scores=masked_scores
        )

    return component_results

def compute_transform(
    component_type,
    params: Tuple,
    smilies: List[str],
    cache: dict,
    valid_mask: np.ndarray[bool],
    index_smiles: Optional[List[str]] = None,
    use_pumas: bool = False,
) -> TransformResults:
    """Compute the component score and transform of it

    :param component_type: type of the component
    :param params: parameters for the component
    :param smilies: list of SMILES
    :param cache: the component's cache
    :param valid_mask: mask for valid SMILES, i.e. false for invalid
    :param index_smiles: list of SMILES to index scores, used when the scored SMILES are fragments
    :param use_pumas: whether to use PUMAS transforms (element-wise float64) or reinvent transforms (array-wise)
    :returns: dataclass with transformed results
    """

    names, scoring_function, transforms, weights = params

    uncertainty_type = getattr(scoring_function, "uncertainty_type", None)
    if uncertainty_type is None:
        uncertainty_type = ""

    check_smiles = index_smiles if index_smiles is not None else smilies

    if use_pumas:
        transform_types = [transform.name if transform else None for transform in transforms]
    else:
        transform_types = [transform.params.type if transform else None for transform in transforms]

    if "gaussianSample" in uncertainty_type:
        # Run the deterministic component ONCE to get mu (predictions) and sigma² (uncertainties).
        # Then sample 200 times from N(mu, sigma) in numpy — no model overhead — and average
        # the post-transform scores.  The component must expose two endpoints in this order:
        #   endpoint 0: "predictions"   (mu)   — has the desired transform (e.g. sigmoid)
        #   endpoint 1: "uncertainties" (sigma²) — typically weight=0, no transform
        logger.debug(f"Computing gaussianSample for component {component_type}")

        component_results = compute_component_scores(
            smilies, scoring_function, cache, valid_mask, index_smiles
        )
        component_results.uncertainty_type = uncertainty_type

        missing_scores = [s for s in check_smiles if s not in component_results.data]
        if missing_scores:
            raise RuntimeError(f"Missing scores for {component_type} for {missing_scores}")

        all_scores = list(
            component_results.fetch_scores(smiles=check_smiles, transpose=True)
        )

        mu = np.array(all_scores[0], dtype=float)
        sigma2 = np.array(all_scores[1], dtype=float) if len(all_scores) > 1 else np.zeros_like(mu)
        sigma = np.sqrt(np.abs(sigma2))

        prediction_transform = transforms[0]
        n_samples = 200
        sampled_transformed = []
        for _ in range(n_samples):
            sample = np.random.normal(mu, sigma)
            if use_pumas:
                t = np.array([prediction_transform(s) for s in sample]) if prediction_transform else sample
            else:
                t = prediction_transform(sample) if prediction_transform else sample
            sampled_transformed.append(t * valid_mask)

        mean_transformed = np.mean(sampled_transformed, axis=0)
        std_transformed = np.std(sampled_transformed, axis=0)

        # Store std in per-SMILES metadata so it appears in the CSV automatically
        for i, smi in enumerate(check_smiles):
            if smi in component_results.data:
                current_meta = component_results.data[smi].metadata or {}
                component_results.data[smi].metadata = {**current_meta, "score_std": float(std_transformed[i])}

        # First endpoint = averaged sampled score; remaining endpoints passed through as-is
        transformed_scores = [mean_transformed]
        for raw_scores, transform in zip(all_scores[1:], transforms[1:]):
            raw = np.array(raw_scores, dtype=float)
            if use_pumas:
                t = np.array([transform(s) for s in raw]) if transform else raw
            else:
                t = transform(raw) if transform else raw
            transformed_scores.append(t * valid_mask)

        transform_result = TransformResults(
            component_type,
            names,
            transform_types,
            transformed_scores,
            component_results,
            weights,
        )

    elif "meanPostMPO" in uncertainty_type:
        logger.debug(f"Computing meanPostMPO for component {component_type}")
        all_transformed = []
        for _ in range(200):
            component_results = compute_component_scores(
                smilies, scoring_function, cache, valid_mask, index_smiles
            )

            missing_scores = [s for s in check_smiles if s not in component_results.data]
            if missing_scores:
                raise RuntimeError(f"Missing scores for {component_type} for {missing_scores}")

            transformed_scores = []
            for scores, transform in zip(
                component_results.fetch_scores(smiles=check_smiles, transpose=True),
                transforms,
            ):
                if use_pumas:
                    transformed = [transform(score) for score in scores] if transform else scores
                else:
                    transformed = transform(scores) if transform else scores
                transformed_scores.append(transformed * valid_mask)

            all_transformed.append(transformed_scores)

        component_results.uncertainty_type = uncertainty_type

        transform_result = TransformResults(
            component_type,
            names,
            transform_types,
            np.array(np.mean(all_transformed, axis=0)),
            component_results,
            weights,
        )

    else:
        component_results = compute_component_scores(
            smilies, scoring_function, cache, valid_mask, index_smiles
        )
        component_results.uncertainty_type = uncertainty_type

        missing_scores = [s for s in check_smiles if s not in component_results.data]
        if missing_scores:
            logger.debug(f"{cache=}")
            logger.debug(f"{component_results.data.keys()=}")
            raise RuntimeError(f"Missing scores for {component_type} for {missing_scores}")

        transformed_scores = []
        for scores, transform in zip(
            component_results.fetch_scores(smiles=check_smiles, transpose=True),
            transforms,
        ):
            # NOTE: The valid SMILES here are the molecules which are both valid
            #       AND not duplicates i.e. duplicate (and invalid) SMILES in the
            #       BATCH are scored as zero.  This is the same behaviour as in
            #       REINVENT3 and earlier.  The duplicate handling is independent
            #       of the diversity filter (DF) which scores ALL duplicates as
            #       zero i.e. the DF acts globally.
            if use_pumas:
                # PUMAS Transforms operate on float64 so the transformed result may be slightly different to reinvent base scoring.
                transformed = [transform(score) for score in scores] if transform else scores
            else:
                transformed = transform(scores) if transform else scores
            transformed_scores.append(transformed * valid_mask)

        transform_result = TransformResults(
            component_type,
            names,
            transform_types,
            transformed_scores,
            component_results,
            weights,
        )

    return transform_result
