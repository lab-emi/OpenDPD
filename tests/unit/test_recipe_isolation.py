"""Editing an experiment must never change reviewed defaults for other users."""

from opendpd.services.recipes import get_recipe, instantiate


def test_instantiated_recipes_do_not_share_mutable_training_or_model_parameters():
    recipe = get_recipe("pa-gru-smoke-v1")
    before = recipe.to_dict()
    first = instantiate(recipe.recipe_id, "first-dataset")
    first.training.batch_size = 512
    first.model.parameters["hidden_size"] = 100
    second = instantiate(recipe.recipe_id, "second-dataset")
    seeded = instantiate(recipe.recipe_id, "third-dataset", seed=42)
    seeded.model.parameters.clear()
    assert recipe.to_dict() == before
    assert second.training.batch_size == before["training"]["batch_size"]
    assert second.model.parameters == before["model"]["parameters"]
    assert seeded.training.seed == 42
    assert second.training.seed == before["training"]["seed"]
