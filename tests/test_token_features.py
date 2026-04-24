"""Tests for the static-feature token path in PPOActorCritic.

Covers:
  * CardFeatureTable values match the underlying CardRegistry.
  * unpack_state_tokens shape + content parity with unpack_state.
  * PPOActorCritic forward parity / shape / NaN-safety with token_features.
  * Empty-zone safety with the static-feature path under both poolers.
  * Gradient flow through token_proj and the static-feature concat path.
  * Constructor errors when token_features=True but no registry is passed.
"""
from __future__ import annotations

import pytest
import torch

from src.cards.card import Card, CardType
from src.cards.factions import Faction
from src.cards.registry import CardDef, CardRegistry, build_registry
from src.config import ModelConfig
from src.encoding.state_utils import (
    ZONE_NAMES,
    unpack_state,
    unpack_state_tokens,
)
from src.encoding.state_encoder import get_state_size
from src.encoding.action_encoder import get_action_space_size
from src.ppo.ppo_actor_critic import (
    PPOActorCritic,
    CardFeatureTable,
    NUMERIC_DIM,
    NUM_FACTIONS,
    STATIC_FEATURE_DIM,
)


def _toy_registry() -> CardRegistry:
    """Hand-built registry exercising cost, defense, type, and faction bits."""
    defs = [
        CardDef(0, "Scout", cost=0, card_type=CardType.SHIP),
        CardDef(1, "Viper", cost=0, card_type=CardType.SHIP),
        CardDef(
            2, "Battle Pod", cost=2, card_type=CardType.SHIP,
            faction=Faction.BLOB,
        ),
        CardDef(
            3, "Blob Wheel", cost=3, card_type=CardType.BASE,
            defense=5, faction=Faction.BLOB,
            ally_factions=Faction.BLOB,
        ),
        CardDef(
            4, "Trade Bot", cost=1, card_type=CardType.SHIP,
            faction=Faction.MACHINE_CULT,
        ),
        CardDef(
            5, "Royal Redoubt", cost=6, card_type=CardType.OUTPOST,
            defense=6, faction=Faction.STAR_EMPIRE,
            ally_factions=Faction.STAR_EMPIRE,
        ),
        CardDef(
            6, "Multi", cost=4, card_type=CardType.SHIP,
            faction=Faction.BLOB | Faction.STAR_EMPIRE,
            ally_factions=Faction.ALL,
        ),
    ]
    return CardRegistry(defs, starter_names=[])


def test_card_feature_table_values_match_registry():
    reg = _toy_registry()
    table = CardFeatureTable(reg)
    f = table.features
    assert f.shape == (reg.num_cards, STATIC_FEATURE_DIM)

    # Scout: cost 0, ship → only is_ship is set.
    scout = f[0]
    assert scout[0] == 0.0 and scout[1] == 0.0
    assert scout[2] == 1.0 and scout[3] == 0.0 and scout[4] == 0.0
    assert torch.all(scout[5:] == 0.0)

    # Blob Wheel: cost 3 → 0.3, defense 5 → 0.5, base, blob faction + ally.
    bw = f[3]
    assert bw[0] == pytest.approx(0.3)
    assert bw[1] == pytest.approx(0.5)
    assert bw[2] == 0.0 and bw[3] == 1.0 and bw[4] == 0.0
    blob_idx = 0  # _FACTION_BITS order: BLOB, MC, SE, TF
    assert bw[5 + blob_idx] == 1.0
    assert bw[5 + NUM_FACTIONS + blob_idx] == 1.0

    # Royal Redoubt: outpost — both is_base and is_outpost set; star empire.
    rr = f[5]
    assert rr[3] == 1.0 and rr[4] == 1.0
    se_idx = 2
    assert rr[5 + se_idx] == 1.0
    assert rr[5 + NUM_FACTIONS + se_idx] == 1.0

    # Multi: two faction bits set; ally_factions == ALL → all four ally bits.
    multi = f[6]
    assert multi[5 + 0] == 1.0  # blob
    assert multi[5 + 2] == 1.0  # star empire
    assert multi[5 + 1] == 0.0  # not machine cult faction
    assert torch.all(multi[5 + NUM_FACTIONS:5 + 2 * NUM_FACTIONS] == 1.0)


def test_card_feature_table_starter_without_carddef_is_zero():
    # CardRegistry will list 'Explorer' in card_names but won't have a CardDef
    # for it (starters added via name only). Their feature row must be zeros.
    defs = [CardDef(0, "Scout", cost=0, card_type=CardType.SHIP)]
    reg = CardRegistry(defs, starter_names=["Scout", "Explorer"])
    assert "Explorer" in reg.card_index_map
    table = CardFeatureTable(reg)
    explorer_idx = reg.card_index_map["Explorer"]
    assert torch.all(table.features[explorer_idx] == 0.0)


def test_unpack_state_tokens_shapes_and_content():
    cards = [f"C{i}" for i in range(10)]
    num_cards = len(cards)
    state_dim = get_state_size(cards)
    action_dim = get_action_space_size(cards)

    x = torch.rand(4, state_dim)
    presence, numerics, single = unpack_state_tokens(x, num_cards, action_dim)
    assert single is False
    assert presence.shape == (4, len(ZONE_NAMES), num_cards)
    assert numerics.shape == (4, NUMERIC_DIM)

    # Zone slices must match unpack_state directly.
    pieces, _ = unpack_state(x, num_cards, action_dim)
    for z, name in enumerate(ZONE_NAMES):
        assert torch.equal(presence[:, z], pieces[name])

    # 1-D input round-trips with single=True (downstream squeezes).
    presence1, numerics1, single1 = unpack_state_tokens(x[0], num_cards, action_dim)
    assert single1 is True
    assert presence1.shape == (1, len(ZONE_NAMES), num_cards)


def _build_token_model(registry: CardRegistry, **cfg_kwargs) -> tuple[PPOActorCritic, int, int]:
    cards = registry.card_names
    num_cards = registry.num_cards
    state_dim = get_state_size(cards)
    action_dim = get_action_space_size(cards)
    cfg = ModelConfig(token_features=True, **cfg_kwargs)
    model = PPOActorCritic(
        state_dim, action_dim, num_cards,
        model_config=cfg, card_registry=registry,
    )
    return model, state_dim, action_dim


@pytest.mark.parametrize("pool_type", ["sum", "attention"])
def test_token_features_forward_shape_and_finite(pool_type):
    reg = _toy_registry()
    model, state_dim, action_dim = _build_token_model(reg, pool_type=pool_type)

    x = torch.rand(4, state_dim) * 0.1
    logits, value = model(x)
    assert logits.shape == (4, action_dim)
    assert value.shape == (4,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("pool_type", ["sum", "attention"])
def test_token_features_empty_zones_no_nan(pool_type):
    reg = _toy_registry()
    model, state_dim, _ = _build_token_model(reg, pool_type=pool_type)
    # Zero state: all presence=0. Sum pool returns zeros; attention pool's
    # any_present gate must zero out the would-be-NaN softmax rows.
    x = torch.zeros(2, state_dim)
    logits, value = model(x)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_token_features_gradient_flows_through_static_path():
    reg = _toy_registry()
    model, state_dim, _ = _build_token_model(reg, pool_type="sum")

    x = torch.rand(2, state_dim) * 0.1
    logits, value = model(x)
    loss = logits.sum() + value.sum()
    loss.backward()

    # token_proj is the only path the static features can influence the loss
    # through, so its gradient must be present and finite.
    assert model.token_proj.weight.grad is not None
    assert torch.isfinite(model.token_proj.weight.grad).all()
    # Static feature buffer is non-learnable, but the slice fed into token_proj
    # must contribute non-zero gradient signal somewhere in the projection.
    assert model.token_proj.weight.grad.abs().sum() > 0


def test_token_features_fast_sum_pool_matches_cube_path():
    """The fast cube-free sum pool must produce numerically identical output
    to the slow ``_build_tokens`` + ``_sum_pool_tokens`` path on the same
    input. Math is exact (linear projection decomposes over concat), so
    differences should be at float32 ULP level only.
    """
    from src.ppo.ppo_actor_critic import _sum_pool_tokens

    reg = _toy_registry()
    model, state_dim, _ = _build_token_model(reg, pool_type="sum")
    model.eval()

    x = torch.rand(8, state_dim) * 2.0  # include presence>1 (stacked cards)
    presence, _, _ = __import__(
        "src.encoding.state_utils", fromlist=["unpack_state_tokens"],
    ).unpack_state_tokens(x, model.num_cards, model.action_dim)

    fast = model._token_features_sum_pool(presence)
    tokens = model._build_tokens(presence)
    slow = _sum_pool_tokens(tokens, presence)

    assert fast.shape == slow.shape
    assert torch.allclose(fast, slow, atol=1e-5, rtol=1e-5), (
        f"max abs diff: {(fast - slow).abs().max().item():.2e}"
    )


def test_token_features_fast_sum_pool_gradients_match_cube_path():
    """Gradients through the fast path must match the slow path at ULP
    precision. Validates that slicing token_proj.weight column-wise preserves
    autograd correctness."""
    from src.ppo.ppo_actor_critic import _sum_pool_tokens

    reg = _toy_registry()
    model_fast, state_dim, _ = _build_token_model(reg, pool_type="sum")
    model_slow, _, _ = _build_token_model(reg, pool_type="sum")
    # Tie weights so any difference is from the pool path, not init.
    model_slow.load_state_dict(model_fast.state_dict())

    x = torch.rand(4, state_dim) * 2.0
    presence, _, _ = __import__(
        "src.encoding.state_utils", fromlist=["unpack_state_tokens"],
    ).unpack_state_tokens(x, model_fast.num_cards, model_fast.action_dim)

    fast = model_fast._token_features_sum_pool(presence)
    fast.sum().backward()

    tokens = model_slow._build_tokens(presence)
    slow = _sum_pool_tokens(tokens, presence)
    slow.sum().backward()

    # token_proj is the only learnable parameter exercised by both pool paths.
    g_fast = model_fast.token_proj.weight.grad
    g_slow = model_slow.token_proj.weight.grad
    assert torch.allclose(g_fast, g_slow, atol=1e-5, rtol=1e-5), (
        f"max abs grad diff: {(g_fast - g_slow).abs().max().item():.2e}"
    )


def test_token_features_fast_sum_pool_handles_empty_presence():
    """When presence is all zero, the fast path must return zeros (not NaN)
    matching the slow path. Bias and W_pres terms drop to 0 because their
    coefficients (P_sum, P_sq) are 0; zone_proj term also zeros out."""
    reg = _toy_registry()
    model, state_dim, _ = _build_token_model(reg, pool_type="sum")
    model.eval()

    x = torch.zeros(3, state_dim)
    logits, value = model(x)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_token_features_requires_registry():
    cards = [f"C{i}" for i in range(5)]
    state_dim = get_state_size(cards)
    action_dim = get_action_space_size(cards)
    with pytest.raises(ValueError, match="requires a CardRegistry"):
        PPOActorCritic(
            state_dim, action_dim, len(cards),
            model_config=ModelConfig(token_features=True),
        )


def test_token_features_registry_size_mismatch_raises():
    reg = _toy_registry()
    cards_wrong = [f"C{i}" for i in range(reg.num_cards + 1)]
    state_dim = get_state_size(cards_wrong)
    action_dim = get_action_space_size(cards_wrong)
    with pytest.raises(ValueError, match="num_cards"):
        PPOActorCritic(
            state_dim, action_dim, len(cards_wrong),
            model_config=ModelConfig(token_features=True),
            card_registry=reg,
        )


def test_token_features_state_dict_incompatible_with_legacy():
    """A token_features=True checkpoint has new params (token_proj,
    card_features.features). Loading it into a token_features=False model
    must fail loudly so checkpoints can't be silently mis-loaded."""
    reg = _toy_registry()
    src_model, state_dim, action_dim = _build_token_model(reg, pool_type="sum")
    sd = src_model.state_dict()
    assert any("token_proj" in k for k in sd.keys())
    # card_features.features is registered non-persistent, so it must NOT
    # appear in the state_dict (it's derived from the registry at construction).
    assert not any("card_features" in k for k in sd.keys())

    legacy = PPOActorCritic(
        state_dim, action_dim, reg.num_cards,
        model_config=ModelConfig(token_features=False),
    )
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(sd)


def test_legacy_path_unchanged_when_token_features_false():
    """token_features=False must produce the same param shapes as before so
    existing V2/V3 checkpoints continue to load without modification."""
    cards = [f"C{i}" for i in range(10)]
    state_dim = get_state_size(cards)
    action_dim = get_action_space_size(cards)
    model = PPOActorCritic(state_dim, action_dim, len(cards))
    keys = set(model.state_dict().keys())
    assert not any("token_proj" in k for k in keys)
    assert not any("card_features" in k for k in keys)


def test_model_config_token_features_round_trip():
    cfg = ModelConfig(token_features=True)
    d = cfg.to_dict()
    assert d["token_features"] is True
    cfg2 = ModelConfig.from_dict(d)
    assert cfg2.token_features is True


def test_attention_pool_zone_permutation_does_not_change_pooling():
    """Within a zone, attention pooling must be permutation-invariant in the
    card axis (modulo presence). Shuffling card order with the same presence
    map yields the same pooled vector for that zone — true of any softmax+sum
    over a set."""
    from src.ppo.ppo_actor_critic import AttentionZonePooling
    Z, C, E = 3, 8, 4
    pool = AttentionZonePooling(Z, E)
    tokens = torch.randn(1, Z, C, E)
    presence = torch.zeros(1, Z, C)
    presence[0, 0, :] = torch.tensor([1.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0])

    out = pool(tokens, presence).view(1, Z, E)

    perm = torch.tensor([3, 7, 0, 5, 1, 2, 4, 6])
    tokens_perm = tokens.clone()
    tokens_perm[0, 0] = tokens[0, 0, perm]
    presence_perm = presence.clone()
    presence_perm[0, 0] = presence[0, 0, perm]

    out_perm = pool(tokens_perm, presence_perm).view(1, Z, E)
    assert torch.allclose(out[0, 0], out_perm[0, 0], atol=1e-6)


def test_v3_checkpoint_loads_when_token_features_false(tmp_path):
    """A V3 checkpoint (no token_features field) must still load into a
    legacy-mode model on V4 code without errors."""
    from src.config import save_checkpoint, load_checkpoint
    reg = _toy_registry()
    state_dim = get_state_size(reg.card_names)
    action_dim = get_action_space_size(reg.card_names)
    cfg = ModelConfig(token_features=False)
    model = PPOActorCritic(state_dim, action_dim, reg.num_cards, model_config=cfg)
    ckpt_path = tmp_path / "v3_legacy.pt"
    save_checkpoint(str(ckpt_path), model.state_dict(), model_config=cfg)

    import torch
    raw = torch.load(str(ckpt_path))
    raw["schema_version"] = 3
    raw["config"]["model"].pop("token_features", None)
    torch.save(raw, str(ckpt_path))

    ckpt = load_checkpoint(str(ckpt_path))
    fresh = PPOActorCritic(state_dim, action_dim, reg.num_cards, model_config=cfg)
    fresh.load_state_dict(ckpt["model_state_dict"])


def test_checkpoint_version_is_four():
    from src.config import CHECKPOINT_VERSION, COMPAT_CHECKPOINT_VERSIONS
    assert CHECKPOINT_VERSION == 4
    assert {2, 3, 4}.issubset(COMPAT_CHECKPOINT_VERSIONS)
