import unittest
from dataclasses import dataclass

from ptcg_abc.rl.phase5_adapters import GameMemory, LegalOptionAdapter, StateAdapter
from ptcg_abc.rl.phase5_encoder import (
    ACTION_FEATURE_NAMES,
    ENTITY_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    Phase5SymbolicEncoder,
)
from ptcg_abc.rl.phase5_policy import (
    AlphaStarTurnPolicy,
    Phase5PolicyUnavailable,
    TORCH_AVAILABLE,
    make_phase5_policy_config,
)


@dataclass
class FakeCardData:
    cardId: int
    name: str
    cardType: int = 0
    hp: int = 100
    ex: bool = False
    megaEx: bool = False


@dataclass
class FakeCard:
    id: int
    playerIndex: int = 0


@dataclass
class FakePokemon:
    id: int
    playerIndex: int
    hp: int = 80
    maxHp: int = 100
    energyCards: list | None = None
    tools: list | None = None
    preEvolution: list | None = None


@dataclass
class FakePlayer:
    active: list | None = None
    bench: list | None = None
    hand: list | None = None
    handCount: int = 0
    discard: list | None = None
    prize: list | None = None
    deckCount: int = 0


@dataclass
class FakeCurrent:
    yourIndex: int = 0
    turn: int = 3
    players: list | None = None
    energyAttached: bool = False
    supporterPlayed: bool = False
    stadium: list | None = None
    looking: list | None = None


@dataclass
class FakeOption:
    type: int
    index: int | None = None
    area: int | None = None
    playerIndex: int | None = None
    inPlayArea: int | None = None
    inPlayIndex: int | None = None
    attackId: int | None = None


@dataclass
class FakeSelect:
    type: int
    context: int
    minCount: int
    maxCount: int
    option: list[FakeOption]


@dataclass
class FakeObservation:
    select: FakeSelect | None
    current: FakeCurrent


def _observation() -> FakeObservation:
    players = [
        FakePlayer(
            active=[FakePokemon(100, 0, energyCards=[FakeCard(300)])],
            bench=[FakePokemon(101, 0)],
            hand=[FakeCard(200), FakeCard(201)],
            handCount=2,
            discard=[FakeCard(202)],
            prize=[None, None, None, None, None, None],
            deckCount=45,
        ),
        FakePlayer(
            active=[FakePokemon(110, 1)],
            bench=[],
            hand=None,
            handCount=5,
            discard=[FakeCard(210, 1)],
            prize=[None, None, None, None, None, None],
            deckCount=47,
        ),
    ]
    return FakeObservation(
        select=FakeSelect(
            type=0,
            context=0,
            minCount=1,
            maxCount=1,
            option=[
                FakeOption(type=7, index=0),
                FakeOption(type=13, attackId=42),
                FakeOption(type=14),
            ],
        ),
        current=FakeCurrent(
            players=players,
            stadium=[FakeCard(250)],
            looking=[],
        ),
    )


def _card_data() -> list[FakeCardData]:
    return [
        FakeCardData(100, "Active A", hp=100, ex=True),
        FakeCardData(101, "Bench A", hp=90),
        FakeCardData(110, "Active B", hp=100),
        FakeCardData(200, "Playable Item", cardType=1),
        FakeCardData(201, "Supporter", cardType=3),
        FakeCardData(202, "Discarded", cardType=1),
        FakeCardData(210, "Opponent Discarded", cardType=1),
        FakeCardData(250, "Stadium", cardType=4),
        FakeCardData(300, "Energy", cardType=5),
    ]


class Phase5AdapterEncoderTests(unittest.TestCase):
    def test_state_and_legal_option_adapters_preserve_canonical_fields(self):
        obs = _observation()
        state = StateAdapter(card_data=_card_data()).parse(obs)
        actions = LegalOptionAdapter(card_data=_card_data()).parse(obs)

        self.assertEqual(state.turn, 3)
        self.assertEqual(state.your_index, 0)
        self.assertEqual(state.us.hand_count, 2)
        self.assertEqual(state.opponent.hand_count, 5)
        self.assertTrue(any(entity.card_id == 100 for entity in state.entities))
        self.assertEqual([action.local_index for action in actions], [0, 1, 2])
        self.assertEqual(actions[0].selected_indices, (0,))
        self.assertEqual(actions[0].card_id, 200)
        self.assertTrue(actions[1].ends_turn)
        self.assertTrue(actions[2].ends_turn)

    def test_memory_belief_and_symbolic_encoder_emit_padded_tensors(self):
        obs = _observation()
        state = StateAdapter(card_data=_card_data()).parse(obs)
        actions = LegalOptionAdapter(card_data=_card_data()).parse(obs)
        memory = GameMemory()
        memory.observe(state)
        belief = memory.belief_state(
            state,
            own_deck_ids=[100, 101, 200, 201, 202, 300] * 4,
            opponent_deck_ids=[110, 210] * 30,
        )
        encoded = Phase5SymbolicEncoder(max_entities=12, max_actions=8).encode(
            state,
            actions,
            belief,
        )

        self.assertEqual(len(encoded.global_features), len(GLOBAL_FEATURE_NAMES))
        self.assertEqual(len(encoded.entity_features), 12)
        self.assertEqual(len(encoded.entity_features[0]), len(ENTITY_FEATURE_NAMES))
        self.assertEqual(len(encoded.legal_action_features), 8)
        self.assertEqual(len(encoded.legal_action_features[0]), len(ACTION_FEATURE_NAMES))
        self.assertEqual(encoded.legal_action_indices[:3], [0, 1, 2])
        self.assertEqual(sum(encoded.legal_action_mask), 3.0)
        self.assertGreater(belief.opponent_unknown_hand_count, 0)

        cross_owner_belief = memory.belief_state(
            state,
            own_deck_ids=[110, 110],
            opponent_deck_ids=[100, 100],
        )
        self.assertEqual(cross_owner_belief.own_deck_candidates.count(110), 2)
        self.assertEqual(cross_owner_belief.opponent_deck_candidates.count(100), 2)

    def test_phase5_policy_config_uses_encoder_dimensions(self):
        config = make_phase5_policy_config(
            global_dim=len(GLOBAL_FEATURE_NAMES),
            entity_dim=len(ENTITY_FEATURE_NAMES),
            action_dim=len(ACTION_FEATURE_NAMES),
            d_model=64,
        )

        self.assertEqual(config.global_dim, len(GLOBAL_FEATURE_NAMES))
        self.assertEqual(config.entity_dim, len(ENTITY_FEATURE_NAMES))
        self.assertEqual(config.action_dim, len(ACTION_FEATURE_NAMES))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
    def test_alphastar_turn_policy_scores_legal_actions(self):
        import torch

        obs = _observation()
        state = StateAdapter(card_data=_card_data()).parse(obs)
        actions = LegalOptionAdapter(card_data=_card_data()).parse(obs)
        encoded = Phase5SymbolicEncoder(max_entities=12, max_actions=8).encode(state, actions)
        config = make_phase5_policy_config(
            global_dim=len(encoded.global_features),
            entity_dim=len(encoded.entity_features[0]),
            action_dim=len(encoded.legal_action_features[0]),
            d_model=64,
        )
        model = AlphaStarTurnPolicy(config)
        output = model(
            torch.tensor([encoded.global_features], dtype=torch.float32),
            torch.tensor([encoded.entity_features], dtype=torch.float32),
            torch.tensor([encoded.entity_mask], dtype=torch.float32),
            torch.tensor([encoded.legal_action_features], dtype=torch.float32),
            torch.tensor([encoded.legal_action_mask], dtype=torch.float32),
            torch.tensor([[encoded.legal_action_features[0]]], dtype=torch.float32),
        )

        self.assertEqual(tuple(output["action_logits"].shape), (1, 8))
        self.assertEqual(tuple(output["action_q"].shape), (1, 8))
        self.assertEqual(tuple(output["tactical_score"].shape), (1, 8))
        self.assertEqual(tuple(output["state_value"].shape), (1,))

    def test_policy_module_reports_missing_torch_cleanly(self):
        if TORCH_AVAILABLE:
            self.skipTest("PyTorch is installed in this environment.")
        config = make_phase5_policy_config(
            global_dim=len(GLOBAL_FEATURE_NAMES),
            entity_dim=len(ENTITY_FEATURE_NAMES),
            action_dim=len(ACTION_FEATURE_NAMES),
        )
        with self.assertRaises(Phase5PolicyUnavailable):
            AlphaStarTurnPolicy(config)


if __name__ == "__main__":
    unittest.main()
