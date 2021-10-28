from typing import Callable, Sequence, Union
from msdm.core.problemclasses.mdp.tabularmdp import \
    TabularMarkovDecisionProcess, HashableState, HashableAction
from msdm.core.distributions import Distribution, DeterministicDistribution
from msdm.core.utils.funcutils import method_cache, cached_property
from msdm.core.problemclasses.mdp.quickmdp import QuickMDP

class QuickTabularMDP(QuickMDP,TabularMarkovDecisionProcess):
    def __init__(
        self,
        next_state_dist: Callable[[HashableState, HashableAction], Distribution[HashableState]]=None,
        *,
        reward: Union[float, Callable[[HashableState, HashableAction, HashableState], float]],
        actions: Union[Sequence[HashableAction], Callable[[HashableState], Sequence[HashableAction]]],
        initial_state_dist: Union[Distribution[HashableState], Callable[[], Distribution[HashableState]]]=None,
        is_terminal: Callable[[HashableState], bool],
        state_list : Sequence[HashableState] = None,
        action_list : Sequence[HashableAction]= None,
        # Deterministic variants.
        next_state: Callable[[HashableState, HashableAction], HashableState] = None,
        initial_state: HashableState=None,
        discount_rate=1.0
    ):
        super().__init__(
            next_state_dist = next_state_dist,
            reward = reward,
            actions = actions,
            initial_state_dist = initial_state_dist,
            is_terminal = is_terminal,
            next_state = next_state,
            initial_state = initial_state,
            discount_rate = discount_rate
        )
        self._state_list = state_list
        self._action_list = action_list

    @cached_property
    def state_list(self) -> Sequence[HashableState]:
        if self._state_list is not None:
            return self._state_list
        return super().state_list

    @cached_property
    def action_list(self) -> Sequence[HashableAction]:
        if self._action_list is not None:
            return self._action_list
        return super().action_list
