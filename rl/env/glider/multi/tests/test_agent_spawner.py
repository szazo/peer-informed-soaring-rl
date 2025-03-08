from ..agent_spawner import AgentSpawnParameters2, AgentSpawner2


def test_init_should_generate_possible_agents():

    # given
    pool_size = 2
    params = AgentSpawnParameters2(pool_size=pool_size)

    # when
    spawner = AgentSpawner2('testprefix', params)

    # then
    expected_possible_agents = ['testprefix0', 'testprefix1']
    assert spawner.possible_agents == expected_possible_agents


def test_should_spawn_multiple_if_required():
    # given
    params = AgentSpawnParameters2(pool_size=4,
                                   time_between_spawns_min_max_s=(0, 0),
                                   parallel_num_min_max=(2, 2))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)

    # when
    agent_ids = spawner.spawn(current_time_s=0., global_agent_count=0)

    # then
    assert agent_ids == ['agent0', 'agent1']


def test_should_not_spawn_more_agents_than_pool_size():
    # given
    params = AgentSpawnParameters2(pool_size=3,
                                   time_between_spawns_min_max_s=(0, 0),
                                   parallel_num_min_max=(4, 4))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)

    # when
    agent_ids = spawner.spawn(current_time_s=0., global_agent_count=0)

    # then
    assert agent_ids == ['agent0', 'agent1', 'agent2']


def test_should_only_spawn_one_when_there_is_time_between_spawns():
    # given
    params = AgentSpawnParameters2(pool_size=4,
                                   time_between_spawns_min_max_s=(10, 10),
                                   parallel_num_min_max=(2, 2))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)

    # when
    agent_ids = spawner.spawn(current_time_s=10., global_agent_count=0)

    # then
    assert agent_ids == ['agent0']


def test_should_not_spawn_if_time_not_reached():
    # given
    params = AgentSpawnParameters2(pool_size=4,
                                   time_between_spawns_min_max_s=(10, 10),
                                   parallel_num_min_max=(2, 2))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)
    spawner.spawn(current_time_s=10., global_agent_count=0)

    # when
    agent_ids = spawner.spawn(current_time_s=19., global_agent_count=0)

    # then
    assert agent_ids == []


def test_should_spawn_when_time_reached():
    # given
    params = AgentSpawnParameters2(pool_size=4,
                                   time_between_spawns_min_max_s=(10, 10),
                                   parallel_num_min_max=(2, 2))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)
    spawner.spawn(current_time_s=10., global_agent_count=0)

    # when
    agent_ids = spawner.spawn(current_time_s=20., global_agent_count=0)

    # then
    assert agent_ids == ['agent1']


def test_should_spawn_if_no_global_agent_but_required():
    # given
    params = AgentSpawnParameters2(pool_size=4,
                                   time_between_spawns_min_max_s=(10, 10),
                                   parallel_num_min_max=(2, 2),
                                   must_spawn_if_no_global_agent=True)
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)

    # when
    agent_ids = spawner.spawn(current_time_s=5., global_agent_count=0)

    # then
    assert agent_ids == ['agent0']


def test_should_spawn_after_killed():
    # given
    params = AgentSpawnParameters2(pool_size=4,
                                   time_between_spawns_min_max_s=(0, 0),
                                   parallel_num_min_max=(2, 2))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)
    spawner.spawn(current_time_s=0., global_agent_count=0)

    # when
    spawner.agent_killed('agent1')
    agent_ids = spawner.spawn(current_time_s=0., global_agent_count=0)

    # then
    assert agent_ids == ['agent2']


def test_should_not_spawn_if_initial_time_offset_is_not_reached():
    # given
    params = AgentSpawnParameters2(pool_size=4,
                                   time_between_spawns_min_max_s=(1, 1),
                                   parallel_num_min_max=(2, 2),
                                   initial_time_offset_s_min_max=(10, 10))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)

    # when
    agent_ids = spawner.spawn(current_time_s=10., global_agent_count=0)

    # then
    assert agent_ids == []


def test_should_spawn_when_initial_time_offset_reached():
    # given
    params = AgentSpawnParameters2(pool_size=4,
                                   time_between_spawns_min_max_s=(1, 1),
                                   parallel_num_min_max=(2, 2),
                                   initial_time_offset_s_min_max=(10, 10))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)

    # when
    agent_ids = spawner.spawn(current_time_s=11., global_agent_count=0)

    # then
    assert agent_ids == ['agent0']


def test_should_return_finished_when_exhausted():
    # given
    params = AgentSpawnParameters2(pool_size=3,
                                   time_between_spawns_min_max_s=(1, 1),
                                   parallel_num_min_max=(3, 3),
                                   initial_time_offset_s_min_max=(0, 0))
    spawner = AgentSpawner2('agent', params)
    spawner.reset(initial_time_s=0.)
    assert not spawner.is_finished

    # when
    agent_ids = spawner.spawn(current_time_s=1., global_agent_count=0)

    # then
    assert agent_ids == ['agent0']
    assert not spawner.is_finished

    # when
    agent_ids = spawner.spawn(current_time_s=2., global_agent_count=0)

    # then
    assert agent_ids == ['agent1']
    assert not spawner.is_finished

    # when
    agent_ids = spawner.spawn(current_time_s=3., global_agent_count=0)

    # then
    assert agent_ids == ['agent2']
    assert not spawner.is_finished

    # when
    spawner.agent_killed('agent0')
    spawner.agent_killed('agent2')

    # then
    assert not spawner.is_finished

    # when
    spawner.agent_killed('agent1')

    # then
    assert spawner.is_finished
