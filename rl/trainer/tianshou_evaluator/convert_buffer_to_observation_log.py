import pandas as pd
from tianshou.data import ReplayBuffer
from .api import ObservationLogger


def convert_buffer_to_observation_log(
        buffer: ReplayBuffer, observation_logger: ObservationLogger | None):

    df = pd.DataFrame()

    length = len(buffer)
    if observation_logger is not None:
        # additional information from the environment specific observation logger
        observation_logger.transform_buffer_to_dataframe(buffer, df)

    df["action"] = buffer.act
    df["reward"] = buffer.rew
    df["terminated"] = buffer.terminated
    df["truncated"] = buffer.truncated
    df["done"] = buffer.done

    info = buffer.info
    if hasattr(info, 'episode'):
        df['episode'] = info.episode
    else:
        # calculate the episode index from 'done' field
        shifted_done = df["done"].shift(1)
        df["episode"] = shifted_done.cumsum().fillna(0).astype("int")

    if hasattr(info, 'step_index'):
        df['index'] = info.step_index
    else:
        # create episode specific index
        df["index"] = df.groupby("episode").cumcount()

    # move columns
    df.insert(0, "episode", df.pop("episode"))
    df.insert(1, "index", df.pop("index"))

    if hasattr(info, 'agent_id'):
        df.insert(2, 'agent_id', info.agent_id)
    if hasattr(info, 'agent_step_index'):
        df.insert(3, 'agent_step_index', info.agent_step_index)

    df = df[:length]
    return df
