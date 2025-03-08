from typing import Literal, Annotated
import numpy as np
import numpy.typing as npt

Vector3 = Annotated[npt.NDArray[np.float_], Literal[3]]
Vector3D = Vector3
Vector2 = Annotated[npt.NDArray[np.float_], Literal[2]]
Vector2D = Vector2
VectorNx2 = Annotated[npt.NDArray[np.float_], Literal['N', 2]]
VectorNx3 = Annotated[npt.NDArray[np.float_], Literal['N', 3]]
VectorN = npt.NDArray[np.float_]
VectorNxN = Annotated[npt.NDArray[np.float_], Literal['N', 'N']]
VectorNxNxN = Annotated[npt.NDArray[np.float_], Literal['N', 'N', 'N']]
