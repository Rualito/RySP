from abc import ABC, abstractmethod
from typing import Any


from qutip import ket, tensor, Qobj, propagator
from qutip.qip.operations import expand_operator
import numpy as np


# TODO: New class: QTerm
# Used to extend functionality for different backends
# Use inheritance of QTerm to implement new class
# add entries to qterm_tensor and qterm_expand_operator


__all__ =[
    'QTerm',
    'QTermQobj',
    'qterm_tensor',
    'qterm_expand_operator',
    'qterm_propagator'
] 

class QTerm(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """__init__ _summary_
        """
        pass
    @abstractmethod
    def tr(self):
        """
        tr _summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """
        return 0
    @abstractmethod
    def expm(self):
        """
        expm _summary_

        _extended_summary_
        """
        pass
    @abstractmethod
    def copy(self, other):
        """
        copy _summary_

        _extended_summary_

        Parameters
        ----------
        other : _type_
            _description_
        """
        pass
    @abstractmethod
    def get_data(self):
        """
        get_data _summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """
        return 0 
    @abstractmethod   
    def __add__(self, other):
        pass
    @abstractmethod
    def __radd__(self, other):
        pass
    @abstractmethod
    def __sub__(self, other):
        pass
    @abstractmethod
    def __rsub__(self, other):
        pass
    @abstractmethod
    def __mul__(self, other):
        pass
    @abstractmethod
    def __rmul__(self, other):
        pass
    @abstractmethod
    def __truediv__(self, other):
        pass
    @abstractmethod
    def __div__(self, other):
        pass
    @abstractmethod
    def __neg__(self, other):
        pass
    @abstractmethod
    def __getitem__(self, other):
        pass
    @abstractmethod
    def __pow__(self, other):
        pass
    @abstractmethod
    def __abs__(self, other):
        pass
    @abstractmethod
    def __str__(self, other):
        pass
    @abstractmethod
    def __repr__(self, other):
        pass
    
    @abstractmethod
    def dag(self):
        pass
    @abstractmethod
    def conj(self, other):
        pass
    @abstractmethod
    def norm(self, other):
        pass
    @abstractmethod
    def proj(self, other):
        pass
    @abstractmethod
    def full(self):
        pass
    @abstractmethod
    def __array__(self, *arg, **kwarg):
        pass
    @abstractmethod
    def overlap(self):
        pass

class QTermQobj(Qobj, QTerm):
    """QTermQobj _summary_

    Parameters
    ----------
    Qobj : _type_
        _description_
    QTerm : _type_
        _description_
    """
    pass


# Add more functionality for other QTerms
def qterm_tensor(*args:QTerm):
    """qterm_tensor _summary_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    if isinstance(args[0], Qobj) or ( type(args[0])==list and isinstance(args[0][0], Qobj) ): 
        return tensor(*args)
    
    raise NotImplementedError(f"Wrong type {type(args[0])=}")

def qterm_expand_operator(oper:QTerm, 
                          N:int,
                          targets:list[int],
                          dims:list[int]|None=None,
                          cyclic_permutation=False):
    """qterm_expand_operator _summary_

    Parameters
    ----------
    oper : QTerm
        _description_
    N : int
        _description_
    targets : list[int]
        _description_
    dims : list[int] | None, optional
        _description_, by default None
    cyclic_permutation : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    if isinstance(oper, Qobj):
        return expand_operator(oper, N, targets, dims, cyclic_permutation)
    raise NotImplementedError(f"Wrong type {oper=}")

def qterm_propagator(H: list[list[Any]]|list[Any]|QTerm, 
                     t: float|list[float]|np.ndarray, 
                     c_op_list: list, 
                     args: dict={}, 
                     options=None, unitary_mode:str='batch', progress_bar=None, **kwargs):
    """
    qterm_propagator _summary_

    _extended_summary_

    Parameters
    ----------
    H : list[list[Any]] | list[Any] | QTerm
        _description_
    t : float | list[float] | np.ndarray
        _description_
    c_op_list : list
        _description_
    args : dict, optional
        _description_, by default {}
    options : _type_, optional
        _description_, by default None
    unitary_mode : str, optional
        _description_, by default 'batch'
    progress_bar : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    if isinstance(H, Qobj) or len(H)>0 and isinstance(H[0], Qobj) or \
        len(H[0])>0 and isinstance(H[0][0], Qobj):# type: ignore
        return propagator(H, t, c_op_list, args, options=options, unitary_mode=unitary_mode, progress_bar=progress_bar, **kwargs)
    raise NotImplementedError(f"Wrong type {H=}")
    